"""2D-signals -> BEV footprint / free-space / hidden-ground pseudo-labels.

An alternative to the 3D-DEM-render footprint generator: instead of rendering the
global DEM into each camera, fuse **per-frame 2D perception** (panoptic segmentation
+ metric depth) into the BEV, using the gravity-aligned ground DEM only as a *ray
target*. Three ground states come out, IMDF-ready:

  FREE       -- walkable ground, directly observed
  FOOTPRINT  -- ground occupied by a vertical object's *base* (not walkable)
  HIDDEN     -- ground that WAS in frustum but was never observed as walkable: something
                stood in front of it (walk-under / occluded). Derived from the visibility
                denominator, not from the shape of the free-space blob

Per frame (poses + PINHOLE intrinsics from COLMAP):
  1. panoptic seg (Mask2Former, MIT) -> per-pixel taxonomy Klass + Role
  2. metric depth (Depth-Anything-V2-Small, Apache; scale+shift fit to the frame's SfM
     points, exactly as script/depth/mono_depth.py)
  3. FREE   : back-project Role.GROUND pixels with metric depth -> BEV free votes
  4. FOOTPRINT: for every image column with an obstacle, take the *bottom-most* obstacle
     pixel (its ground-contact) and intersect that camera ray with the ground DEM
     (NOT mono-depth -- the object base is exactly where mono's vertical-structure noise
     is worst; ray-DEM uses only calibration + the SfM-solid DEM). Canopy over a path is
     high up, so its column's bottom pixel is the trunk base -> footprint lands on the
     trunk, not the whole canopy. This is the fix for "canopy marks everything occupied".
  5. DEPTH CONTACT GATE (--contact-depth-tol): step 4's "bottom-most obstacle pixel = ground
     contact" is an assumption, and it fails whenever a mask's bottom abuts ground that is
     really far behind it -- a bench seat or a canopy silhouetted against open lawn. Compare
     the measured mono depth at that pixel against the range at which the ray meets the DEM:
     equal if the object stands there, much nearer if it floats. Note this does not contradict
     step 4 -- mono depth never *places* a contact here, it only *vetoes* one. A relative
     comparison at a single pixel is what mono depth is reliable for; metric placement at an
     object edge is what it is not.

Fusion is by RATIO, not raw count. Each frame also rasterises which cells had their ground in
frustum at all (`visible_cells`), giving the denominator the raster never had: three votes
means something very different from 3 observations (unanimous) than from 40 (7.5% = noise).
Cells additionally need votes from >= --min-bearings distinct camera bearings, because ten
votes from one viewpoint are ONE correlated observation -- counting them ten times manufactures
confidence from a single mistake, which is what drew the radial BEV streaks. HIDDEN then falls
out geometrically: in frustum, yet never seen free.

This is stricter than the old pixel-count rule, and honestly so: at 40 frames the previous
2.3% FOOTPRINT was largely single-view claims counted once per pixel (1103 of 1196 voted cells
had exactly one bearing). Multi-view evidence scales with sampling -- 40 -> 150 frames takes
visibility 49.7% -> 72.0% (median 2 -> 4 views) and FOOTPRINT 0.2% -> 1.6%, now actually
corroborated.

Run (from repo root):
  uv run --extra segmentation --with /home/play/Code/lingbot-vision \
    python script/backbone/footprint2d.py --session maguro-park-after-itchy --num 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPT_ROOT = _HERE.parent
sys.path.insert(0, str(_SCRIPT_ROOT))
sys.path.insert(0, str(_SCRIPT_ROOT / "semantic_bev"))
sys.path.insert(0, str(_SCRIPT_ROOT / "depth"))

from lar_session import Session                      # noqa: E402
from colmap_io import read_model, qvec2rotmat        # noqa: E402
from geometry import from_reconstruction, gravity_up, align_rotation, build_dem  # noqa: E402
from taxonomy import Klass, Role, role_of, color_lut, by_klass  # noqa: E402
from mono_depth import sparse_samples, fit_metric    # noqa: E402

# BEV ground states
UNKNOWN, FREE, FOOTPRINT, HIDDEN = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# panoptic segmentation -> per-pixel taxonomy Klass (COCO things+stuff, MIT)
# ---------------------------------------------------------------------------
class SegKlass:
    """Mask2Former -> (klass_map, instance_map) in our taxonomy (keyword remap).

    mode='semantic' (default): dense class only, instance_map all zeros. Uses the
      large ADE model already cached by the linprobe runs -- no download. Base-contour
      footprints only need the obstacle *class* mask, so this is enough for v1.
    mode='panoptic': also returns 'thing' instance ids (COCO). Enables per-object
      footprint polygons, but trees are 'stuff' in both COCO/ADE (no per-tree ids) and
      the weights are a fresh ~850 MB download."""

    def __init__(self, model_name: str, mode: str = "semantic", device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        from taxonomy import keyword_klass

        self.torch = torch
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device).eval()
        id2label = self.model.config.id2label
        n = max(int(i) for i in id2label) + 1
        self.lut = np.zeros(n, dtype=np.uint8)
        for i, name in id2label.items():
            self.lut[int(i)] = int(keyword_klass(name))

    def __call__(self, image_bgr: np.ndarray):
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        if self.mode == "semantic":
            seg = self.processor.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0]
            return self.lut[seg.cpu().numpy().astype(np.int64)].astype(np.uint8), np.zeros((H, W), np.int32)
        res = self.processor.post_process_panoptic_segmentation(out, target_sizes=[(H, W)])[0]
        seg = res["segmentation"].cpu().numpy().astype(np.int64)  # (H,W) segment id, -1 = void
        klass = np.zeros((H, W), np.uint8)
        inst = np.zeros((H, W), np.int32)
        for s in res["segments_info"]:
            m = seg == s["id"]
            klass[m] = self.lut[int(s["label_id"])]
            inst[m] = int(s["id"])
        return klass, inst


# ---------------------------------------------------------------------------
# open-vocabulary instances: Grounding DINO boxes -> SAM masks
# ---------------------------------------------------------------------------
class GroundedSAM:
    """(masks, boxes, labels, scores) for one BGR image, all from cached weights."""

    def __init__(self, det_model: str, sam_model: str, prompt: str,
                 box_th: float, text_th: float, device: str | None = None):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  SamModel, SamProcessor)
        self.torch = torch
        self.prompt = prompt
        self.box_th, self.text_th = box_th, text_th
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dproc = AutoProcessor.from_pretrained(det_model)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(det_model).to(self.device).eval()
        self.sproc = SamProcessor.from_pretrained(sam_model)
        self.sam = SamModel.from_pretrained(sam_model).to(self.device).eval()

    def __call__(self, bgr: np.ndarray):
        from PIL import Image as PILImage
        torch = self.torch
        pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = self.dproc(images=pil, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.det(**inp)
        res = self.dproc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.box_th, text_threshold=self.text_th,
            target_sizes=[bgr.shape[:2]])[0]
        boxes = res["boxes"].cpu().numpy()
        if not len(boxes):
            return np.zeros((0, *bgr.shape[:2]), bool), boxes, [], np.zeros(0, np.float32)
        labels = [str(x) for x in res.get("text_labels", res["labels"])]
        scores = res["scores"].cpu().numpy().astype(np.float32)
        si = self.sproc(pil, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        with torch.no_grad():
            so = self.sam(**si, multimask_output=False)
        masks = self.sproc.image_processor.post_process_masks(
            so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu())[0]
        return masks[:, 0].numpy().astype(bool), boxes, labels, scores



def snap_label(raw: str, prompt: str) -> str:
    """Grounding DINO returns BPE-merged labels -- '##nding machine kiosk' for two adjacent
    prompt phrases. The '##' is a WordPiece continuation marker and the merge crosses phrase
    boundaries, so the raw string names no real class. Snap it to the prompt phrase sharing the
    most words; cosmetic while every rescue maps to one Klass, but load-bearing the moment
    per-label classes are wanted."""
    toks = set(raw.replace("##", "").split())
    best, hits = raw.strip(), 0
    for phrase in (x.strip() for x in prompt.split(".")):
        if not phrase:
            continue
        n = len(toks & set(phrase.split()))
        if n > hits:
            best, hits = phrase, n
    return best


# ---------------------------------------------------------------------------
# metric depth (DA2-small, SfM-scaled) -- same recipe as mono_depth.py
# ---------------------------------------------------------------------------
class MonoDepth:
    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=model_name,
                             device=0 if torch.cuda.is_available() else -1)

    def metric(self, bgr, recon, img, factor):
        """Dense per-pixel metric z-depth at 1/factor resolution, or None if unfittable."""
        from PIL import Image as PILImage
        H, W = bgr.shape[0] // factor, bgr.shape[1] // factor
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)
        disp = self.pipe(PILImage.fromarray(rgb))["predicted_depth"].squeeze().float().cpu().numpy()
        if disp.shape != (H, W):
            disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR)
        scale = W / recon.cameras[img.camera_id].width
        px, py, z = sparse_samples(recon, img, scale)
        if not len(z):
            return None, (H, W)
        ix = np.clip(np.round(px).astype(int), 0, W - 1)
        iy = np.clip(np.round(py).astype(int), 0, H - 1)
        ab = fit_metric(disp[iy, ix], z)
        if ab is None:
            return None, (H, W)
        a, b = ab
        denom = a * disp + b
        depth = np.where(denom > 1e-6, 1.0 / denom, 0.0).astype(np.float32)
        depth[(depth <= 0) | ~np.isfinite(depth)] = 0.0
        return depth, (H, W)


# ---------------------------------------------------------------------------
# geometry helpers (all in the gravity-aligned local frame of the GroundField)
# ---------------------------------------------------------------------------
def camera_local(img, gf):
    """Return (C_local (3,), Rwc_T_local (3,3)) so that a pixel's cam-dir d_cam maps to
    a local-frame ray direction via  dir_local = Rwc_T_local @ d_cam."""
    Rwc = qvec2rotmat(img.qvec)              # world->cam
    C_world = -Rwc.T @ img.tvec              # camera centre in world
    C_local = gf.R @ C_world
    Rwc_T_local = gf.R @ Rwc.T               # cam-dir -> local-frame dir
    return C_local, Rwc_T_local


def pixel_dirs(px, py, cam, factor):
    """PINHOLE cam-space ray dirs (unnormalised, z=1) for pixel arrays at 1/factor res."""
    fx, fy, cx, cy = cam.params[:4]
    fx, fy, cx, cy = fx / factor, fy / factor, cx / factor, cy / factor
    return np.stack([(px - cx) / fx, (py - cy) / fy, np.ones_like(px)], axis=1)  # (M,3)


def ray_dem_intersect(C_local, dirs_local, gf, z_near=0.4, z_far=70.0, dz=0.2,
                      max_range=20.0):
    """Intersect local-frame rays with the ground DEM height field.

    Marches cam-depth z; a hit is the first sign change of (ray_height - dem(ray_uv)).
    Returns (uv (M,2), ok (M,) bool, zt (M,) float). Rejects hits outside the grid,
    off-coverage, or beyond ``max_range`` m (far base-contacts are unreliable: ray-DEM at
    range + thin DEM coverage scatters, so we trust only nearby contacts and let other
    frames cover the rest).

    ``zt`` is the **camera z-depth** of the hit, which is what makes ``depth_contact_gate``
    possible: ``pixel_dirs`` returns z=1 directions and rotating into the local frame
    preserves the camera-frame z-component, so ts (hence zt) is z-depth in exactly the same
    convention ``MonoDepth.metric`` returns -- the two are directly comparable, no
    conversion."""
    ts = np.arange(z_near, z_far, dz)                       # (K,)
    P = C_local[None, None, :] + ts[None, :, None] * dirs_local[:, None, :]  # (M,K,3)
    uv = P[:, :, :2].reshape(-1, 2)
    ground = gf.height_at(uv).reshape(P.shape[0], P.shape[1])
    resid = P[:, :, 2] - ground                            # + above ground, - below
    below = resid < 0
    ok = below.any(1)
    k = np.argmax(below, axis=1)                           # first below-ground step
    k = np.clip(k, 1, len(ts) - 1)
    # linear interp between k-1 (above) and k (below)
    r0 = resid[np.arange(len(k)), k - 1]
    r1 = resid[np.arange(len(k)), k]
    frac = r0 / np.maximum(r0 - r1, 1e-6)
    zt = ts[k - 1] + frac * dz
    hit = C_local[None, :] + zt[:, None] * dirs_local      # (M,3)
    hit_uv = hit[:, :2]
    # keep only in-grid, covered cells, within trustworthy range
    cells = gf.cell_of(hit_uv)
    cov = gf.coverage.reshape(-1)[cells]
    return hit_uv, ok & cov & (zt <= max_range), zt


def depth_contact_gate(dmap, rows, cols, zt, ok, tol: float, log=None, tag: str = ""):
    """Reject 2D-adjacent-but-not-3D-touching ground contacts using measured depth.

    The bottom-most-obstacle-pixel heuristic assumes an object's silhouette bottom is where
    it meets the ground. The semantic canopy guard (``footprint_instances.contact_columns``)
    only asks "are the pixels below me ground?", which is a **2D** test and passes exactly in
    the failure case: a bench seat or canopy silhouetted against open lawn ten metres beyond.
    The pixels below *are* ground -- just ground that is far behind the object -- so the
    ray-DEM hit lands metres past it.

    The geometric test: if the object really stands at the contact, its measured depth equals
    the range at which that ray meets the ground. Floating/occluding -> the object is much
    *nearer* than the ground its ray eventually hits, so ``d_obj << zt``.

    Compared against the DEM hit rather than the neighbouring ground pixel on purpose. Mono
    depth is smoothed across object boundaries (the discontinuity bleeds over several pixels),
    so a boundary-crossing comparison washes out the very jump it looks for, and costs two
    noisy samples. This is one noisy sample against the *trusted* pose+DEM.

    The tolerance is **relative** (|d-zt| / zt): grazing distant rays otherwise fail
    spuriously -- the same lesson the depression gate taught in ``base_points.py``.

    It is also measured against the frame's **median depth ratio**, not against 1.0, and that
    matters more than it sounds. ``MonoDepth.metric`` fits scale+shift per frame from that
    frame's SfM points; when the fit is poor the whole depth map is off by a constant factor.
    Measured on maguro-park frame 0: d_mono median 1.31 m against z_dem 3.04 m, a systematic
    2.3x error -- an absolute test keeps 0.4% of its contacts and silently deletes the frame,
    while a median-relative test keeps 72%. Healthy frames sit at ratio 0.91-0.93, so
    normalising costs them nothing. This is the same trick as base_points' relative ground
    datum: absorb the systematic per-frame bias, test only the outlier.

    The assumption normalising buys into is that most bottom-most-obstacle pixels in a frame
    are genuine contacts, so the median tracks the true scale. That holds here because ground
    is everywhere and floating silhouettes are the minority -- and the ratio is logged so a
    frame whose median is wildly off (a depth-fit failure worth fixing at the source) stays
    visible rather than being quietly normalised away.

    Pixels with no valid mono depth are **kept**, not dropped: absence of evidence isn't
    evidence of floating, and punishing them would silently couple footprint recall to mono
    coverage. Their count is reported so the blind spot stays visible.
    """
    if tol <= 0 or dmap is None:
        return ok
    d_obj = dmap[rows, cols]
    have = d_obj > 0
    ref = ok & have
    # per-frame scale reference; too few samples to trust a median -> fall back to absolute
    if int(ref.sum()) >= 20:
        med = float(np.median(d_obj[ref] / np.maximum(zt[ref], 1e-6)))
        if not np.isfinite(med) or med <= 1e-3:
            med = 1.0
    else:
        med = 1.0
    agree = np.abs(d_obj - med * zt) <= tol * med * np.maximum(zt, 1e-6)
    gated = ok & (~have | agree)
    if log is not None:
        n0 = int(ok.sum())
        if n0:
            warn = "  << depth scale suspect" if not 0.7 <= med <= 1.4 else ""
            log(f"    depth gate{tag}: {n0} -> {int(gated.sum())} contacts "
                f"({int((ok & have & ~agree).sum())} floating, "
                f"{int((ok & ~have).sum())} no-depth kept, "
                f"frame d/z median {med:.3f}){warn}")
    return gated



# Multi-view evidence: 32 bearing bins packed into a uint32 per cell.
NBEARINGS = 32
_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def popcount32(a: np.ndarray) -> np.ndarray:
    """Number of distinct bearing bins set per cell. uint8-LUT so it works on any numpy."""
    return _POP8[a.astype(np.uint32).view(np.uint8).reshape(-1, 4)].sum(1).astype(np.int32)


def bearing_bits(cell_uv: np.ndarray, C_local: np.ndarray) -> np.ndarray:
    """Quantised camera->cell bearing as a one-hot uint32.

    Ten votes from one viewpoint are ONE piece of evidence, not ten: errors along a ray are
    correlated, so counting them ten times manufactures confidence out of a single mistake --
    which is exactly what draws the radial streaks in the BEV. Counting *distinct bearings*
    instead is the trick footprint_instances already uses for its polygons.
    """
    ang = np.degrees(np.arctan2(cell_uv[:, 1] - C_local[1], cell_uv[:, 0] - C_local[0]))
    b = ((ang % 360.0) * (NBEARINGS / 360.0)).astype(np.int64) % NBEARINGS
    return (np.uint32(1) << b.astype(np.uint32))


def visible_cells(C_local, RwcT, cam, gf, H, W, stride, factor, z_far, dz, max_range):
    """Cells whose GROUND is geometrically in frustum this frame, occlusion ignored.

    This is the denominator the raster never had. ``ray_dem_intersect`` hits the ground height
    field only, so a ray through a tree trunk still lands on the ground behind it -- which is
    the point: it separates "this cell's ground was never in view" (UNKNOWN) from "it was in
    view but something stood in front of it" (HIDDEN). That distinction used to be
    reverse-engineered with a morphological close; here it falls straight out of the geometry.
    """
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d_cam = pixel_dirs(xs.ravel().astype(np.float64), ys.ravel().astype(np.float64), cam, factor)
    dirs = (RwcT @ d_cam.T).T
    hit_uv, ok, _ = ray_dem_intersect(C_local, dirs, gf, z_far=z_far, dz=dz, max_range=max_range)
    if not ok.any():
        return np.zeros(0, np.int64), np.zeros((0, 2))
    uv = hit_uv[ok]
    cells = gf.cell_of(uv)
    keep = np.unique(cells, return_index=True)[1]        # one vote per cell per frame
    return cells[keep], uv[keep]


# ---------------------------------------------------------------------------
def sample_frames(recon, num):
    imgs = sorted(recon.images.values(), key=lambda im: im.name)
    idx = np.linspace(0, len(imgs) - 1, num).round().astype(int)
    return [imgs[i] for i in sorted(set(idx))]


# ---------------------------------------------------------------------------
# ground DEM: sparse COLMAP (default) or dense mono-depth fusion
# ---------------------------------------------------------------------------
def _pinhole(cam):
    """(fx, fy, cx, cy) from PINHOLE / SIMPLE_PINHOLE / SIMPLE_RADIAL."""
    p = cam.params
    if cam.model == "PINHOLE":
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    return float(p[0]), float(p[0]), float(p[1]), float(p[2])  # SIMPLE_* : one focal


def _backproject_mono(recon, depth_dir, stride, voxel, depth_max, log):
    """Fused world point cloud from every frame's metric mono-depth map (voxel-deduped).

    Same depth contract as footprint_labels/depth_backproject: ``<stem>.npy`` float32 (H,W)
    metric z-depth, intrinsics scaled to depth resolution. Far/sky mono depth explodes
    (1/(a·disp+b)) so cap it at ``depth_max`` m before fusing."""
    depth_dir = Path(depth_dir)
    all_pos, n, raw = [], 0, 0
    for im in recon.images.values():
        dp = depth_dir / f"{Path(im.name).stem}.npy"
        if not dp.exists():
            continue
        depth = np.load(dp).astype(np.float32)
        h, w = depth.shape
        cam = recon.cameras[im.camera_id]
        fx, fy, cx, cy = _pinhole(cam)
        sx, sy = w / cam.width, h / cam.height
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        ys, xs = np.arange(0, h, stride), np.arange(0, w, stride)
        gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
        d = depth[gy, gx]
        ok = (d > 0) & np.isfinite(d)
        if depth_max > 0:
            ok &= d < depth_max
        if not ok.any():
            continue
        u, v, dd = gx[ok].astype(np.float32), gy[ok].astype(np.float32), d[ok]
        cam_pts = np.stack([(u - cx) / fx * dd, (v - cy) / fy * dd, dd], axis=1)
        R = qvec2rotmat(im.qvec)
        all_pos.append((cam_pts @ R + (-R.T @ im.tvec)).astype(np.float32))
        n += 1; raw += int(ok.sum())
    if not all_pos:
        raise SystemExit(f"no mono depth maps found in {depth_dir}")
    pos = np.concatenate(all_pos)
    vox = np.floor(pos / voxel).astype(np.int64)
    _, inv = np.unique(vox, axis=0, return_inverse=True)
    sums = np.zeros((int(inv.max()) + 1, 3), np.float64)
    np.add.at(sums, inv, pos)
    positions = (sums / np.bincount(inv)[:, None]).astype(np.float32)
    log(f"  mono depth: {raw} pts from {n} maps -> {len(positions)} voxels @ {voxel} m")
    return positions


def mono_dem_field(recon, depth_dir, cell_size, stride, voxel, depth_max, log):
    """A GroundField whose DEM surface is fused mono depth (dense) instead of sparse COLMAP.

    Gravity frame still comes from the cameras, so it lines up with the COLMAP obstacle
    geometry footprint2d votes from. Only the DEM *surface/coverage* changes."""
    up = gravity_up(np.array([im.qvec for im in recon.images.values()]))
    R = align_rotation(up)
    world = _backproject_mono(recon, depth_dir, stride, voxel, depth_max, log)
    local = (R @ world.T).T
    return build_dem(local[:, :2], local[:, 2], R, up, cell_size=cell_size, log=log)


def run(args):
    s = Session(args.session)
    model = args.model or str(s.colmap_model)   # need TRACKS for depth scale-fit -> raw model
    images_dir = Path(args.images or str(s.images))
    out = Path(args.out or (s.root / "output" / f"{args.session}-footprint2d"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[footprint2d] model={model}\n  images={images_dir}\n  out={out}")
    recon = read_model(model)
    if args.dem_source == "mono":
        depth_dir = args.depth_dir or str(s.depth_dir("mono"))
        print(f"  DEM source: mono depth <- {depth_dir}")
        gf = mono_dem_field(recon, depth_dir, args.cell_size, args.depth_stride,
                            args.depth_voxel, args.depth_max, log=print)
    else:
        gf, *_ = from_reconstruction(recon, cell_size=args.cell_size)
    rows, cols = gf.dem.shape
    ncells = rows * cols

    pan = SegKlass(args.seg_model, mode=args.seg_mode)
    depth = MonoDepth(args.depth_model)
    gsam = (GroundedSAM(args.det_model, args.sam_model, args.rescue_prompt,
                        args.rescue_box_th, args.rescue_text_th)
            if args.rescue_prompt.strip() else None)
    rescue_klass = int(Klass[args.rescue_klass])
    rescued: dict[str, int] = {}

    # Frame counts, not pixel counts: a ratio against obs_ct is only meaningful if numerator
    # and denominator are both "how many frames".
    obs_ct = np.zeros(ncells, np.int32)
    free_ct = np.zeros(ncells, np.int32)
    foot_ct = np.zeros(ncells, np.int32)
    free_bear = np.zeros(ncells, np.uint32)
    foot_bear = np.zeros(ncells, np.uint32)
    struct_ct = np.zeros((ncells, int(max(Klass)) + 1), np.int32)  # per-cell obstacle class votes

    frames = sample_frames(recon, args.num)
    print(f"  {len(frames)} frames, grid {cols}x{rows} @ {args.cell_size} m")
    if args.contact_depth_tol > 0:
        print(f"  depth contact gate: |d_mono - z_dem| <= {args.contact_depth_tol:.0%} of z_dem")
    if gsam is not None:
        print(f"  obstacle rescue -> {args.rescue_klass}: {args.rescue_prompt}")
    ndbg = 0
    gate_pre = gate_post = 0
    for fi, img in enumerate(frames):
        bgr = cv2.imread(str(images_dir / img.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        dmap, (H, W) = depth.metric(bgr, recon, img, args.data_factor)
        if dmap is None:
            continue
        small = cv2.resize(bgr, (W, H))
        klass, inst = pan(small)
        roles = np.vectorize(lambda k: int(role_of(int(k))))(np.arange(int(max(Klass)) + 1))
        role_map = roles[klass]
        cam = recon.cameras[img.camera_id]
        C_local, RwcT = camera_local(img, gf)
        # The denominator must be a SUPERSET of every numerator or ratios exceed 1 and cells
        # get disqualified for "not being observed" in the very frame that observed them. The
        # raycast alone is not: it is coverage-gated and capped at --obs-max-range, while FREE
        # back-projects mono depth with neither restriction. So seed obs with the raycast and
        # union in whatever this frame actually voted -- a cell seen as ground was, tautologically,
        # in view.
        ocells, _ = visible_cells(C_local, RwcT, cam, gf, H, W, args.obs_stride,
                                  args.data_factor, args.z_far, args.dz, args.obs_max_range)
        frame_obs = [ocells]

        # ---- FREE: ground pixels back-projected with metric depth ----
        gmask = (role_map == int(Role.GROUND)) & (dmap > 0)
        gy, gx = np.where(gmask)
        if len(gx) > args.max_ground:                       # subsample for speed
            sel = np.random.default_rng(fi).choice(len(gx), args.max_ground, replace=False)
            gy, gx = gy[sel], gx[sel]
        if len(gx):
            d_cam = pixel_dirs(gx.astype(np.float64), gy.astype(np.float64), cam, args.data_factor)
            dir_local = (RwcT @ d_cam.T).T
            pts = C_local[None, :] + dmap[gy, gx][:, None] * dir_local
            cells = gf.cell_of(pts[:, :2])
            uniq, ui = np.unique(cells, return_index=True)
            np.add.at(free_ct, uniq, 1)
            np.bitwise_or.at(free_bear, uniq, bearing_bits(pts[ui, :2], C_local))
            frame_obs.append(uniq)

        # ---- RESCUE: open-vocab masks for solid objects the closed-set seg calls UNKNOWN ----
        # ADE-150 has no "vending machine", so Mask2Former labels one UNKNOWN -> Role.IGNORE and
        # a large solid obstacle contributes NO footprint at all. Measured on maguro-park frame
        # 8, the vending-machine region is 100% UNKNOWN/IGNORE: the machines are invisible to
        # the occupancy map. For a navigation map that is the dangerous direction to fail in --
        # free space asserted where something solid stands. Grounding DINO + SAM fill the hole
        # from a text prompt, since the long tail of park objects can never be enumerated in a
        # closed label set. Rescued pixels never overwrite a class the segmenter was confident
        # about; they only fill IGNORE/UNKNOWN gaps.
        if gsam is not None:
            rmasks, _, rlabels, rscores = gsam(small)
            # Highest score first: rescue is order-dependent (each mask only fills what earlier
            # ones left), so without a fixed order the result depends on detector output order.
            for i in np.argsort(-np.asarray(rscores)) if len(rscores) else []:
                rm = rmasks[i]
                # A single park object is not a third of the frame. Grounding DINO happily
                # returns such boxes ('playground equipment' covering 192k px = 28% of frame 8,
                # i.e. the vending machines and everything around them); letting one through
                # blankets the BEV with phantom occupancy, the same over-blocking this pipeline
                # rejected mono depth for.
                if rm.mean() > args.rescue_max_frac:
                    continue
                fill = rm & (role_map != int(Role.OBSTACLE))
                if int(fill.sum()) < args.rescue_min_px:
                    continue
                klass[fill] = rescue_klass
                role_map[fill] = int(Role.OBSTACLE)
                lab = snap_label(rlabels[i], args.rescue_prompt)
                rescued[lab] = rescued.get(lab, 0) + 1

        # ---- FOOTPRINT: bottom-most obstacle pixel per column, ray-DEM ----
        obst = (role_map == int(Role.OBSTACLE))
        base_klass = None
        if obst.any():
            has = obst.any(0)
            base_row = (obst * np.arange(H)[:, None]).argmax(0)   # bottom-most True row/col
            bx = np.where(has)[0]
            by = base_row[bx]
            base_klass = klass[by, bx]
            d_cam = pixel_dirs(bx.astype(np.float64), by.astype(np.float64), cam, args.data_factor)
            dir_local = (RwcT @ d_cam.T).T
            hit_uv, ok, zt = ray_dem_intersect(C_local, dir_local, gf,
                                               z_far=args.z_far, dz=args.dz,
                                               max_range=args.max_range)
            n_pre = int(ok.sum())
            ok = depth_contact_gate(dmap, by, bx, zt, ok, args.contact_depth_tol,
                                    log=(print if fi < args.debug_frames else None))
            gate_pre += n_pre
            gate_post += int(ok.sum())
            if ok.any():
                cells = gf.cell_of(hit_uv[ok])
                uniq, ui = np.unique(cells, return_index=True)
                np.add.at(foot_ct, uniq, 1)
                np.bitwise_or.at(foot_bear, uniq, bearing_bits(hit_uv[ok][ui], C_local))
                np.add.at(struct_ct, (cells, base_klass[ok]), 1)   # class votes stay per-pixel
                frame_obs.append(uniq)

        obs_ct[np.unique(np.concatenate(frame_obs))] += 1

        # ---- per-frame debug overlay (first few) ----
        if ndbg < args.debug_frames:
            dbg = small.copy()
            dbg[gmask] = (0.5 * dbg[gmask] + np.array([0, 120, 0])).astype(np.uint8)
            if base_klass is not None:
                # red = contact kept, magenta = rejected by the depth gate (2D-adjacent to
                # ground but not touching it in 3D). Eyeballing these is how you tune the tol.
                for x, y, keep in zip(bx, by, ok):
                    cv2.circle(dbg, (int(x), int(y)), 2,
                               (0, 0, 255) if keep else (255, 0, 255), -1)
            cv2.imwrite(str(out / f"frame_{ndbg}_{Path(img.name).stem}.png"), dbg)
            ndbg += 1
        if (fi + 1) % 10 == 0:
            print(f"  {fi + 1}/{len(frames)}  free~{int(free_ct.sum())} foot~{int(foot_ct.sum())}")

    # ---- fuse -> ground-state raster ----
    # Evidence is now a RATIO, not a count. Three votes means something entirely different
    # when the cell was in view 3 times (unanimous) than when it was in view 40 (7.5% -- noise).
    # Without the denominator both looked identical, which is what the raw-count rule did.
    state = np.full(ncells, UNKNOWN, np.uint8)
    obs_ok = obs_ct >= args.min_obs
    denom = np.maximum(obs_ct, 1)
    p_free = free_ct / denom
    p_foot = foot_ct / denom
    nb_foot = popcount32(foot_bear)
    free = obs_ok & (free_ct >= args.min_free) & (p_free >= args.min_free_frac)
    foot = (obs_ok & (foot_ct >= args.min_foot) & (p_foot >= args.min_foot_frac)
            & (nb_foot >= args.min_bearings))
    state[free] = FREE
    state[foot] = FOOTPRINT                                  # footprint wins ties
    state = state.reshape(rows, cols)

    # HIDDEN: ground occluded *behind an obstacle*, not merely unobserved. A pure
    # morphological close of FREE over UNKNOWN marks every enclosed gap -- including open
    # fringe grass that just wasn't walked -- which over-claims (the yellow blobs). Gate the
    # fill to cells within `hidden_occ_r` of a base-contact (a real occluder): the
    # camera-facing side of an obstacle is observed -> FREE, so the UNKNOWN cells left next to
    # a footprint are precisely its shadow. No footprint nearby -> just unobserved, not hidden.
    if args.hidden_mode == "visibility":
        # HIDDEN is no longer inferred from the SHAPE of the free space. A cell whose ground
        # was in frustum (obs_ct) yet never actually observed as walkable is, by definition,
        # occluded -- something stood in front of it. The morphological close below could only
        # guess that from geometry of the FREE blob, and needed an occluder-proximity fudge to
        # stop it claiming every unwalked fringe.
        hidden = (obs_ok & ~free & ~foot).reshape(rows, cols)
        state[hidden] = HIDDEN
    else:
        state_f = state.reshape(-1)
        k = max(3, int(args.hidden_close) | 1)                  # odd
        freem = (state == FREE).astype(np.uint8)
        closed = cv2.morphologyEx(freem, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        occ_r = max(1, int(args.hidden_occ_r))
        occ = (foot_ct.reshape(rows, cols) > 0).astype(np.uint8)
        occ_dil = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * occ_r + 1,) * 2))
        hidden = (closed == 1) & (state == UNKNOWN) & (occ_dil > 0)
        state[hidden] = HIDDEN
        del state_f

    struct = struct_ct.reshape(rows, cols, -1).argmax(2).astype(np.uint8)
    struct[state != FOOTPRINT] = int(Klass.UNKNOWN)

    np.savez(out / "footprint2d.npz", state=state, structure=struct,
             obs_count=obs_ct.reshape(rows, cols),
             foot_bearings=popcount32(foot_bear).reshape(rows, cols),
             free_count=free_ct.reshape(rows, cols), foot_count=foot_ct.reshape(rows, cols),
             dem=gf.dem, coverage=gf.coverage, cell_size=args.cell_size)
    _render(out, state, struct, gf)
    n = state.size
    print(f"[footprint2d] FREE {np.mean(state==FREE)*100:.1f}%  "
          f"FOOTPRINT {np.mean(state==FOOTPRINT)*100:.1f}%  "
          f"HIDDEN {np.mean(state==HIDDEN)*100:.1f}%  "
          f"UNKNOWN {np.mean(state==UNKNOWN)*100:.1f}%  (of {n} cells)")
    seen = obs_ct > 0
    print(f"  visibility: {100*np.mean(seen):.1f}% of cells ever in frustum, "
          f"median {int(np.median(obs_ct[seen])) if seen.any() else 0} views where seen "
          f"(max {int(obs_ct.max())});  footprint cells with >=2 bearings: "
          f"{int((nb_foot >= 2).sum())} of {int((foot_ct > 0).sum())} voted")
    if rescued:
        top = sorted(rescued.items(), key=lambda kv: -kv[1])
        print(f"  obstacle rescue: {sum(rescued.values())} masks over {len(rescued)} labels -> "
              + ", ".join(f"{k}x{v}" for k, v in top[:8]))
    if args.contact_depth_tol > 0 and gate_pre:
        print(f"  depth contact gate: {gate_pre} -> {gate_post} contacts "
              f"({100*(1-gate_post/max(gate_pre,1)):.0f}% rejected as not touching ground in 3D)")
    print(f"  wrote {out}/footprint2d_bev.png  (+ overlay, npz, {ndbg} debug frames)")


def _render(out, state, struct, gf):
    """North-up: row 0 = origin_v (south), flip so up = +v (north)."""
    rgb = np.zeros((*state.shape, 3), np.uint8)
    rgb[state == FREE] = (60, 170, 60)
    rgb[state == FOOTPRINT] = (40, 40, 220)
    rgb[state == HIDDEN] = (60, 200, 230)
    rgb[state == UNKNOWN] = (35, 35, 35)
    up = lambda a: cv2.resize(np.flipud(a), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / "footprint2d_bev.png"), up(rgb))

    # overlay footprint/free on the DEM hillshade for context
    dem = gf.dem
    gy, gx = np.gradient(dem)
    hill = np.clip(0.5 + 0.5 * (-gx - gy) / (np.hypot(gx, gy).max() + 1e-6), 0, 1)
    base = cv2.cvtColor((hill * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    base[state == FOOTPRINT] = (40, 40, 220)
    base[state == HIDDEN] = (60, 200, 230)
    m = state == FREE
    base[m] = (0.5 * base[m] + np.array([30, 85, 30])).astype(np.uint8)
    cv2.imwrite(str(out / "footprint2d_overlay.png"), up(base))

    # structure footprint (dominant obstacle class per footprint cell)
    lut = np.array(color_lut(), np.uint8)[:, ::-1]          # RGB->BGR for cv2
    srgb = (0.25 * base).astype(np.uint8)
    hask = struct != int(Klass.UNKNOWN)
    srgb[hask] = lut[struct[hask]]
    cv2.imwrite(str(out / "footprint2d_structure.png"), up(srgb))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="maguro-park-after-itchy")
    ap.add_argument("--model", default=None, help="COLMAP text model with TRACKS (raw, not refined)")
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num", type=int, default=40, help="frames to fuse")
    ap.add_argument("--cell-size", type=float, default=0.5)
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--dem-source", default="colmap", choices=["colmap", "mono"],
                    help="ground DEM from sparse COLMAP points (default) or dense fused mono depth")
    ap.add_argument("--depth-dir", default=None, help="per-view mono depth .npy dir (default: session depth-mono)")
    ap.add_argument("--depth-stride", type=int, default=8, help="pixel stride when back-projecting mono depth")
    ap.add_argument("--depth-voxel", type=float, default=0.1, help="voxel size (m) for mono-depth dedup")
    ap.add_argument("--depth-max", type=float, default=40.0, help="drop mono-depth pixels beyond this range (m)")
    ap.add_argument("--seg-mode", default="semantic", choices=["semantic", "panoptic"])
    ap.add_argument("--seg-model", default="facebook/mask2former-swin-large-ade-semantic",
                    help="semantic: ADE (cached); panoptic: e.g. mask2former-swin-large-coco-panoptic")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--min-free", type=int, default=2, help="ground votes -> FREE")
    ap.add_argument("--min-foot", type=int, default=2, help="base-contact votes -> FOOTPRINT")
    ap.add_argument("--max-ground", type=int, default=8000, help="ground px/frame subsample")
    ap.add_argument("--z-far", type=float, default=70.0)
    ap.add_argument("--dz", type=float, default=0.2)
    ap.add_argument("--max-range", type=float, default=20.0, help="max base-contact distance (m)")
    # --- obstacle rescue (open-vocab fill for classes the closed set lacks) ---
    ap.add_argument("--rescue-prompt",
                    default="vending machine. kiosk. public toilet. statue. planter. "
                            "bollard. bike rack. playground equipment. picnic table.",
                    help="open-vocab classes to rescue into Role.OBSTACLE when the closed-set "
                         "segmenter leaves them UNKNOWN/IGNORE. Empty string disables (and "
                         "skips loading the detector).")
    ap.add_argument("--rescue-klass", default="FURNITURE",
                    choices=[k.name for k in Klass],
                    help="taxonomy class assigned to rescued pixels")
    ap.add_argument("--rescue-min-px", type=int, default=600,
                    help="ignore rescued masks smaller than this (noise)")
    ap.add_argument("--rescue-max-frac", type=float, default=0.25,
                    help="reject a rescued mask covering more than this fraction of the frame "
                         "(open-vocab detectors emit whole-scene boxes that would blanket the "
                         "BEV with phantom occupancy)")
    ap.add_argument("--rescue-box-th", type=float, default=0.35)
    ap.add_argument("--rescue-text-th", type=float, default=0.25)
    ap.add_argument("--det-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam-model", default="facebook/sam-vit-base")
    ap.add_argument("--contact-depth-tol", type=float, default=0.15,
                    help="reject a ground contact when measured mono depth disagrees with the "
                         "ray-DEM hit range by more than this FRACTION of the range "
                         "(catches masks that touch ground in 2D but float in 3D). 0 disables.")
    # --- multi-view evidence ---
    ap.add_argument("--obs-max-range", type=float, default=40.0,
                    help="range cap for the visibility raycast. Deliberately looser than "
                         "--max-range: that bounds where a CONTACT is trustworthy, this only "
                         "asks whether the ground was in view at all.")
    ap.add_argument("--obs-stride", type=int, default=8,
                    help="pixel stride for the per-frame visibility raycast (the denominator)")
    ap.add_argument("--min-obs", type=int, default=2,
                    help="frames that must have had a cell's ground in frustum before we call it")
    ap.add_argument("--min-free-frac", type=float, default=0.25,
                    help="fraction of observing frames that must see a cell as ground -> FREE")
    ap.add_argument("--min-foot-frac", type=float, default=0.20,
                    help="fraction of observing frames that must vote base-contact -> FOOTPRINT")
    ap.add_argument("--min-bearings", type=int, default=2,
                    help="distinct camera bearings (of 32 bins) required for FOOTPRINT; "
                         "votes from a single viewpoint are one correlated observation, and "
                         "counting them individually is what draws the radial BEV streaks")
    ap.add_argument("--hidden-mode", default="visibility", choices=["visibility", "morphology"],
                    help="visibility: in-frustum but never seen free = occluded (geometric). "
                         "morphology: the older close+dilate heuristic, kept for comparison.")
    ap.add_argument("--hidden-close", type=int, default=5, help="HIDDEN gap-fill kernel (cells)")
    ap.add_argument("--hidden-occ-r", type=int, default=4,
                    help="HIDDEN only within this many cells of a base-contact (occlusion shadow)")
    ap.add_argument("--debug-frames", type=int, default=4)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
