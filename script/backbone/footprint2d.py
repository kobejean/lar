"""2D-signals -> BEV footprint / free-space / hidden-ground pseudo-labels.

An alternative to the 3D-DEM-render footprint generator: instead of rendering the
global DEM into each camera, fuse **per-frame 2D perception** (panoptic segmentation
+ metric depth) into the BEV, using the gravity-aligned ground DEM only as a *ray
target*. Three ground states come out, IMDF-ready:

  FREE       -- walkable ground, directly observed
  FOOTPRINT  -- ground occupied by a vertical object's *base* (not walkable)
  HIDDEN     -- ground enclosed by free space but unobserved in every view (walk-under
                / occluded); recovered by multi-view fusion, not any single frame

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

Fuse across frames in BEV; HIDDEN falls out where FREE encloses unobserved cells.

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
    Returns (uv (M,2), ok (M,) bool). Rejects hits outside the grid, off-coverage, or
    beyond ``max_range`` m (far base-contacts are unreliable: ray-DEM at range + thin
    DEM coverage scatters, so we trust only nearby contacts and let other frames cover
    the rest)."""
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
    return hit_uv, ok & cov & (zt <= max_range)


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

    free_ct = np.zeros(ncells, np.int32)
    foot_ct = np.zeros(ncells, np.int32)
    struct_ct = np.zeros((ncells, int(max(Klass)) + 1), np.int32)  # per-cell obstacle class votes

    frames = sample_frames(recon, args.num)
    print(f"  {len(frames)} frames, grid {cols}x{rows} @ {args.cell_size} m")
    ndbg = 0
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
            np.add.at(free_ct, cells, 1)

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
            hit_uv, ok = ray_dem_intersect(C_local, dir_local, gf,
                                           z_far=args.z_far, dz=args.dz,
                                           max_range=args.max_range)
            if ok.any():
                cells = gf.cell_of(hit_uv[ok])
                np.add.at(foot_ct, cells, 1)
                np.add.at(struct_ct, (cells, base_klass[ok]), 1)

        # ---- per-frame debug overlay (first few) ----
        if ndbg < args.debug_frames:
            dbg = small.copy()
            dbg[gmask] = (0.5 * dbg[gmask] + np.array([0, 120, 0])).astype(np.uint8)
            if base_klass is not None:
                for x, y in zip(bx, by):
                    cv2.circle(dbg, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.imwrite(str(out / f"frame_{ndbg}_{Path(img.name).stem}.png"), dbg)
            ndbg += 1
        if (fi + 1) % 10 == 0:
            print(f"  {fi + 1}/{len(frames)}  free~{int(free_ct.sum())} foot~{int(foot_ct.sum())}")

    # ---- fuse -> ground-state raster ----
    state = np.full(ncells, UNKNOWN, np.uint8)
    free = free_ct >= args.min_free
    foot = foot_ct >= args.min_foot
    state[free] = FREE
    state[foot] = FOOTPRINT                                  # footprint wins ties
    state = state.reshape(rows, cols)

    # HIDDEN: ground occluded *behind an obstacle*, not merely unobserved. A pure
    # morphological close of FREE over UNKNOWN marks every enclosed gap -- including open
    # fringe grass that just wasn't walked -- which over-claims (the yellow blobs). Gate the
    # fill to cells within `hidden_occ_r` of a base-contact (a real occluder): the
    # camera-facing side of an obstacle is observed -> FREE, so the UNKNOWN cells left next to
    # a footprint are precisely its shadow. No footprint nearby -> just unobserved, not hidden.
    k = max(3, int(args.hidden_close) | 1)                  # odd
    freem = (state == FREE).astype(np.uint8)
    closed = cv2.morphologyEx(freem, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    occ_r = max(1, int(args.hidden_occ_r))
    occ = (foot_ct.reshape(rows, cols) > 0).astype(np.uint8)   # any base-contact = an occluder
    occ_dil = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * occ_r + 1,) * 2))
    hidden = (closed == 1) & (state == UNKNOWN) & (occ_dil > 0)
    state[hidden] = HIDDEN

    struct = struct_ct.reshape(rows, cols, -1).argmax(2).astype(np.uint8)
    struct[state != FOOTPRINT] = int(Klass.UNKNOWN)

    np.savez(out / "footprint2d.npz", state=state, structure=struct,
             free_count=free_ct.reshape(rows, cols), foot_count=foot_ct.reshape(rows, cols),
             dem=gf.dem, coverage=gf.coverage, cell_size=args.cell_size)
    _render(out, state, struct, gf)
    n = state.size
    print(f"[footprint2d] FREE {np.mean(state==FREE)*100:.1f}%  "
          f"FOOTPRINT {np.mean(state==FOOTPRINT)*100:.1f}%  "
          f"HIDDEN {np.mean(state==HIDDEN)*100:.1f}%  "
          f"UNKNOWN {np.mean(state==UNKNOWN)*100:.1f}%  (of {n} cells)")
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
    ap.add_argument("--hidden-close", type=int, default=5, help="HIDDEN gap-fill kernel (cells)")
    ap.add_argument("--hidden-occ-r", type=int, default=4,
                    help="HIDDEN only within this many cells of a base-contact (occlusion shadow)")
    ap.add_argument("--debug-frames", type=int, default=4)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
