"""Per-object footprint polygons + ground-contact keypoints (Grounded-SAM -> BEV).

``footprint2d.py`` fuses a *class-level* BEV: cells are FREE / FOOTPRINT / HIDDEN, but every
tree in a row of trees melts into one anonymous red arc. Semantic segmentation cannot fix
this -- trees are "stuff" in both COCO and ADE, so there are no per-tree ids to fuse. This
script adds the missing level: **instances**, each with its own footprint polygon and a single
ground-contact keypoint, which is what IMDF (``amenity.landmark`` / ``unit.structure``) and any
downstream navigation graph actually want.

Per frame:
  1. **Grounding DINO** (Apache-2.0, open-vocab) -> one box per object from a text prompt.
  2. **SAM** (Apache-2.0) -> a real mask per box (a box is not a footprint; a leaning tree's
     box covers metres of path it doesn't occupy).
  3. **Ground-contact columns.** For every image column the mask spans, take its bottom-most
     masked pixel and keep it only if the pixels just below are Role.GROUND in the semantic
     seg. This is the canopy guard: a canopy's mask bottom is mid-air with more foliage under
     it, so it is rejected, while the trunk base -- which sits on path/grass -- is kept. Without
     the guard a floating canopy's ray-DEM hit lands metres behind the tree.
  4. **Ray-DEM** those contact pixels (calibration + the gravity DEM, never mono depth at the
     object edge where it is worst) -> BEV cells = this object's ground extent in this frame.
     Too few valid contacts -> keep the object as a point-only landmark (box bottom-centre).
  5. **Depth contact gate** (``--contact-depth-tol``). Step 3 is a *2D* test and cannot catch a
     mask whose bottom abuts ground that is really far behind it -- a bench seat or canopy
     silhouetted against open lawn passes the canopy guard, because the pixels below genuinely
     are ground. So compare the measured mono depth at the contact against the range at which
     that ray meets the DEM: if the object stands there they agree; if it floats, it is much
     nearer than the ground its ray hits. Note this does not contradict step 4 -- mono depth is
     never used to *place* the contact (it stays ray-DEM), only to *veto* one. A relative
     comparison at a single pixel is what mono depth is reliable for; metric placement at an
     object edge is what it is not.

Across frames: union-find clusters observations by label + contact proximity, so the same tree
seen from 12 views collapses to one instance and a one-frame false positive stays a singleton
and is dropped. Per cluster the polygon is the cells voted by >= --min-cell-views observations
(largest connected component, contoured); the keypoint is the median contact.

Run (from repo root; offline avoids the HF HEAD-hang, all weights are cached):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --extra segmentation \
    python script/backbone/footprint_instances.py --session maguro-park-after-itchy \
    --num 40 --dem-source mono
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SCRIPT_ROOT))
sys.path.insert(0, str(_SCRIPT_ROOT / "semantic_bev"))
sys.path.insert(0, str(_SCRIPT_ROOT / "depth"))

from lar_session import Session                      # noqa: E402
from colmap_io import read_model                     # noqa: E402
from geometry import from_reconstruction             # noqa: E402
from taxonomy import Klass, Role, role_of            # noqa: E402
from footprint2d import (                            # noqa: E402
    MonoDepth, SegKlass, camera_local, depth_contact_gate, pixel_dirs, ray_dem_intersect,
    mono_dem_field, sample_frames,
)

DEFAULT_PROMPT = "tree. bench. pole. sign. trash can. street lamp. rock. bush. fence."

# stable BGR per canonical label (matches base_points.py so the two outputs read alike)
CLASS_COLORS = {
    "tree": (60, 170, 60), "bench": (200, 150, 40), "pole": (40, 140, 220),
    "sign": (40, 40, 220), "trash can": (180, 60, 200), "street lamp": (60, 200, 200),
    "rock": (120, 120, 120), "bush": (80, 200, 140), "fence": (150, 150, 60),
}
_FALLBACK = [(200, 100, 60), (60, 100, 200), (100, 200, 60), (200, 60, 160)]

# per-label merge radius (m): how far two contacts of the same label can be and still be
# the same object. A tree trunk localises tightly; a bench or fence spans metres.
MERGE_RADIUS = {"tree": 1.2, "bush": 1.5, "bench": 1.5, "fence": 2.0}
DEFAULT_RADIUS = 1.0


def color_for(label: str) -> tuple[int, int, int]:
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    return _FALLBACK[hash(label) % len(_FALLBACK)]


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


# ---------------------------------------------------------------------------
# ground contacts of one instance mask
# ---------------------------------------------------------------------------
def contact_columns(mask: np.ndarray, ground: np.ndarray, stride: int, probe: int):
    """Columns where ``mask``'s bottom-most pixel is a genuine ground contact.

    The canopy guard: a column's lowest masked pixel is only a *base* if what lies immediately
    below it is walkable ground. Under a canopy the pixels below are more tree/background, so
    the column is rejected and the object contributes no phantom footprint metres behind it.
    Columns bottoming out at the image edge are rejected too -- the object continues out of
    frame, so its true base is unseen."""
    H, W = mask.shape
    cols = np.where(mask.any(0))[0][::stride]
    if not len(cols):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rows = H - 1 - mask[::-1][:, cols].argmax(0)          # bottom-most masked row per column
    keep = rows < H - 2
    cols, rows = cols[keep], rows[keep]
    if not len(cols):
        return cols, rows
    below = np.clip(rows[None, :] + np.arange(1, probe + 1)[:, None], 0, H - 1)
    is_ground = ground[below, cols[None, :]].mean(0) > 0.5
    return cols[is_ground], rows[is_ground]


# ---------------------------------------------------------------------------
# multi-view fusion: union-find over (label, contact proximity)
# ---------------------------------------------------------------------------
def cluster_observations(obs, radius_of):
    """Group per-frame observations of the same physical object. Returns list of index lists."""
    from scipy.spatial import cKDTree
    n = len(obs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    xy = np.array([o["contact"][:2] for o in obs], np.float64)
    labels = [o["label"] for o in obs]
    tree = cKDTree(xy)
    rmax = max(radius_of(l) for l in set(labels)) if labels else 0.0
    for i, j in tree.query_pairs(rmax):
        if labels[i] != labels[j]:
            continue
        r = radius_of(labels[i])
        if np.linalg.norm(xy[i] - xy[j]) <= r:
            union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def polygon_of(cell_bearings, rows, cols, spec, min_bearings, close_k, contact, max_extent):
    """Cell votes -> (contour in local metres, area m^2, mask). None if nothing survives.

    ``cell_bearings`` maps cell -> set of quantised camera bearings that voted it. Counting
    *distinct bearings* rather than observations is what kills the ray-streak artefact: depth
    along a ray is the weak axis of ray-DEM, so a mis-estimated contact smears cells backwards
    away from the camera -- but only along that one line of sight. Consecutive frames share a
    vantage and happily re-vote the same wrong streak, so an observation count cannot tell the
    difference. Cells on the object's real ground extent are hit from every side you walk past
    it; cells on a streak are hit from one. Requiring >= 2 bearings is a small-scale visual
    hull: keep only what multiple viewpoints agree on."""
    grid = np.zeros(rows * cols, np.int32)
    for c, bearings in cell_bearings.items():
        grid[c] = len(bearings)
    m = (grid.reshape(rows, cols) >= min_bearings).astype(np.uint8)
    if max_extent > 0 and m.any():                       # safety net: no runaway tails
        yy, xx = np.mgrid[0:rows, 0:cols]
        du = spec.origin_u + (xx + 0.5) * spec.cell_size - contact[0]
        dv = spec.origin_v + (yy + 0.5) * spec.cell_size - contact[1]
        m &= (np.hypot(du, dv) <= max_extent).astype(np.uint8)
    if not m.any():
        return None
    k = max(3, int(close_k) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if nlab <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    cnt = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2)   # simplify to ~1 cell
    poly = np.stack([spec.origin_u + (cnt[:, 0] + 0.5) * spec.cell_size,
                     spec.origin_v + (cnt[:, 1] + 0.5) * spec.cell_size], axis=1)
    return poly, float(m.sum()) * spec.cell_size ** 2, m


# ---------------------------------------------------------------------------
def run(args):
    s = Session(args.session)
    model = args.model or str(s.colmap_model)
    images_dir = Path(args.images or str(s.images))
    out = Path(args.out or (s.root / "output" / f"{args.session}-instances"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[instances] model={model}\n  images={images_dir}\n  out={out}")
    recon = read_model(model)
    if args.dem_source == "mono":
        depth_dir = args.depth_dir or str(s.depth_dir("mono"))
        print(f"  DEM source: mono depth <- {depth_dir}")
        gf = mono_dem_field(recon, depth_dir, args.cell_size, args.depth_stride,
                            args.depth_voxel, args.depth_max, log=print)
    else:
        gf, *_ = from_reconstruction(recon, cell_size=args.cell_size)
    rows, cols = gf.dem.shape

    seg = SegKlass(args.seg_model, mode="semantic")
    gsam = GroundedSAM(args.det_model, args.sam_model, args.prompt, args.box_th, args.text_th)
    # The semantic canopy guard in contact_columns is a 2D test and cannot tell a base from a
    # silhouette against distant ground; depth_contact_gate settles it geometrically.
    mono = MonoDepth(args.depth_model) if args.contact_depth_tol > 0 else None
    roles = np.array([int(role_of(int(k))) for k in range(int(max(Klass)) + 1)], np.uint8)

    frames = sample_frames(recon, args.num)
    print(f"  {len(frames)} frames, grid {cols}x{rows} @ {args.cell_size} m, prompt: {args.prompt}")

    obs, ndbg = [], 0
    gate_pre = gate_post = 0
    for fi, img in enumerate(frames):
        bgr = cv2.imread(str(images_dir / img.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[0] // args.data_factor, bgr.shape[1] // args.data_factor
        small = cv2.resize(bgr, (W, H))
        klass, _ = seg(small)
        ground = roles[klass] == int(Role.GROUND)
        dmap = mono.metric(bgr, recon, img, args.data_factor)[0] if mono is not None else None
        masks, boxes, labels, scores = gsam(small)
        cam = recon.cameras[img.camera_id]
        C_local, RwcT = camera_local(img, gf)
        dbg = small.copy() if ndbg < args.debug_frames else None

        for mi in range(len(masks)):
            mask = masks[mi]
            if mask.sum() < args.min_mask_px:
                continue
            cx, cy = contact_columns(mask, ground, args.col_stride, args.ground_probe)
            point_only = len(cx) < args.min_contacts
            if point_only:                                # fall back to box bottom-centre
                x0, y0, x1, y1 = boxes[mi]
                cx = np.array([(x0 + x1) / 2], np.int64)
                cy = np.array([min(y1, H - 3)], np.int64)
            d_cam = pixel_dirs(cx.astype(np.float64), cy.astype(np.float64), cam, args.data_factor)
            dir_local = (RwcT @ d_cam.T).T
            hit_uv, ok, zt = ray_dem_intersect(C_local, dir_local, gf, z_far=args.z_far,
                                               dz=args.dz, max_range=args.max_range)
            gate_pre += int(ok.sum())
            ok = depth_contact_gate(dmap, cy, cx, zt, ok, args.contact_depth_tol)
            gate_post += int(ok.sum())
            if not ok.any():
                continue
            hits = hit_uv[ok]
            contact = np.array([np.median(hits[:, 0]), np.median(hits[:, 1])])
            # bearing camera->contact, quantised: the viewpoint identity used to gate polygons
            bear = np.degrees(np.arctan2(contact[1] - C_local[1], contact[0] - C_local[0]))
            obs.append({
                "frame": Path(img.name).stem, "label": labels[mi], "score": float(scores[mi]),
                "contact": np.array([contact[0], contact[1],
                                     float(gf.height_at(contact[None, :])[0])]),
                "cells": np.unique(gf.cell_of(hits)) if not point_only else np.zeros(0, np.int64),
                "bearing": int(bear // args.bearing_bin),
                "point_only": bool(point_only),
            })
            if dbg is not None:
                c = color_for(labels[mi])
                dbg[mask] = (0.6 * dbg[mask] + 0.4 * np.array(c)).astype(np.uint8)
                for x, y in zip(cx, cy):
                    cv2.circle(dbg, (int(x), int(y)), 2, (0, 0, 255) if not point_only else (0, 200, 255), -1)
                cv2.putText(dbg, labels[mi], (int(boxes[mi][0]), max(12, int(boxes[mi][1]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        if dbg is not None:
            cv2.imwrite(str(out / f"frame_{ndbg}_{Path(img.name).stem}.png"), dbg)
            ndbg += 1
        if (fi + 1) % 10 == 0:
            print(f"  {fi + 1}/{len(frames)}  observations={len(obs)}")

    if not obs:
        raise SystemExit("no instance observations -- lower --box-th or check the prompt")

    # ---- fuse observations into instances ----
    groups = cluster_observations(obs, lambda l: MERGE_RADIUS.get(l, args.merge_radius))
    print(f"  {len(obs)} observations -> {len(groups)} raw clusters")

    instances = []
    for g in sorted(groups, key=len, reverse=True):
        members = [obs[i] for i in g]
        nviews = len({m["frame"] for m in members})
        if nviews < args.min_views:                       # a one-frame blip is a false positive
            continue
        contacts = np.array([m["contact"] for m in members])
        contact = np.median(contacts, axis=0)
        votes: dict[int, set] = {}
        for m in members:
            for c in m["cells"]:
                votes.setdefault(int(c), set()).add(m["bearing"])
        nbear = len({m["bearing"] for m in members})
        poly = polygon_of(votes, rows, cols, gf.spec,
                          min(args.min_bearings, max(1, nbear)), args.close_k,
                          contact, args.max_extent)
        label = max(set(m["label"] for m in members), key=[m["label"] for m in members].count)
        world = gf.R.T @ contact
        instances.append({
            "id": len(instances), "label": label,
            "score": float(np.mean([m["score"] for m in members])),
            "n_views": nviews, "n_obs": len(members),
            "contact_local": [float(x) for x in contact],
            "contact_world": [float(x) for x in world],
            "area_m2": None if poly is None else round(poly[1], 2),
            "polygon_local": None if poly is None else [[float(a), float(b)] for a, b in poly[0]],
            "_mask": None if poly is None else poly[2],
        })

    print(f"  {len(instances)} instances (>= {args.min_views} views)")
    by_label: dict[str, int] = {}
    for it in instances:
        by_label[it["label"]] = by_label.get(it["label"], 0) + 1
    for k, v in sorted(by_label.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s} {v}")
    npoly = sum(1 for it in instances if it["polygon_local"])
    print(f"  {npoly}/{len(instances)} have a footprint polygon "
          f"(rest are point-only landmarks)")

    _render(out, instances, gf)
    ids = np.zeros((rows, cols), np.int32)
    for it in instances:
        if it["_mask"] is not None:
            ids[it["_mask"] > 0] = it["id"] + 1
    np.savez(out / "instances.npz", instance_ids=ids, dem=gf.dem, coverage=gf.coverage,
             cell_size=args.cell_size, origin=[gf.spec.origin_u, gf.spec.origin_v])
    for it in instances:
        it.pop("_mask")
    (out / "instances.json").write_text(json.dumps(
        {"session": args.session, "cell_size": args.cell_size, "frames": len(frames),
         "prompt": args.prompt, "instances": instances}, indent=1))
    if args.contact_depth_tol > 0 and gate_pre:
        print(f"  depth contact gate: {gate_pre} -> {gate_post} contacts "
              f"({100*(1-gate_post/max(gate_pre,1)):.0f}% rejected as not touching ground in 3D)")
    print(f"  wrote {out}/instances.json, instances.npz, instances_bev.png")


def _render(out, instances, gf):
    dem = gf.dem
    gy, gx = np.gradient(dem)
    hill = np.clip(0.5 + 0.5 * (-gx - gy) / (np.hypot(gx, gy).max() + 1e-6), 0, 1)
    base = cv2.cvtColor((hill * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    base = (0.55 * base + 60).astype(np.uint8)
    S = 4
    big = cv2.resize(np.flipud(base), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
    rows = base.shape[0]

    def to_px(u, v):
        """local metres -> flipped, upscaled image pixel"""
        cxp = (u - gf.spec.origin_u) / gf.spec.cell_size
        cyp = (v - gf.spec.origin_v) / gf.spec.cell_size
        return int(cxp * S), int((rows - 1 - cyp) * S)

    for it in instances:
        c = color_for(it["label"])
        if it["polygon_local"]:
            pts = np.array([to_px(u, v) for u, v in it["polygon_local"]], np.int32)
            cv2.fillPoly(big, [pts], tuple(int(0.45 * x) for x in c))
            cv2.polylines(big, [pts], True, c, 2)
    for it in instances:
        c = color_for(it["label"])
        x, y = to_px(it["contact_local"][0], it["contact_local"][1])
        cv2.circle(big, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(big, (x, y), 4, c, -1)
        cv2.putText(big, f"{it['label'][:6]}{it['id']}", (x + 6, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "instances_bev.png"), big)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="maguro-park-after-itchy")
    ap.add_argument("--model", default=None, help="COLMAP text model with TRACKS (raw)")
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num", type=int, default=40, help="frames to fuse")
    ap.add_argument("--cell-size", type=float, default=0.5)
    ap.add_argument("--data-factor", type=int, default=2)
    # ground DEM (same contract as footprint2d)
    ap.add_argument("--dem-source", default="mono", choices=["colmap", "mono"])
    ap.add_argument("--depth-dir", default=None)
    ap.add_argument("--depth-stride", type=int, default=8)
    ap.add_argument("--depth-voxel", type=float, default=0.1)
    ap.add_argument("--depth-max", type=float, default=40.0)
    # detection / segmentation
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="open-vocab classes, '. '-separated")
    ap.add_argument("--det-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam-model", default="facebook/sam-vit-base")
    ap.add_argument("--seg-model", default="facebook/mask2former-swin-large-ade-semantic")
    ap.add_argument("--box-th", type=float, default=0.3)
    ap.add_argument("--text-th", type=float, default=0.25)
    ap.add_argument("--min-mask-px", type=int, default=400)
    # ground contacts
    ap.add_argument("--col-stride", type=int, default=4, help="column subsample within a mask")
    ap.add_argument("--contact-depth-tol", type=float, default=0.15,
                    help="reject a contact when measured mono depth disagrees with the "
                         "ray-DEM hit range by more than this FRACTION of the range. The "
                         "3D counterpart to the 2D canopy guard. 0 disables (and skips "
                         "loading the depth model).")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="mono depth model backing --contact-depth-tol")
    ap.add_argument("--ground-probe", type=int, default=4,
                    help="pixels below the mask that must be GROUND for a valid contact")
    ap.add_argument("--min-contacts", type=int, default=4,
                    help="valid contact columns needed for a polygon; below this -> point-only")
    ap.add_argument("--z-far", type=float, default=70.0)
    ap.add_argument("--dz", type=float, default=0.2)
    ap.add_argument("--max-range", type=float, default=20.0)
    # fusion
    ap.add_argument("--merge-radius", type=float, default=DEFAULT_RADIUS,
                    help="default same-object contact radius (m); per-class overrides in MERGE_RADIUS")
    ap.add_argument("--min-views", type=int, default=3, help="views needed to keep an instance")
    ap.add_argument("--min-bearings", type=int, default=2,
                    help="distinct camera bearings that must vote a cell into the polygon")
    ap.add_argument("--bearing-bin", type=float, default=20.0, help="bearing quantisation (deg)")
    ap.add_argument("--max-extent", type=float, default=6.0,
                    help="drop polygon cells beyond this radius from the contact (m); 0 = off")
    ap.add_argument("--close-k", type=int, default=3, help="polygon morphological close (cells)")
    ap.add_argument("--debug-frames", type=int, default=6)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
