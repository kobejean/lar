"""Object footprint *points/lines* from open-vocab detection + BEV foot-point clustering.

The dedup problem with semantic masks: the same tree, projected from 970 frames, smears into
one un-separable blob. The fix multi-view BEV localization uses is the **foot point** — reduce
each detected object to a single ground-contact point (bottom-centre of its box), project that
to the ground plane, and **cluster across frames**. Points from the same object collapse to one
landmark; a stray bad frame is an outlier that doesn't cluster. Instance detection (not semantic
segmentation) is what makes objects separable in the first place.

Pipeline (per sampled frame):
  1. **Grounding DINO** (Apache-2.0, open-vocab) detects `tree / bench / pole / sign / ...` →
     one box per instance from a text prompt.
  2. **Foot point** = bottom-centre of the box.
  3. Look up the **DEM ground depth** at that pixel (the ground surface we trust — reuses the
     footprint_labels DEM render — instead of mono depth at the noisy object edge) and
     back-project → a single world point on the ground.
Then across all frames: gravity-align, **DBSCAN per class** → one deduped footprint per object,
with an open-vocab label. Drops straight into IMDF `amenity.landmark` (trees) / `unit.structure`.

Geometry-only DEM by default; pass `--dem-source mono` (same as footprint_labels) for a denser
ground surface so more foot pixels resolve to a ground depth.

Run (from repo root; needs the `segmentation` extra for transformers/torch):
  uv run --extra segmentation python script/backbone/base_points.py --session maguro-park-after-itchy --sample 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "semantic_bev"))

from colmap_io import qvec2rotmat, read_model  # noqa: E402
from lar_session import Session  # noqa: E402
from footprint_labels import (  # noqa: E402
    backproject_positions, build_ground_mesh, make_frame_labels, pinhole_params,
)

DEFAULT_PROMPT = "tree. bench. pole. sign. trash can. street lamp. rock. bush."
# stable BGR colours per canonical class (fallback cycles for unknowns)
CLASS_COLORS = {
    "tree": (60, 170, 60), "bench": (200, 150, 40), "pole": (40, 140, 220),
    "sign": (40, 40, 220), "trash can": (180, 60, 200), "street lamp": (60, 200, 200),
    "rock": (120, 120, 120), "bush": (80, 200, 140),
}
_FALLBACK = [(200, 100, 60), (60, 100, 200), (100, 200, 60), (200, 60, 160)]


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def load_detector(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def detect(proc, model, pil: Image.Image, prompt: str, device: str,
           box_thr: float, text_thr: float):
    """Return list of (box xyxy in original px, label str, score)."""
    inputs = proc(images=pil, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[pil.size[::-1]])[0]
    boxes = res["boxes"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    labels = res.get("text_labels", res.get("labels"))
    return list(zip(boxes, labels, scores))


def canonical(label: str, classes: list[str]) -> str:
    """Map a (possibly multi-word / partial) detection phrase to a prompt class."""
    lab = str(label).lower().strip()
    for c in classes:
        if c in lab or lab in c:
            return c
    return lab or "object"


# --------------------------------------------------------------------------- #
# foot point -> ground world point
# --------------------------------------------------------------------------- #
def foot_world(box, ground_depth, scale, cam, im, min_depth=0.5):
    """Bottom-centre of the box -> world point on the DEM ground (or None if off-ground)."""
    x0, y0, x1, y1 = box
    h, w = ground_depth.shape
    fx_px = int(round(np.clip((x0 + x1) * 0.5 * scale, 0, w - 1)))
    fy_px = int(round(np.clip(y1 * scale, 0, h - 1)))
    d = ground_depth[fy_px, fx_px]
    if not np.isfinite(d) or d < min_depth:
        return None
    fx, fy, cx, cy = (v * scale for v in pinhole_params(cam))
    cam_pt = np.array([(fx_px - cx) / fx * d, (fy_px - cy) / fy * d, d])
    R = qvec2rotmat(im.qvec)
    return cam_pt @ R + (-R.T @ im.tvec)  # world = cam @ R + C


# --------------------------------------------------------------------------- #
# clustering (DBSCAN via KDTree; no sklearn)
# --------------------------------------------------------------------------- #
def dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = len(points)
    labels = np.full(n, -1, np.int64)
    if n == 0:
        return labels
    tree = cKDTree(points)
    visited = np.zeros(n, bool)
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        seeds = tree.query_ball_point(points[i], eps)
        if len(seeds) < min_samples:
            continue  # provisional noise; may still be absorbed into a cluster below
        labels[i] = cid
        k = 0
        while k < len(seeds):
            j = seeds[k]; k += 1
            if labels[j] == -1:
                labels[j] = cid
            if not visited[j]:
                visited[j] = True
                nj = tree.query_ball_point(points[j], eps)
                if len(nj) >= min_samples:
                    seeds.extend(nj)
        cid += 1
    return labels


def cluster_footprints(pts_uv: np.ndarray, labels: list[str], scores: np.ndarray,
                       classes: list[str], eps: float, min_samples: int, log=print):
    """Per-class DBSCAN -> one landmark per cluster (score-weighted centroid)."""
    out = []
    labels = np.asarray(labels)
    for c in sorted(set(labels)):
        m = labels == c
        p = pts_uv[m]; sc = scores[m]
        cl = dbscan(p, eps, min_samples)
        for k in sorted(set(cl) - {-1}):
            sel = cl == k
            w = sc[sel]
            centre = np.average(p[sel], axis=0, weights=w)
            out.append({"class": str(c), "u": float(centre[0]), "v": float(centre[1]),
                        "count": int(sel.sum()), "score": float(w.mean())})
    out.sort(key=lambda d: -d["count"])
    log(f"  clustered {len(pts_uv)} foot points -> {len(out)} objects "
        f"({len(set(labels))} classes)")
    return out


# --------------------------------------------------------------------------- #
# BEV render
# --------------------------------------------------------------------------- #
def colour_for(cls: str, i: int):
    return CLASS_COLORS.get(cls, _FALLBACK[i % len(_FALLBACK)])


def render_bev(gf, objects, traj_uv, out_path: Path, ppm: float = 6.0):
    spec = gf.spec
    origin_u, origin_v, cs = spec.origin_u, spec.origin_v, spec.cell_size
    W = int(spec.cols * cs * ppm) + 40
    H = int(spec.rows * cs * ppm) + 40

    def to_px(u, v):
        return int((u - origin_u) * ppm) + 20, int((v - origin_v) * ppm) + 20

    img = np.full((H, W, 3), 248, np.uint8)
    # faint observed-DEM coverage as background context
    cov = cv2.resize(gf.coverage.astype(np.uint8) * 255, (W - 40, H - 40),
                     interpolation=cv2.INTER_NEAREST)
    img[20:H - 20, 20:W - 20][cov > 0] = (233, 233, 233)

    # camera trajectory
    if len(traj_uv) > 1:
        pts = np.array([to_px(u, v) for u, v in traj_uv], np.int32)
        cv2.polylines(img, [pts], False, (170, 170, 170), 1, cv2.LINE_AA)

    classes = sorted({o["class"] for o in objects})
    cidx = {c: i for i, c in enumerate(classes)}
    for o in objects:
        col = colour_for(o["class"], cidx[o["class"]])
        px, py = to_px(o["u"], o["v"])
        r = int(np.clip(3 + o["count"] * 0.5, 4, 18))
        cv2.circle(img, (px, py), r, col, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), r, (40, 40, 40), 1, cv2.LINE_AA)

    # legend
    y = 30
    for c in classes:
        col = colour_for(c, cidx[c])
        n = sum(o["count"] for o in objects if o["class"] == c)
        k = sum(1 for o in objects if o["class"] == c)
        cv2.circle(img, (30, y), 6, col, -1, cv2.LINE_AA)
        cv2.putText(img, f"{c}: {k} objs ({n} det)", (44, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    cv2.imwrite(str(out_path), img)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session")
    ap.add_argument("--model", help="COLMAP text model dir (default: session.best_model())")
    ap.add_argument("--images", help="image dir (default: session.images)")
    ap.add_argument("--out", help="output dir (default: output/<session>-basepoints)")
    ap.add_argument("--dem-source", choices=["colmap", "mono"], default="colmap")
    ap.add_argument("--depth-dir", help="mono depth dir (default: session.depth_dir('mono'))")
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="open-vocab classes, '.'-separated")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=1.2, help="DBSCAN cluster radius (m)")
    ap.add_argument("--min-samples", type=int, default=3, help="min detections to accept an object")
    ap.add_argument("--cell-size", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=512, help="DEM render resolution for depth lookup")
    ap.add_argument("--max-range", type=float, default=30.0)
    ap.add_argument("--sample", type=int, default=80, help="frames spaced evenly across the capture")
    ap.add_argument("--save-detections", type=int, default=8, help="save N per-frame detection overlays")
    args = ap.parse_args()

    s = Session(args.session) if args.session else None
    if s:
        model = Path(args.model) if args.model else s.best_model()
        images = Path(args.images) if args.images else s.images
        out = Path(args.out) if args.out else s.root / "output" / f"{s.name}-basepoints"
        depth_dir = Path(args.depth_dir) if args.depth_dir else s.depth_dir("mono")
    else:
        if not (args.model and args.images and args.out):
            ap.error("without --session, pass --model, --images and --out")
        model, images, out = Path(args.model), Path(args.images), Path(args.out)
        depth_dir = Path(args.depth_dir) if args.depth_dir else None
    out.mkdir(parents=True, exist_ok=True)
    det_dir = out / "detections"
    det_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model {model}\nimages {images}\nout {out}\ndevice {device}")
    recon = read_model(str(model))

    dem_world = None
    if args.dem_source == "mono":
        if depth_dir is None:
            ap.error("--dem-source mono needs --depth-dir (or --session)")
        print(f"DEM source: mono depth <- {depth_dir}")
        dem_world = backproject_positions(recon, depth_dir, list(recon.images), 8, 0.1)
    mesh, occ_world = build_ground_mesh(
        recon, args.cell_size, (0.4, 2.0), 3, 0.8, dem_world=dem_world)
    gf = mesh.gf

    classes = [c.strip().lower() for c in args.prompt.split(".") if c.strip()]
    print(f"classes: {classes}")
    proc, det_model = load_detector(args.detector, device)

    ims = sorted(recon.images.values(), key=lambda im: im.name)
    if args.sample and args.sample < len(ims):
        idx = np.linspace(0, len(ims) - 1, args.sample).round().astype(int)
        ims = [ims[i] for i in idx]

    all_uv, all_lab, all_score = [], [], []
    traj_uv = []
    n_det = 0
    for fi, im in enumerate(ims):
        cam = recon.cameras[im.camera_id]
        R = qvec2rotmat(im.qvec)
        cam_centre = -R.T @ im.tvec                 # camera centre in world
        traj_uv.append((cam_centre @ gf.R.T)[:2])   # -> gravity-local (u, v)
        res = make_frame_labels(mesh, occ_world, im, cam, args.size,
                                args.max_range, 0.5, 0)
        if res is None:
            continue
        ground_depth = res["depth"]
        scale = args.size / max(cam.width, cam.height)
        pil = Image.open(images / im.name).convert("RGB")
        dets = detect(proc, det_model, pil, args.prompt, device, args.box_thr, args.text_thr)

        overlay = cv2.imread(str(images / im.name)) if fi < args.save_detections else None
        for box, label, score in dets:
            cls = canonical(label, classes)
            w = foot_world(box, ground_depth, scale, cam, im)
            if w is None:
                continue
            local = w @ gf.R.T
            all_uv.append(local[:2]); all_lab.append(cls); all_score.append(float(score))
            n_det += 1
            if overlay is not None:
                x0, y0, x1, y1 = box.astype(int)
                col = colour_for(cls, classes.index(cls) if cls in classes else 0)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 2)
                cv2.circle(overlay, (int((x0 + x1) / 2), y1), 6, (0, 0, 255), -1)
                cv2.putText(overlay, f"{cls} {score:.2f}", (x0, max(y0 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        if overlay is not None:
            cv2.imwrite(str(det_dir / f"{Path(im.name).stem}.jpg"), overlay)
        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(ims)} frames, {n_det} foot points")

    traj_uv = [(t[0], t[1]) for t in traj_uv]
    print(f"detected {n_det} foot points over {len(ims)} frames")
    if not all_uv:
        raise SystemExit("no foot points landed on the ground — check detector/prompt/DEM")

    objects = cluster_footprints(
        np.array(all_uv), all_lab, np.array(all_score), classes,
        args.eps, args.min_samples)

    render_bev(gf, objects, traj_uv, out / "bev_footprints.png")
    (out / "objects.json").write_text(json.dumps(objects, indent=2))
    print(f"\n{len(objects)} objects -> {out}/bev_footprints.png")
    by_cls: dict[str, int] = {}
    for o in objects:
        by_cls[o["class"]] = by_cls.get(o["class"], 0) + 1
    for c, k in sorted(by_cls.items(), key=lambda x: -x[1]):
        print(f"  {c:14s} {k}")


if __name__ == "__main__":
    main()
