#!/usr/bin/env python3
"""Monocular metric depth (Depth Anything V2), scale-aligned to SfM -> bake-off contract.

Per frame: run Depth Anything V2 (affine-invariant *disparity*), then fit it to the frame's
sparse SfM depth so the result is **metric**. The fit is the standard MiDaS-style
scale+shift in disparity space: for the SfM points observed in the frame we have metric
depth ``z``; we solve ``1/z ≈ a·disp + b`` by least squares, then output
``depth = 1/(a·disp + b)``. This gives dense depth — including thin structures (poles,
chair legs) and low-texture regions MVS/GS miss — anchored to the SfM metric scale.

  uv run --extra segmentation python script/depth/mono_depth.py \
      --session maguro-park-after-itchy --data-factor 2

Per-frame alignment means residual scale error between frames; the back-projection harness
fuses many frames per voxel, which averages it down. Writes <out>/<stem>.npy (+ .conf.npy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# SfM tracks (keypoint -> 3D point) come from the semantic_bev COLMAP reader.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "semantic_bev"))
from colmap_io import read_model  # noqa: E402


def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sparse_samples(recon, img, scale):
    """Observed SfM points -> (pixel_x, pixel_y at depth-res, metric z-depth)."""
    R, t = _qvec2rotmat(img.qvec), img.tvec
    pids, xys = img.point3d_ids, img.xys
    keep = pids >= 0
    if not keep.any():
        return np.empty(0), np.empty(0), np.empty(0)
    px, py, z = [], [], []
    for pid, (u, v) in zip(pids[keep], xys[keep]):
        p = recon.points3d.get(int(pid))
        if p is None:
            continue
        zc = (R @ p.xyz + t)[2]
        if zc > 0:
            px.append(u * scale); py.append(v * scale); z.append(zc)
    return np.array(px), np.array(py), np.array(z)


def fit_metric(disp_at_pts, z, trim=0.1):
    """Solve 1/z ≈ a·disp + b (least squares, one robust trim). Returns (a, b) or None."""
    if len(z) < 20:
        return None
    inv = 1.0 / z
    A = np.stack([disp_at_pts, np.ones_like(disp_at_pts)], axis=1)
    ab, *_ = np.linalg.lstsq(A, inv, rcond=None)
    resid = np.abs(A @ ab - inv)
    keep = resid <= np.quantile(resid, 1 - trim)
    ab, *_ = np.linalg.lstsq(A[keep], inv[keep], rcond=None)
    return float(ab[0]), float(ab[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None)
    ap.add_argument("--model", default=None, help="COLMAP text model (poses + tracks)")
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-factor", type=int, default=2)
    # Default is the Apache-2.0 *Small* model: DA-V2 Base/Large are CC-BY-NC (non-commercial).
    # Per script/depth/depth_bench.py it's ~tied on near-field accuracy with the best backend
    # and ~6x faster. Swap in any HF depth model (e.g. Intel/dpt-beit-large-512, MIT) here.
    ap.add_argument("--model-name", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = ap.parse_args()

    if args.session:
        from lar_session import Session  # _ROOT (script/) already on path
        s = Session(args.session)
        args.model = args.model or str(s.best_model())
        args.images = args.images or str(s.images)
        args.out = args.out or str(s.depth_dir("mono"))
    if not (args.model and args.images and args.out):
        ap.error("need --session or explicit --model/--images/--out")

    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=args.model_name, device=device)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recon = read_model(args.model)
    images_dir = Path(args.images)
    f = args.data_factor
    n_ok = n_skip = 0

    for i, img in enumerate(sorted(recon.images.values(), key=lambda im: im.name)):
        bgr = cv2.imread(str(images_dir / img.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[0] // f, bgr.shape[1] // f
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pred = pipe(PILImage.fromarray(rgb))["predicted_depth"]  # disparity-like, higher=closer
        disp = pred.squeeze().float().cpu().numpy()
        if disp.shape != (H, W):
            disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR)

        scale = W / recon.cameras[img.camera_id].width  # full-res keypoints -> depth res
        px, py, z = sparse_samples(recon, img, scale)
        ab = None
        if len(z):
            ix = np.clip(np.round(px).astype(int), 0, W - 1)
            iy = np.clip(np.round(py).astype(int), 0, H - 1)
            ab = fit_metric(disp[iy, ix], z)
        if ab is None:
            n_skip += 1
            continue
        a, b = ab
        denom = a * disp + b
        depth = np.where(denom > 1e-6, 1.0 / denom, 0.0).astype(np.float32)
        depth[(depth <= 0) | ~np.isfinite(depth)] = 0.0
        np.save(out / f"{Path(img.name).stem}.npy", depth)
        n_ok += 1
        if (i + 1) % 100 == 0 or i + 1 == len(recon.images):
            print(f"  {i + 1}/{len(recon.images)} (aligned {n_ok}, skipped {n_skip})")
    print(f"done -> {out}  (aligned {n_ok}, skipped {n_skip} frames with too few SfM points)")


if __name__ == "__main__":
    main()
