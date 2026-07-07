#!/usr/bin/env python3
"""Render per-view metric depth from a trained 3DGS/2DGS model -> depth bake-off contract.

Loads a trained ``point_cloud.ply`` (the geometry only — colors are irrelevant for depth)
and renders each training view's **expected depth** (gsplat ``render_mode="RGB+ED"``, i.e.
alpha-normalised metric z-depth). Writes ``<out>/<image_stem>.npy`` for the
``semantic_bev`` depth back-projection harness. Serves the **3DGS** backend directly and
the **2DGS** backend once a 2DGS model is trained (same .ply layout, surface-accurate depth).

  uv run --extra gsplat python export_depth.py \
      --session maguro-park-after-itchy --gs-dir ../../output/<name>-gsplat-sem \
      --backend 3dgs --data-factor 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from gsplat import rasterization, rasterization_2dgs
from colmap_dataset import read_cameras, read_images


def _read_float_ply(path: Path) -> dict[str, np.ndarray]:
    """Read an all-float32 binary_little_endian PLY (our exporter's format)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY")
        if b"binary_little_endian" not in f.readline():
            raise ValueError("only binary_little_endian PLY supported")
        count, names = None, []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            elif line.startswith(b"property"):
                names.append(line.split()[-1].decode())
            elif line == b"end_header":
                break
        dt = np.dtype([(n, "<f4") for n in names])
        data = np.frombuffer(f.read(count * dt.itemsize), dtype=dt, count=count)
    return {n: np.ascontiguousarray(data[n]) for n in names}


def load_gaussians(ply_path: Path, device: str):
    """Load geometry (means, quats, scales, opacities) from a 3DGS/2DGS .ply."""
    p = _read_float_ply(ply_path)
    means = np.stack([p["x"], p["y"], p["z"]], axis=1)
    scales = np.stack([p[f"scale_{i}"] for i in range(3)], axis=1)  # log-space
    quats = np.stack([p[f"rot_{i}"] for i in range(4)], axis=1)
    opac = p["opacity"]                                             # logit
    t = lambda a: torch.from_numpy(a).float().to(device)
    return t(means), t(quats), torch.exp(t(scales)), torch.sigmoid(t(opac))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="fills --model/--gs-dir/--out from the layout")
    ap.add_argument("--backend", default="3dgs", help="tag for the output dir (3dgs/2dgs)")
    ap.add_argument("--gs-dir", default=None, help="trained gsplat run dir (has point_cloud.ply)")
    ap.add_argument("--model", default=None, help="COLMAP text model (cameras/poses)")
    ap.add_argument("--out", default=None, help="depth output dir (<stem>.npy)")
    ap.add_argument("--data-factor", type=int, default=2)
    args = ap.parse_args()

    if args.session:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from lar_session import Session
        s = Session(args.session)
        # 2dgs backend reads the surfel model dir; 3dgs (or anything else) the volumetric one.
        gs_mode = "2dgs" if args.backend == "2dgs" else "3dgs"
        args.model = args.model or str(s.best_model())
        args.gs_dir = args.gs_dir or str(s.gsplat_out(semantic=True, mode=gs_mode))
        args.out = args.out or str(s.depth_dir(args.backend))
    if not (args.gs_dir and args.model and args.out):
        ap.error("need --session or explicit --gs-dir/--model/--out")

    assert torch.cuda.is_available(), "depth render needs a CUDA GPU"
    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    means, quats, scales, opac = load_gaussians(Path(args.gs_dir) / "point_cloud.ply", device)
    print(f"loaded {means.shape[0]} Gaussians from {args.gs_dir}")
    colors = torch.zeros(means.shape[0], 3, device=device)  # unused; depth only

    cams = read_cameras(Path(args.model) / "cameras.txt")
    views = read_images(Path(args.model) / "images.txt", cams)
    f = args.data_factor
    for i, v in enumerate(views):
        W, H = v.width // f, v.height // f
        K = v.K.copy()
        K[0, :] *= W / v.width
        K[1, :] *= H / v.height
        vm = torch.from_numpy(v.viewmat).float().to(device)[None]
        Kt = torch.from_numpy(K).float().to(device)[None]
        if args.backend == "2dgs":
            # Surfel rasterizer: median depth is the ray/surface intersection -- the
            # sharpest, most surface-accurate depth a 2DGS model produces. gsplat 1.5.3's
            # 2dgs path needs unpacked + a camera dim on non-SH colors (see train.rasterize).
            _, _, _, _, _, median, _ = rasterization_2dgs(
                means, quats, scales, opac, colors[None], vm, Kt, W, H,
                sh_degree=None, packed=False, render_mode="RGB+ED")
            depth = median[0, ..., 0].cpu().numpy().astype(np.float32)
        else:
            render, _, _ = rasterization(means, quats, scales, opac, colors, vm, Kt, W, H,
                                         sh_degree=None, packed=True, render_mode="RGB+ED")
            depth = render[0, ..., 3].cpu().numpy().astype(np.float32)  # expected metric z-depth
        np.save(out / f"{Path(v.name).stem}.npy", depth)
        if (i + 1) % 100 == 0 or i + 1 == len(views):
            print(f"  rendered depth {i + 1}/{len(views)}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
