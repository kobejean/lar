#!/usr/bin/env python3
"""Render a few RGB previews from an already-trained point_cloud.ply.

Reloads the exported INRIA-layout Gaussians and rasterises them from a spread of
training cameras -- a quick way to eyeball a finished run from more than the single
end-of-training view train.py dumps.

  uv run --extra gsplat python render_previews.py \
      --ply   ../../output/<run>/point_cloud.ply \
      --model ../../output/<run-refined>/colmap/sparse/0 \
      --num 8 --data-factor 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from gsplat import rasterization
from colmap_dataset import read_cameras, read_images


def load_ply(path: Path, device: str):
    """Parse an INRIA 3DGS binary PLY back into gsplat rasterization inputs."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fields, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                fields.append(line.split()[-1])
        data = np.fromfile(f, dtype="<f4", count=n * len(fields)).reshape(n, len(fields))

    col = {name: i for i, name in enumerate(fields)}
    means = torch.from_numpy(data[:, [col["x"], col["y"], col["z"]]].copy())
    f_dc = data[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]          # (N, 3)
    rest_names = sorted((k for k in col if k.startswith("f_rest_")),
                        key=lambda s: int(s.split("_")[-1]))
    f_rest = data[:, [col[k] for k in rest_names]]                          # (N, 3*(K-1))
    k_minus1 = f_rest.shape[1] // 3
    # INRIA lays f_rest out channel-major: [R band coeffs, G band coeffs, B band coeffs].
    shN = f_rest.reshape(n, 3, k_minus1).transpose(0, 2, 1)                 # (N, K-1, 3)
    sh = np.concatenate([f_dc[:, None, :], shN], axis=1)                    # (N, K, 3)
    sh_degree = int(round(sh.shape[1] ** 0.5)) - 1

    opacities = torch.sigmoid(torch.from_numpy(data[:, [col["opacity"]]].copy()).squeeze(-1))
    scales = torch.exp(torch.from_numpy(
        data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]].copy()))
    quats = torch.from_numpy(data[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]].copy())
    return dict(
        means=means.to(device), colors=torch.from_numpy(sh.copy()).to(device),
        opacities=opacities.to(device), scales=scales.to(device), quats=quats.to(device),
        sh_degree=sh_degree, n=n,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--model", required=True, help="COLMAP text model dir (cameras.txt/images.txt)")
    p.add_argument("--out", default=None, help="output dir (default: alongside the ply)")
    p.add_argument("--num", type=int, default=8, help="number of views to render")
    p.add_argument("--data-factor", type=int, default=2)
    args = p.parse_args()

    device = "cuda"
    ply = Path(args.ply)
    out = Path(args.out) if args.out else ply.parent
    out.mkdir(parents=True, exist_ok=True)

    g = load_ply(ply, device)
    print(f"loaded {g['n']} Gaussians (sh_degree={g['sh_degree']}) from {ply}")

    model_dir = Path(args.model)
    cameras = read_cameras(model_dir / "cameras.txt")
    views = read_images(model_dir / "images.txt", cameras)
    idxs = np.linspace(0, len(views) - 1, args.num).round().astype(int)
    print(f"{len(views)} cameras; rendering {len(idxs)} evenly-spaced views")

    f = args.data_factor
    for j, vi in enumerate(idxs):
        v = views[int(vi)]
        W, H = v.width // f, v.height // f
        K = v.K.copy()
        K[0, :] *= W / v.width
        K[1, :] *= H / v.height
        viewmat = torch.from_numpy(v.viewmat).float().to(device)
        Kt = torch.from_numpy(K).float().to(device)
        with torch.no_grad():
            renders, _, _ = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"],
                opacities=g["opacities"], colors=g["colors"],
                viewmats=viewmat[None], Ks=Kt[None], width=W, height=H,
                sh_degree=g["sh_degree"], packed=True, render_mode="RGB",
            )
        rgb = renders[0].clamp(0, 1).cpu().numpy()
        img = (rgb * 255).clip(0, 255).astype(np.uint8)
        path = out / f"render_{j:02d}_{Path(v.name).stem}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  [{j + 1}/{len(idxs)}] {v.name} -> {path.name}")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
