#!/usr/bin/env python3
"""Post-hoc prune outlier Gaussians from a trained point_cloud.ply.

The metric (un-normalized) park reconstruction is fundamentally good -- opaque, sharp,
median scale ~0.1 m -- but carries a heavy tail of pathological Gaussians (max scale
tens of km, anisotropy in the thousands). Those giant/needle splats smear across the
frame as the near-camera "veil" and the starburst spikes. Training regularizers meant
to prevent them (opacity_reg/scale_reg) are tuned for scenes normalized to unit scale
and collapse this metric scene instead -- so we clean up after the fact rather than
fight the optimizer.

Keeps a Gaussian iff:  max_axis_scale < --max-scale  AND  anisotropy < --max-aniso
(anisotropy = largest/smallest axis; flat ground/canopy disks are legitimately ~8-10x,
so the cut targets only the extreme needles). Optionally also drop opacity < --min-opacity.

Preserves every PLY field, so the output re-opens in any 3DGS viewer. If a row-aligned
<stem>_labels.npy / _confidence.npy sit alongside the input, they're filtered too.

  uv run --extra gsplat python prune_gaussians.py \
      --ply ../../output/<run>/point_cloud.ply \
      --max-scale 2.0 --max-aniso 100 --min-opacity 0.0 \
      --out ../../output/<run>/point_cloud_pruned.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_ply(path: Path):
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
    return data, fields


def write_ply(path: Path, data: np.ndarray, fields: list[str]) -> None:
    data = np.ascontiguousarray(data, dtype="<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {data.shape[0]}\n"
        + "".join(f"property float {f}\n" for f in fields)
        + "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--out", default=None, help="default: <ply stem>_pruned.ply")
    p.add_argument("--max-scale", type=float, default=2.0,
                   help="drop Gaussians whose largest axis (metres) exceeds this")
    p.add_argument("--max-aniso", type=float, default=100.0,
                   help="drop Gaussians whose largest/smallest axis ratio exceeds this")
    p.add_argument("--min-opacity", type=float, default=0.0,
                   help="also drop Gaussians below this opacity (0 = keep all opacities)")
    args = p.parse_args()

    ply = Path(args.ply)
    out = Path(args.out) if args.out else ply.with_name(ply.stem + "_pruned.ply")

    data, fields = read_ply(ply)
    c = {k: i for i, k in enumerate(fields)}
    n = data.shape[0]

    sc = np.exp(data[:, [c["scale_0"], c["scale_1"], c["scale_2"]]])
    smax = sc.max(1)
    aniso = smax / np.clip(sc.min(1), 1e-9, None)
    opacity = 1.0 / (1.0 + np.exp(-data[:, c["opacity"]]))

    keep = (smax < args.max_scale) & (aniso < args.max_aniso)
    if args.min_opacity > 0.0:
        keep &= opacity > args.min_opacity

    write_ply(out, data[keep], fields)
    print(f"{ply.name}: kept {keep.sum()}/{n} ({keep.mean() * 100:.1f}%), "
          f"dropped {(~keep).sum()} -> {out}")

    # Filter any row-aligned semantic sidecars so labels stay in sync with the splats.
    stem = ply.with_name(ply.stem)
    for suffix in ("_labels.npy", "_confidence.npy"):
        side = stem.with_name(stem.name + suffix)
        if side.exists():
            arr = np.load(side)
            if arr.shape[0] == n:
                np.save(out.with_name(out.stem + suffix), arr[keep])
                print(f"  filtered sidecar {side.name} -> {out.stem + suffix}")


if __name__ == "__main__":
    main()
