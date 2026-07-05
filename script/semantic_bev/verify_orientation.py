"""Sanity-overlay the camera trajectory on the semantic BEV.

The operator walked the park's paths, so camera positions should fall on PATH/PAVEMENT
cells in a coherent map. This checks that the raster's pixel mapping matches the world
frame (complements the definitive ARKit-Kabsch handedness test, which proves the frame
itself isn't mirrored vs reality).

    uv run python verify_orientation.py --model <poses_txt> --level <out_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from colmap_io import read_model
from pipeline import _qvec2rotmat
from taxonomy import Klass, Role, role_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", required=True, help="dir with level0.npz / .meta.json / _semantic.png")
    ap.add_argument("--prefix", default="level0")
    args = ap.parse_args()

    d = Path(args.level)
    meta = json.load(open(d / f"{args.prefix}.meta.json"))
    npz = np.load(d / f"{args.prefix}.npz")
    semantic = npz["semantic"]
    preview = cv2.imread(str(d / f"{args.prefix}_semantic.png"))

    recon = read_model(args.model)
    centers = np.array([-_qvec2rotmat(im.qvec).T @ im.tvec for im in recon.images.values()])
    u_axis, v_axis = (a for a in (0, 1, 2) if a != meta["up_axis"])

    # same mapping the raster uses: col = (u-origin)/cell, row = (v-origin)/cell  (no flip)
    col = ((centers[:, u_axis] - meta["origin_u"]) / meta["cell_size"]).astype(int)
    row = ((centers[:, v_axis] - meta["origin_v"]) / meta["cell_size"]).astype(int)
    inb = (col >= 0) & (col < meta["cols"]) & (row >= 0) & (row < meta["rows"])
    col, row = col[inb], row[inb]

    # draw trajectory (bright polyline + points)
    for a, b in zip(range(len(col) - 1), range(1, len(col))):
        cv2.line(preview, (col[a], row[a]), (col[b], row[b]), (255, 255, 255), 1)
    for c, r in zip(col, row):
        cv2.circle(preview, (c, r), 1, (0, 0, 255), -1)
    out = d / f"{args.prefix}_trajectory.png"
    big = cv2.resize(preview, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out), big)

    # quantify: what surface do cameras sit over?
    klass_at = semantic[row, col]
    roles = np.array([int(role_of(k)) for k in range(int(max(Klass)) + 1)])
    counts = np.bincount(klass_at, minlength=int(max(Klass)) + 1)
    walkable = sum(counts[k] for k in [int(Klass.PATH), int(Klass.PAVEMENT), int(Klass.STAIRS)])
    ground = sum(counts[k] for k in range(len(counts)) if roles[k] == int(Role.GROUND))
    tot = len(col)
    print(f"camera samples in-bounds: {tot}")
    print(f"  on PATH/PAVEMENT/STAIRS: {100 * walkable / tot:.1f}%")
    print(f"  on any GROUND class:     {100 * ground / tot:.1f}%")
    top = sorted(enumerate(counts), key=lambda x: -x[1])[:5]
    print("  breakdown:", ", ".join(f"{Klass(k).name}:{100 * n / tot:.0f}%" for k, n in top if n))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
