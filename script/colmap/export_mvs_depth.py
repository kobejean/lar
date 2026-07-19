#!/usr/bin/env python3
"""COLMAP dense MVS -> per-view metric depth for the semantic-BEV depth bake-off.

Runs the standard COLMAP dense pipeline on a (refined) sparse model —
``image_undistorter`` -> ``patch_match_stereo`` (geometric consistency) — and converts the
resulting geometric depth maps into the bake-off contract: ``<out>/<image_stem>.npy``
(float32 metric z-depth, 0 = invalid). Also writes the undistorted model as text to
``<out>/model/`` so the back-projection harness uses matching intrinsics/poses.

Launch inside the COLMAP env so both `colmap` and `uv` resolve, e.g.:
  ~/bin/micromamba run -n colmap uv run python script/colmap/export_mvs_depth.py \
      --session maguro-park-after-itchy --data-factor 2

Notes: MVS is globally-consistent metric geometry, strong on textured/solid surfaces
(vending machines, tables, plazas), weak on thin structures (poles, chair legs) and
low-texture grass. On 8 GB keep --max-image-size modest and cache_size small.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def read_colmap_array(path: Path) -> np.ndarray:
    """Read a COLMAP dense depth/normal map (.bin): '<W>&<H>&<C>&' header then float32."""
    with open(path, "rb") as f:
        w, h, c = np.genfromtxt(f, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        f.seek(0)
        n = 0
        while n < 3:
            if f.read(1) == b"&":
                n += 1
        arr = np.fromfile(f, np.float32)
    arr = arr.reshape((int(w), int(h), int(c)), order="F")
    return np.transpose(arr, (1, 0, 2)).squeeze()


def run(cmd, log):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="fills --model/--images/--out from the layout")
    ap.add_argument("--model", default=None, help="COLMAP sparse model dir (text or bin)")
    ap.add_argument("--images", default=None, help="source images dir")
    ap.add_argument("--out", default=None, help="depth output dir")
    ap.add_argument("--data-factor", type=int, default=2, help="max_image_size = 1920 // this")
    ap.add_argument("--max-image-size", type=int, default=None, help="override MVS image size")
    ap.add_argument("--cache-size", type=int, default=8, help="patch-match GB cache (8 GB card)")
    ap.add_argument("--colmap", default="colmap", help="colmap binary (on PATH inside the env)")
    args = ap.parse_args()

    if args.session:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from lar_session import Session
        s = Session(args.session)
        args.model = args.model or str(s.best_model())
        args.images = args.images or str(s.images)
        args.out = args.out or str(s.depth_dir("mvs"))
    if not (args.model and args.images and args.out):
        ap.error("need --session or explicit --model/--images/--out")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ws = out / "_mvs_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    max_size = args.max_image_size or (1920 // args.data_factor)

    run([args.colmap, "image_undistorter", "--image_path", args.images,
         "--input_path", args.model, "--output_path", ws,
         "--output_type", "COLMAP", "--max_image_size", max_size], print)
    run([args.colmap, "patch_match_stereo", "--workspace_path", ws,
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.cache_size", args.cache_size,
         "--PatchMatchStereo.max_image_size", max_size], print)
    # Undistorted model as text so the harness reads matching intrinsics/poses.
    (out / "model").mkdir(exist_ok=True)
    run([args.colmap, "model_converter", "--input_path", ws / "sparse",
         "--output_path", out / "model", "--output_type", "TXT"], print)

    depth_maps = sorted((ws / "stereo" / "depth_maps").glob("*.geometric.bin"))
    if not depth_maps:
        raise SystemExit("no geometric depth maps produced (patch_match_stereo failed?)")
    print(f"converting {len(depth_maps)} depth maps -> {out}")
    for i, dm in enumerate(depth_maps):
        stem = dm.name[:-len("_image.jpeg.geometric.bin")] + "_image" \
            if dm.name.endswith("_image.jpeg.geometric.bin") else dm.name.split(".")[0]
        depth = read_colmap_array(dm).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        np.save(out / f"{stem}.npy", depth)
        if (i + 1) % 100 == 0 or i + 1 == len(depth_maps):
            print(f"  {i + 1}/{len(depth_maps)}")
    print(f"done -> {out}   (pass --model {out}/model to the BEV harness, or --session handles it)")


if __name__ == "__main__":
    main()
