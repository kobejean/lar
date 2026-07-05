#!/usr/bin/env python3
"""Export a refined lar map (frames.json + map.json) as a COLMAP text model.

lar_refine_colmap writes its result as map.g2o / map.json / frames.json, which
COLMAP's GUI can't open directly. This converts the refined camera poses and
landmark positions into a COLMAP sparse text model (cameras/images/points3D.txt)
so you can inspect the *refined* reconstruction in `colmap gui`.

Reuses the ARKit<->COLMAP convention from colmap_pose.py so it round-trips with
the rest of the pipeline (ARKit camera-to-world -> COLMAP world-to-camera:
R_w2c = diag(1,-1,-1) @ R_c2w^T, t = -R_w2c @ C).

Usage:
    uv run python script/colmap/export_refined_colmap_model.py \
        output/<session>-refined [--out output/<session>-refined/sparse]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from colmap_pose import (
    extract_rotation_translation_from_extrinsics,
    rotation_matrix_to_quaternion,
)

# Capture resolution for these sessions (1920x1440 landscape). Only used for the
# camera WIDTH/HEIGHT header; PINHOLE intrinsics come from the frame itself.
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1440


def parse_intrinsics(intr):
    """9-element column-major 3x3 -> (fx, fy, cx, cy)."""
    K = np.array(intr).reshape(3, 3, order="F")
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refined_dir", help="dir with frames.json + map.json from lar_refine_colmap")
    ap.add_argument("--out", default=None, help="output model dir (default: <refined_dir>/sparse/0)")
    args = ap.parse_args()

    refined = Path(args.refined_dir)
    out = Path(args.out) if args.out else refined / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)

    frames = json.load(open(refined / "frames.json"))
    world = json.load(open(refined / "map.json"))
    landmarks = world["landmarks"]

    # cameras.txt + images.txt: one PINHOLE camera per frame; image_id = frame_id+1
    # to match the 1-indexed ids the pipeline uses in the COLMAP database.
    with open(out / "cameras.txt", "w") as fc, open(out / "images.txt", "w") as fi:
        fc.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fi.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] (empty)\n")
        for fr in frames:
            image_id = fr["id"] + 1
            fx, fy, cx, cy = parse_intrinsics(fr["intrinsics"])
            fc.write(f"{image_id} PINHOLE {IMAGE_WIDTH} {IMAGE_HEIGHT} "
                     f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

            R_w2c, t = extract_rotation_translation_from_extrinsics(
                fr["extrinsics"], apply_colmap_conversion=True)
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
            name = f"{fr['id']:08d}_image.jpeg"
            fi.write(f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {image_id} {name}\n\n")

    # points3D.txt: refined landmark positions (ARKit world coords). Neutral gray,
    # empty track (fine for point-cloud viewing in the GUI).
    with open(out / "points3D.txt", "w") as fp:
        fp.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for lm in landmarks:
            x, y, z = lm["position"]
            fp.write(f"{lm['id']} {x:.6f} {y:.6f} {z:.6f} 180 180 180 1.0\n")

    print(f"Wrote COLMAP model: {out}")
    print(f"  cameras/images: {len(frames)}   points3D: {len(landmarks)}")
    print(f"\nView with:\n  colmap gui --import_path {out} "
          f"--database_path {refined}/  --image_path <session>/colmap")


if __name__ == "__main__":
    main()
