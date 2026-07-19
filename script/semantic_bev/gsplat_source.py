"""Geometry source: a trained *semantic* 3DGS model -> (positions, labels, confidence).

An alternative to the COLMAP sparse-point source (`colmap_io` + `labeling`). A semantic
3DGS run (``script/gsplat/train.py --semantic``) bakes a taxonomy class into every
Gaussian, so the BEV builder can consume the Gaussians directly:

  - **denser** than COLMAP tracks (a whole optimised splat cloud, not just triangulated
    keypoints), and
  - **already labelled** — no segmentation/voting pass needed here; the labels were
    distilled multi-view-consistently during Phase 2 of the gsplat trainer.

Returns the same ``(positions, labels, confidence)`` triple as the COLMAP path, so
``ground_model.build_level`` is unchanged. The gsplat model is in the *same world
coordinates* as the COLMAP model it was trained from, so gravity/up detection (which needs
camera orientations) still comes from that COLMAP model in ``pipeline.py``.

Consumes the gsplat run's export dir:
  point_cloud.ply            -- Gaussian means (+ opacity, used to prune floaters)
  point_cloud_labels.npy     -- per-Gaussian Klass id  (row-aligned with the .ply)
  point_cloud_confidence.npy -- per-Gaussian peak softmax prob (optional; 1.0 if absent)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_float_ply(path: Path) -> dict[str, np.ndarray]:
    """Read an all-float32 binary_little_endian PLY (the format script/gsplat writes).

    Only 'property float <name>' vertex fields are supported -- which is exactly what the
    gsplat exporter emits. Returns {field_name: (N,) float32}.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = f.readline().strip()
        if b"binary_little_endian" not in fmt:
            raise ValueError(f"{path}: only binary_little_endian PLY is supported, got {fmt!r}")
        count, names = None, []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            elif line.startswith(b"property"):
                parts = line.split()
                if parts[1] != b"float":
                    raise ValueError(f"{path}: non-float property {line!r} not supported")
                names.append(parts[-1].decode())
            elif line == b"end_header":
                break
        dtype = np.dtype([(n, "<f4") for n in names])
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
    return {n: np.ascontiguousarray(data[n]) for n in names}


def load_gsplat_points(gsplat_dir: str | Path, min_opacity: float = 0.1,
                       log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a semantic 3DGS export as ``(positions (N,3), labels (N,) uint8, conf (N,))``.

    Gaussians with opacity below ``min_opacity`` are dropped: 3DGS leaves many faint
    floaters that are visually minor but geometric noise for a ground/height model.
    """
    d = Path(gsplat_dir)
    ply = _read_float_ply(d / "point_cloud.ply")
    positions = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)
    # 'opacity' is stored as a logit (pre-sigmoid), matching the trainer's representation.
    opacity = 1.0 / (1.0 + np.exp(-ply["opacity"])) if "opacity" in ply else np.ones(len(positions))

    labels = np.load(d / "point_cloud_labels.npy").astype(np.uint8)
    conf_path = d / "point_cloud_confidence.npy"
    if conf_path.exists():
        confidence = np.load(conf_path).astype(np.float32)
    else:
        confidence = np.ones(len(labels), dtype=np.float32)
        log("  no _confidence.npy (older gsplat run); using uniform confidence 1.0")

    if not (len(positions) == len(labels) == len(confidence)):
        raise ValueError(
            f"gsplat export row mismatch: {len(positions)} points, {len(labels)} labels, "
            f"{len(confidence)} conf -- point_cloud.ply and the .npy files must be aligned"
        )

    keep = opacity >= min_opacity
    log(f"  gsplat: {len(positions)} Gaussians, kept {int(keep.sum())} "
        f"with opacity >= {min_opacity}")
    return positions[keep], labels[keep], confidence[keep]
