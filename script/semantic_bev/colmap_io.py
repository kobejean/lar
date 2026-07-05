"""Minimal COLMAP text-model reader for the semantic-BEV pipeline.

We only need three things from a COLMAP reconstruction:
  - 3D points (position + colour) and their *tracks* (which image observed them, at
    which 2D keypoint index),
  - per-image 2D keypoints (so a track entry resolves to an exact pixel), and
  - camera intrinsics (kept for the later dense-mask reprojection stage).

Crucially, a point's track already tells us the exact pixel it was seen at in every
observing image, so labelling a point with a semantic class needs *no* reprojection --
we just sample each observing image's segmentation mask at the stored pixel.

Only the text format (``poses_txt/``) is parsed here; that's what the LAR COLMAP
pipeline emits alongside the ``.bin`` model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # model-specific; PINHOLE -> [fx, fy, cx, cy]


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (4,) world-to-camera quaternion [qw, qx, qy, qz]
    tvec: np.ndarray  # (3,) world-to-camera translation
    camera_id: int
    name: str
    xys: np.ndarray  # (N, 2) keypoint pixel coords
    point3d_ids: np.ndarray  # (N,) POINT3D_ID per keypoint (-1 if not triangulated)


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray  # (3,)
    rgb: np.ndarray  # (3,) uint8
    error: float
    image_ids: np.ndarray  # (T,) observing image ids
    point2d_idxs: np.ndarray  # (T,) keypoint index within each observing image


@dataclass
class Reconstruction:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points3d: dict[int, Point3D]

    @property
    def num_points(self) -> int:
        return len(self.points3d)


def _read_lines(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _read_lines(path):
        t = line.split()
        cam_id = int(t[0])
        cameras[cam_id] = Camera(
            id=cam_id,
            model=t[1],
            width=int(t[2]),
            height=int(t[3]),
            params=np.array(t[4:], dtype=np.float64),
        )
    return cameras


def read_images_text(path: Path) -> dict[int, Image]:
    """Two *physical* lines per image: a pose header, then a (X, Y, POINT3D_ID) list.

    The POINTS2D line can be **empty** (a refined image with a pose but no surviving 2D
    points -- valid COLMAP). Pair by physical line position, not by filtering blanks first,
    or those empty lines desync every subsequent pose.
    """
    images: dict[int, Image] = {}
    with open(path) as f:
        lines = f.read().split("\n")

    i, n = 0, len(lines)
    while i < n:
        header = lines[i].strip()
        if not header or header.startswith("#"):
            i += 1
            continue
        pts_line = lines[i + 1] if i + 1 < n else ""
        i += 2

        h = header.split()
        img_id = int(h[0])
        qvec = np.array(h[1:5], dtype=np.float64)
        tvec = np.array(h[5:8], dtype=np.float64)
        camera_id = int(h[8])
        name = h[9]

        toks = pts_line.split()
        if toks:
            vals = np.array(toks, dtype=np.float64).reshape(-1, 3)
            xys, pids = vals[:, :2].copy(), vals[:, 2].astype(np.int64)
        else:
            xys, pids = np.empty((0, 2)), np.empty((0,), dtype=np.int64)
        images[img_id] = Image(
            id=img_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name,
            xys=xys, point3d_ids=pids,
        )
    return images


def read_points3d_text(path: Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    for line in _read_lines(path):
        t = line.split()
        pid = int(t[0])
        xyz = np.array(t[1:4], dtype=np.float64)
        rgb = np.array(t[4:7], dtype=np.uint8)
        error = float(t[7])
        track = np.array(t[8:], dtype=np.int64).reshape(-1, 2)
        points[pid] = Point3D(
            id=pid,
            xyz=xyz,
            rgb=rgb,
            error=error,
            image_ids=track[:, 0].copy(),
            point2d_idxs=track[:, 1].copy(),
        )
    return points


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion [qw,qx,qy,qz] -> 3x3 world-to-camera rotation."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_model(model_dir: str | Path) -> Reconstruction:
    """Read a COLMAP text model directory (``cameras.txt``/``images.txt``/``points3D.txt``)."""
    d = Path(model_dir)
    return Reconstruction(
        cameras=read_cameras_text(d / "cameras.txt"),
        images=read_images_text(d / "images.txt"),
        points3d=read_points3d_text(d / "points3D.txt"),
    )
