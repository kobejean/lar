"""COLMAP text-model -> gsplat training inputs.

Reads a COLMAP text reconstruction (``cameras.txt`` / ``images.txt`` / ``points3D.txt``
as emitted by the LAR pipeline in ``<session>/colmap/poses_txt``) and turns it into:

  - a list of posed cameras (intrinsics ``K`` + world-to-camera view matrix + on-disk
    image path + size), and
  - the sparse point cloud (xyz + rgb) used to *initialise* the Gaussians.

Why the text model and not the ``.bin``: COLMAP 4.x reworked the binary
``images.bin`` layout around the new rig/frame tables, but ``model_converter`` still
writes a stable, pose-carrying ``images.txt`` (verified: the LAR pipeline exports it
via ``export_poses``). Parsing text sidesteps the 4.x binary-format churn entirely.

Poses in ``images.txt`` are world->camera (COLMAP convention), which is exactly the
viewmat convention gsplat's ``rasterization`` expects, so no inversion is needed.

Only pinhole-family cameras are handled (``PINHOLE`` / ``SIMPLE_PINHOLE``). The LAR
pipeline forces ``PINHOLE`` with ARKit intrinsics, so images are already undistorted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------- parse

@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def _iter_data_lines(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras(path: Path) -> dict[int, CameraModel]:
    cams: dict[int, CameraModel] = {}
    for line in _iter_data_lines(path):
        t = line.split()
        cam_id, model, w, h = int(t[0]), t[1], int(t[2]), int(t[3])
        p = [float(x) for x in t[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            raise ValueError(
                f"camera {cam_id} uses unsupported model {model!r}; the LAR pipeline "
                "emits PINHOLE. Re-run COLMAP with a pinhole model (images must be "
                "undistorted for 3DGS)."
            )
        cams[cam_id] = CameraModel(w, h, fx, fy, cx, cy)
    return cams


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP world->camera quaternion (w, x, y, z) -> 3x3 rotation."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


@dataclass
class CameraView:
    image_id: int
    name: str
    camera_id: int
    viewmat: np.ndarray  # (4, 4) world-to-camera
    K: np.ndarray        # (3, 3), at full resolution
    width: int
    height: int


def read_images(path: Path, cameras: dict[int, CameraModel]) -> list[CameraView]:
    """Two *physical* lines per image: a pose header, then a POINTS2D line.

    The POINTS2D line can be **empty** (a refined image that kept its pose but has no
    surviving 2D points — valid COLMAP). So we pair by physical line position rather than
    filtering blanks first: a blank-skipping pass would drop those empty lines and desync
    every subsequent pose. Comments only appear in the header block, never mid-record.
    """
    views: list[CameraView] = []
    with open(path) as f:
        lines = f.read().split("\n")

    i, n = 0, len(lines)
    while i < n:
        header = lines[i].strip()
        if not header or header.startswith("#"):
            i += 1
            continue
        i += 2  # consume the header AND its POINTS2D line (which may be empty)

        h = header.split()
        image_id = int(h[0])
        qw, qx, qy, qz = (float(x) for x in h[1:5])
        tx, ty, tz = (float(x) for x in h[5:8])
        camera_id = int(h[8])
        name = h[9]

        R = _qvec_to_rotmat(qw, qx, qy, qz)
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = [tx, ty, tz]

        cam = cameras[camera_id]
        views.append(
            CameraView(image_id, name, camera_id, viewmat, cam.K(), cam.width, cam.height)
        )
    views.sort(key=lambda v: v.name)
    return views


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz (N,3) float32, rgb (N,3) float32 in [0,1]); tracks are ignored."""
    xyz, rgb = [], []
    for line in _iter_data_lines(path):
        t = line.split()
        xyz.append((float(t[1]), float(t[2]), float(t[3])))
        rgb.append((int(t[4]), int(t[5]), int(t[6])))
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    return xyz, rgb


# --------------------------------------------------------------------------- dataset

def camera_center(viewmat: np.ndarray) -> np.ndarray:
    """Camera position in world coords: C = -R^T t for a world->camera viewmat."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    return -R.T @ t


class ColmapDataset:
    """Posed images + init points, pre-decoded to the training resolution in RAM.

    Images are decoded once and cached as (H, W, 3) float32 in [0,1] at the downscaled
    resolution, so the training loop only pays a host->device copy per step. At park
    scale prefer ``data_factor >= 2`` to keep the RAM cache (and VRAM) in budget.
    """

    def __init__(
        self,
        model_dir: str | Path,
        image_dir: str | Path,
        data_factor: int = 1,
        limit: int | None = None,
        log=print,
    ):
        model_dir = Path(model_dir)
        image_dir = Path(image_dir)
        self.data_factor = data_factor

        cameras = read_cameras(model_dir / "cameras.txt")
        views = read_images(model_dir / "images.txt", cameras)
        if limit is not None:
            views = views[:limit]
        self.points_xyz, self.points_rgb = read_points(model_dir / "points3D.txt")

        self.views: list[CameraView] = []
        self.images: list[np.ndarray] = []  # float32 (H,W,3) in [0,1]
        for i, v in enumerate(views):
            img_path = image_dir / v.name
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"could not read image {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            v2, rgb = self._downscale(v, rgb)
            self.views.append(v2)
            self.images.append(rgb.astype(np.float32) / 255.0)
            if (i + 1) % 100 == 0 or i + 1 == len(views):
                log(f"  loaded {i + 1}/{len(views)} images")

        self.scene_scale = self._scene_scale()
        log(
            f"dataset: {len(self.views)} views, {len(self.points_xyz)} init points, "
            f"scene_scale={self.scene_scale:.2f}m (data_factor={data_factor})"
        )

    def _downscale(self, v: CameraView, rgb: np.ndarray) -> tuple[CameraView, np.ndarray]:
        f = self.data_factor
        if f <= 1:
            return v, rgb
        H, W = rgb.shape[:2]
        nW, nH = W // f, H // f
        rgb = cv2.resize(rgb, (nW, nH), interpolation=cv2.INTER_AREA)
        # Scale intrinsics to match. Use the achieved size, not W/f, so rounding
        # of odd dimensions stays consistent with the resized image.
        sx, sy = nW / W, nH / H
        K = v.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return CameraView(v.image_id, v.name, v.camera_id, v.viewmat, K, nW, nH), rgb

    def _scene_scale(self) -> float:
        centers = np.stack([camera_center(v.viewmat) for v in self.views])
        centroid = centers.mean(axis=0)
        dists = np.linalg.norm(centers - centroid, axis=1)
        return float(np.mean(dists)) * 1.1 + 1e-6

    def __len__(self) -> int:
        return len(self.views)
