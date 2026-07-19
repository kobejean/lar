"""Depth back-projection geometry source — the shared harness for the depth bake-off.

Every "real geometry" approach (COLMAP MVS, 2DGS, mono-depth, 3DGS) reduces to the same
thing: **per-view metric depth**. This module is the common back-end. It back-projects
every labelled pixel to its *true* 3D position using that depth — instead of assuming the
pixel lies on the ground, which is what makes `dense_projection` smear above-ground
objects. Output is the standard ``(positions, labels, confidence)`` triple, so
``ground_model.build_level`` is unchanged and every backend is compared apples-to-apples.

Depth-map contract (one directory per backend, produced by that backend's exporter):

    <depth_dir>/<image_stem>.npy        float32 (H, W) metric z-depth in metres; <=0 / NaN = invalid
    <depth_dir>/<image_stem>.conf.npy   optional float32 (H, W) confidence in [0, 1]

Depth resolution may differ from the source image; the pinhole intrinsics are scaled to
the depth resolution. Points are accumulated across all frames and voxel-deduped (mean
position, confidence-weighted majority label per voxel) so memory stays bounded.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from colmap_io import Reconstruction
from labeling import MaskStore
from taxonomy import Klass


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pinhole_params(cam) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) from a PINHOLE / SIMPLE_PINHOLE camera."""
    p = cam.params
    if cam.model == "PINHOLE":
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    if cam.model == "SIMPLE_PINHOLE":
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    raise ValueError(f"camera model {cam.model!r} not supported for back-projection")


def _load_depth(depth_dir: Path, stem: str):
    dp = depth_dir / f"{stem}.npy"
    if not dp.exists():
        return None, None
    depth = np.load(dp).astype(np.float32)
    cp = depth_dir / f"{stem}.conf.npy"
    conf = np.load(cp).astype(np.float32) if cp.exists() else None
    return depth, conf


def backproject_labeled_points(
    recon: Reconstruction, store: MaskStore, depth_dir: str | Path, image_ids: list[int],
    *, stride: int = 4, voxel: float = 0.05, min_conf: float = 0.0, log=print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project labelled pixels via per-view depth -> voxel-deduped labelled cloud."""
    depth_dir = Path(depth_dir)
    n_klass = int(max(Klass)) + 1
    all_pos, all_lab, all_conf = [], [], []
    n_frames = raw = 0

    for img_id in image_ids:
        im = recon.images[img_id]
        depth, conf = _load_depth(depth_dir, Path(im.name).stem)
        if depth is None:
            continue
        H, W = depth.shape
        cam = recon.cameras[im.camera_id]
        fx, fy, cx, cy = _pinhole_params(cam)
        sx, sy = W / cam.width, H / cam.height           # scale intrinsics to depth res
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy

        mask = store.load(im.name)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        ys = np.arange(0, H, stride)
        xs = np.arange(0, W, stride)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        d = depth[gy, gx]
        lab = mask[gy, gx]
        valid = (d > 0) & np.isfinite(d) & (lab != int(Klass.UNKNOWN))
        if conf is not None:
            cvals = conf[gy, gx]
            valid &= cvals >= min_conf
        if not valid.any():
            continue

        u, v, dd, ll = gx[valid].astype(np.float32), gy[valid].astype(np.float32), d[valid], lab[valid]
        cw = conf[gy, gx][valid].astype(np.float32) if conf is not None else np.ones(len(dd), np.float32)
        # pixel + metric z-depth -> camera coords -> world coords.
        cam_pts = np.stack([(u - cx) / fx * dd, (v - cy) / fy * dd, dd], axis=1)
        R = _qvec2rotmat(im.qvec)
        C = -R.T @ im.tvec
        world = cam_pts @ R + C                          # world_i = R^T @ cam_i + C

        all_pos.append(world.astype(np.float32))
        all_lab.append(ll.astype(np.int64))
        all_conf.append(cw)
        n_frames += 1
        raw += len(world)

    if not all_pos:
        raise ValueError(f"no depth maps found in {depth_dir} for the requested images")

    pos = np.concatenate(all_pos)
    lab = np.concatenate(all_lab)
    cw = np.concatenate(all_conf)
    log(f"  back-projected {raw} points from {n_frames} depth maps (stride {stride})")

    # Voxel dedup: mean position + confidence-weighted majority label per voxel.
    vox = np.floor(pos / voxel).astype(np.int64)
    uniq, inv = np.unique(vox, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sums, inv, pos)
    positions = (sums / counts[:, None]).astype(np.float32)

    flat = inv * n_klass + lab
    votes = np.bincount(flat, weights=cw, minlength=len(uniq) * n_klass).reshape(len(uniq), n_klass)
    labels = votes.argmax(1).astype(np.uint8)
    confidence = (votes.max(1) / votes.sum(1)).astype(np.float32)
    log(f"  voxel-deduped to {len(positions)} points @ {voxel} m")
    return positions, labels, confidence
