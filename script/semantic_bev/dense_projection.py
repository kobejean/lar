"""Dense-mask projection: build the ground semantic raster from *all* mask pixels.

The point-vote path samples the dense per-pixel masks only at sparse triangulated points, so
the semantic raster is speckly and low-coverage. This is the CPU-only fallback for when 3DGS
is too heavy: it projects each ground cell centre into every view and reads the cached dense
mask there, so a cell is labelled by however many pixels saw it -- dense, smooth, high-coverage.

Occlusion trick: a canopy-covered path cell projects to *tree* pixels in top-down-ish views
and to *path* pixels in along-the-path views. We only count **ground-class** votes, so the
tree/sky votes are discarded and any clear ground view wins. Votes are inverse-depth weighted
(near, reliable views count more) and depth-capped (distant cells near the horizon, where
occluders live, are dropped).

Geometry (height) still comes from whatever built the Level; this only rebuilds `semantic`.
"""

from __future__ import annotations

import numpy as np

from colmap_io import Reconstruction, qvec2rotmat
from grid_transform import cell_to_world
from ground_model import Level
from labeling import MaskStore
from taxonomy import Klass, Role, role_of


def project_dense_semantics(recon: Reconstruction, store: MaskStore, level: Level,
                            image_ids: list[int], max_depth: float = 20.0,
                            log=print) -> tuple[np.ndarray, np.ndarray]:
    """Return (semantic raster uint8, observed mask bool) from dense-mask projection."""
    spec = level.spec
    rows, cols = spec.rows, spec.cols
    u_axis, v_axis = spec.horiz_axes

    # World-space centre of every cell (height from the Level's field).
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))  # jj=col(=u), ii=row(=v)
    P = np.zeros((rows * cols, 3))
    u_world, v_world = cell_to_world(jj.ravel() + 0.5, ii.ravel() + 0.5, spec)  # cell centres
    P[:, u_axis] = u_world
    P[:, v_axis] = v_world
    P[:, spec.up_axis] = level.height.ravel() * spec.up_sign  # world up-coord = height*sign

    ncells = rows * cols
    nclass = int(max(Klass)) + 1
    votes = np.zeros((ncells, nclass), dtype=np.float32)
    ground_cols = np.array([role_of(k) == Role.GROUND for k in range(nclass)])

    for n, img_id in enumerate(image_ids):
        im = recon.images[img_id]
        R = qvec2rotmat(im.qvec)
        cam = P @ R.T + im.tvec           # (ncells, 3) camera coords
        z = cam[:, 2]
        near = (z > 0.1) & (z < max_depth)
        if not near.any():
            continue
        fx, fy, cx, cy = recon.cameras[im.camera_id].params[:4]
        px = fx * cam[:, 0] / z + cx
        py = fy * cam[:, 1] / z + cy
        mask = store.load(im.name)
        H, W = mask.shape
        inb = near & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = np.nonzero(inb)[0]
        if len(idx) == 0:
            continue
        labels = mask[py[idx].astype(np.int64), px[idx].astype(np.int64)]
        w = (1.0 / z[idx]).astype(np.float32)  # inverse-depth: near views weigh more
        np.add.at(votes, (idx, labels), w)
        if (n + 1) % 200 == 0 or n + 1 == len(image_ids):
            log(f"  projected {n + 1}/{len(image_ids)} views")

    votes[:, ~ground_cols] = 0.0          # ground map: ignore non-ground (occlusion-robust)
    total = votes.sum(1)
    observed = total > 0
    sem = np.zeros(ncells, dtype=np.uint8)
    sem[observed] = votes[observed].argmax(1).astype(np.uint8)
    log(f"  dense projection: {int(observed.sum())}/{ncells} cells ground-labelled "
        f"({100 * observed.mean():.1f}%)")
    return sem.reshape(rows, cols), observed.reshape(rows, cols)
