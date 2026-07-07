"""Pure-geometry ground model: gravity-aligned DEM + ground/vertical split.

This is the *geometry front-end* for the semantic-BEV pipeline, deliberately kept
independent of semantics. Given a point cloud (COLMAP sparse points today, MVS/2DGS
surfels later) it produces three things the rest of the pipeline consumes:

  1. a **gravity-aligned frame** (full rotation, not axis-snapping) so "up" is true
     gravity and the two horizontal axes are level;
  2. a robust **ground height field** (DEM) -- the lower envelope of the cloud, pinned
     to the ground and smoothed only across *observed* cells;
  3. a per-point **height-above-ground (HAG)** and a ground / vertical / floater split.

Why geometry-first (vs. the semantics-defines-ground design in ground_model.py):
On the refined LAR model the point tracks/colours/errors are stripped, so track-pixel
semantic voting cannot run -- but the geometry is still excellent (ground sharp to
~15-20 cm where cameras walked). Nailing the DEM + HAG first gives every later stage a
clean scaffold: height for the BEV, a ground mask for semantic projection, and a
vertical/obstacle mask for occupancy. Semantics then colours this surface via dense-mask
projection (poses + intrinsics), which needs no tracks.

Design notes that matter:
- **Gravity is measured, not guessed.** ``gravity_up`` averages camera image-up over all
  frames (phones held upright). A 1 deg residual tilt is ~1.5 m of false slope across a
  90 m park, so we rotate by the *full* vector and keep the horizontal axes level.
- **DEM = robust lower envelope.** Ground points form a razor-sharp bottom cluster;
  vegetation stacks above it. A low per-cell quantile seeds the surface; an iterative
  refit inside an *asymmetric* band (tight below, looser above) locks it to the ground
  without climbing into the bush/canopy layer.
- **Smooth only within coverage.** Most cells (a walked park is mostly unvisited) have no
  ground points; naive blurring bleeds the flat extrapolated fill up into real ground.
  We use normalised convolution (blur observed / blur mask) so unobserved cells never
  drag the surface, then nearest-fill the remainder purely for a continuous raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ----- gravity alignment -------------------------------------------------------

def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def gravity_up(qvecs: np.ndarray) -> np.ndarray:
    """World-space up unit vector, averaged over camera orientations.

    COLMAP cameras look +Z_cam with +Y_cam pointing *down* in the image, so world-up for
    an upright phone is ``-R[1, :]``. Averaging over frames cancels per-frame tilt.
    ``qvecs`` is (N, 4) COLMAP quaternions [qw, qx, qy, qz].
    """
    ups = np.array([-_qvec2rotmat(q)[1, :] for q in qvecs])
    m = ups.mean(0)
    return m / (np.linalg.norm(m) + 1e-12)


def align_rotation(up: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping world -> local so gravity-up becomes +Z, horizontals level.

    Rows are the local axes in world coords; ``local = R @ world``.
    """
    up = up / np.linalg.norm(up)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - up * (a @ up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up])


# ----- grid / DEM --------------------------------------------------------------

@dataclass
class GridSpec:
    cell_size: float
    origin_u: float           # world (local-frame) coord of column 0
    origin_v: float           # world (local-frame) coord of row 0
    cols: int
    rows: int


@dataclass
class GroundField:
    spec: GridSpec
    dem: np.ndarray           # (rows, cols) float32 ground height (metres, +up)
    coverage: np.ndarray      # (rows, cols) bool: cell had ground points (DEM trusted)
    R: np.ndarray             # (3,3) world->local gravity-aligned rotation
    up: np.ndarray            # (3,) world-space gravity-up unit vector

    def cell_of(self, uv: np.ndarray) -> np.ndarray:
        """Map local-frame (N,2) horizontal coords -> flat cell ids (clipped)."""
        ix = np.clip(((uv[:, 0] - self.spec.origin_u) / self.spec.cell_size).astype(np.int64),
                     0, self.spec.cols - 1)
        iy = np.clip(((uv[:, 1] - self.spec.origin_v) / self.spec.cell_size).astype(np.int64),
                     0, self.spec.rows - 1)
        return iy * self.spec.cols + ix

    def height_at(self, uv: np.ndarray) -> np.ndarray:
        """Ground height sampled at local-frame (N,2) horizontal coords."""
        return self.dem.reshape(-1)[self.cell_of(uv)]


def _cell_quantile(cell_ids: np.ndarray, values: np.ndarray, ncells: int,
                   q: float, mincount: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell quantile + count (vectorised group-by via sort)."""
    out = np.full(ncells, np.nan, np.float32)
    cnt = np.zeros(ncells, np.int32)
    if len(cell_ids) == 0:
        return out, cnt
    order = np.argsort(cell_ids, kind="stable")
    cs, vs = cell_ids[order], values[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(cs)]))
    for s, e in zip(starts, ends):
        c = cs[s]
        cnt[c] = e - s
        if e - s >= mincount:
            out[c] = np.quantile(vs[s:e], q)
    return out, cnt


def _nearest_fill(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid cells with nearest valid value (cv2 distance transform, no scipy)."""
    if valid.all() or not valid.any():
        return grid
    holes = (~valid).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        holes, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(labels.max() + 1, grid.dtype)
    lut[labels[valid]] = grid[valid]
    return lut[labels]


def _normalised_blur(field: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``field`` using only ``valid`` cells (normalised convolution).

    Unobserved cells contribute nothing and are not dragged toward the fill value, so
    real ground never gets pulled up by the flat extrapolated background.
    """
    w = valid.astype(np.float32)
    num = cv2.GaussianBlur(np.where(valid, field, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), field)
    return out.astype(np.float32)


def build_dem(local_uv: np.ndarray, height: np.ndarray, R: np.ndarray, up: np.ndarray, *,
              cell_size: float = 0.5, bounds_pct: float = 0.5,
              seed_q: float = 0.08, refit_q: float = 0.25,
              band_below: float = 0.25, band_above: float = 0.35,
              iters: int = 4, smooth_sigma: float = 1.2, min_seed: int = 3,
              log=print) -> GroundField:
    """Robust lower-envelope ground DEM from gravity-aligned points.

    ``local_uv`` is (N,2) horizontal coords in the gravity-aligned frame, ``height`` the
    matching (N,) up-axis coord. Seeds with a low per-cell quantile then iteratively
    refits inside an asymmetric band so the surface locks to the ground, not vegetation.
    """
    u, v = local_uv[:, 0], local_uv[:, 1]
    lo_u, hi_u = np.percentile(u, [bounds_pct, 100 - bounds_pct])
    lo_v, hi_v = np.percentile(v, [bounds_pct, 100 - bounds_pct])
    cols = int(np.ceil((hi_u - lo_u) / cell_size)) + 1
    rows = int(np.ceil((hi_v - lo_v) / cell_size)) + 1
    spec = GridSpec(cell_size, float(lo_u), float(lo_v), cols, rows)
    ncells = rows * cols
    log(f"  grid {cols}x{rows} @ {cell_size} m "
        f"({cols * cell_size:.0f}x{rows * cell_size:.0f} m), {len(height)} pts")

    ix = np.clip(((u - lo_u) / cell_size).astype(np.int64), 0, cols - 1)
    iy = np.clip(((v - lo_v) / cell_size).astype(np.int64), 0, rows - 1)
    cid = iy * cols + ix

    # Seed: low quantile per cell = lower envelope.
    seed, cnt = _cell_quantile(cid, height, ncells, seed_q, min_seed)
    valid = (cnt >= min_seed).reshape(rows, cols)
    g = _normalised_blur(np.nan_to_num(seed).reshape(rows, cols).astype(np.float32),
                         valid, smooth_sigma)
    g = _nearest_fill(g, valid)     # continuous raster for sampling; trust = `coverage`

    # Iterative asymmetric refit: pull the surface onto the ground cluster.
    coverage = valid
    for it in range(iters):
        hag = height - g.reshape(-1)[cid]
        inl = (hag > -band_below) & (hag < band_above)
        refit, rc = _cell_quantile(cid[inl], height[inl], ncells, refit_q, 2)
        cov = (rc >= 2).reshape(rows, cols)
        gnew = _normalised_blur(np.nan_to_num(refit).reshape(rows, cols).astype(np.float32),
                                cov, smooth_sigma)
        gnew = _nearest_fill(gnew, cov)
        delta = float(np.abs(gnew - g)[cov].mean()) if cov.any() else 0.0
        g, coverage = gnew, cov
        log(f"  iter{it}: inliers={inl.mean():.2f} meanΔ={delta:.3f} m "
            f"coverage={cov.mean() * 100:.1f}%")

    # Soften the extrapolated fill: nearest-fill leaves blocky Voronoi steps in
    # unobserved cells (untrusted, but ugly and bad for gradients/hillshade). Blur only
    # the holes; observed cells stay exact.
    holes = ~coverage
    if holes.any():
        soft = cv2.GaussianBlur(g, (0, 0), max(2.0, 8.0 * 0.5 / cell_size))
        g = np.where(coverage, g, soft).astype(np.float32)

    return GroundField(spec, g.astype(np.float32), coverage, R, up)


# ----- per-point classification ------------------------------------------------

GROUND, VERTICAL, FLOATER = 0, 1, 2


def classify_points(local_uv: np.ndarray, height: np.ndarray, gf: GroundField, *,
                    band_below: float = 0.25, band_above: float = 0.35
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return (hag, label) where label in {GROUND, VERTICAL, FLOATER} by height-above-ground."""
    hag = height - gf.height_at(local_uv)
    label = np.full(len(hag), VERTICAL, np.uint8)
    label[hag <= -band_below] = FLOATER
    label[(hag > -band_below) & (hag < band_above)] = GROUND
    return hag, label


# ----- convenience: from a COLMAP reconstruction -------------------------------

def from_reconstruction(recon, *, cell_size: float = 0.5, log=print, **dem_kw):
    """One-call geometry: gravity-align a COLMAP recon, build the DEM, classify points.

    Returns (gf, local_xyz, hag, label). ``local_xyz`` is every point in the gravity-
    aligned frame (columns u, v, height).
    """
    xyz = np.array([p.xyz for p in recon.points3d.values()])
    up = gravity_up(np.array([im.qvec for im in recon.images.values()]))
    R = align_rotation(up)
    local = (R @ xyz.T).T
    axis = int(np.argmax(np.abs(up)))
    tilt = np.degrees(np.arccos(min(abs(up[axis]), 1.0)))
    log(f"gravity up = [{up[0]:+.4f} {up[1]:+.4f} {up[2]:+.4f}]  "
        f"({tilt:.2f} deg off axis {axis})")
    gf = build_dem(local[:, :2], local[:, 2], R, up, cell_size=cell_size, log=log, **dem_kw)
    hag, label = classify_points(local[:, :2], local[:, 2], gf)
    n = len(label)
    log(f"  points: ground={np.mean(label == GROUND) * 100:.0f}% "
        f"vertical={np.mean(label == VERTICAL) * 100:.0f}% "
        f"floater={np.mean(label == FLOATER) * 100:.0f}%  (n={n})")
    return gf, local, hag, label
