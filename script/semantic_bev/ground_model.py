"""The 2.5D ground model: stacked single-valued 2D levels, IMDF-style.

Per the agreed design, a park is a list of :class:`Level`s (one for now; bridges/overpasses
add more later). Each level is a set of *single-valued* 2D rasters over the same X/Z grid:

    height    -- ground surface elevation (metres, up-axis), single-valued so slopes are fine
    semantic  -- Klass id per cell (path / grass / ...)
    occupancy -- free / blocked / unknown
    coverage  -- was this cell actually observed, or filled by interpolation

Because height is single-valued *within* a level, there is no brittle multi-surface
per-cell logic. Multi-level support is "append another Level"; the container never changes.

Geometry here comes from semantic-labelled points (COLMAP now, 2DGS later) -- the builder
only consumes ``(positions, labels, confidence)``, so the geometry source is pluggable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from taxonomy import Klass, Role, color_lut, role_of

FREE, BLOCKED, UNKNOWN_OCC = 0, 1, 2


@dataclass
class GridSpec:
    cell_size: float          # metres per cell
    up_axis: int              # 0=x, 1=y, 2=z (world axis pointing up)
    up_sign: float            # +1 or -1 so that up_sign * pos[up_axis] increases upward
    origin_u: float           # world coord of column 0 (first horizontal axis)
    origin_v: float           # world coord of row 0 (second horizontal axis)
    cols: int
    rows: int

    @property
    def horiz_axes(self) -> tuple[int, int]:
        return tuple(a for a in (0, 1, 2) if a != self.up_axis)  # (u_axis, v_axis)


@dataclass
class Level:
    ordinal: int
    spec: GridSpec
    height: np.ndarray        # (rows, cols) float32, nan where unobserved
    semantic: np.ndarray      # (rows, cols) uint8 Klass
    occupancy: np.ndarray     # (rows, cols) uint8
    coverage: np.ndarray      # (rows, cols) bool


@dataclass
class GroundModel:
    levels: list[Level] = field(default_factory=list)


# ----- helpers -----------------------------------------------------------------

def _grouped_reduce(cell_ids: np.ndarray, values: np.ndarray, ncells: int,
                    how: str, q: float = 0.2):
    """Per-cell reduction. Returns (out[ncells] float, count[ncells] int)."""
    out = np.full(ncells, np.nan, dtype=np.float32)
    count = np.zeros(ncells, dtype=np.int32)
    if len(cell_ids) == 0:
        return out, count
    order = np.argsort(cell_ids, kind="stable")
    cs = cell_ids[order]
    vs = values[order]
    # group boundaries
    bounds = np.flatnonzero(np.diff(cs)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(cs)]))
    for s, e in zip(starts, ends):
        cid = cs[s]
        seg = vs[s:e]
        count[cid] = e - s
        out[cid] = np.quantile(seg, q) if how == "quantile" else seg.mean()
    return out, count


def _grouped_majority(cell_ids: np.ndarray, klasses: np.ndarray, weights: np.ndarray,
                      ncells: int, n_klass: int) -> np.ndarray:
    """Per-cell confidence-weighted majority class."""
    out = np.zeros(ncells, dtype=np.uint8)
    if len(cell_ids) == 0:
        return out
    flat = cell_ids.astype(np.int64) * n_klass + klasses.astype(np.int64)
    acc = np.bincount(flat, weights=weights, minlength=ncells * n_klass)
    acc = acc.reshape(ncells, n_klass)
    nonempty = acc.sum(axis=1) > 0
    out[nonempty] = acc[nonempty].argmax(axis=1).astype(np.uint8)
    return out


def _nearest_fill(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid cells with the nearest valid value (cv2 distance transform, no scipy)."""
    if valid.all() or not valid.any():
        return grid
    holes = (~valid).astype(np.uint8)
    # DIST_LABEL_PIXEL: each zero (valid) pixel gets a unique label; holes inherit nearest.
    _, labels = cv2.distanceTransformWithLabels(
        holes, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    # Map label -> value by reading labels at valid pixels.
    valid_labels = labels[valid]
    lut = np.zeros(labels.max() + 1, dtype=grid.dtype)
    lut[valid_labels] = grid[valid]
    return lut[labels]


# ----- builder -----------------------------------------------------------------

def build_level(positions: np.ndarray, labels: np.ndarray, confidence: np.ndarray,
                cell_size: float = 0.5, up_axis: int = 1, up_sign: float = 1.0,
                ground_quantile: float = 0.2, obstacle_band: tuple[float, float] = (0.3, 2.2),
                bounds_pct: float = 0.02, ground_height_clip: tuple[float, float] = (0.01, 0.99),
                min_obstacle_count: int = 3, obstacle_dominance: float = 1.0,
                occupancy_morph: bool = True,
                fill_holes: bool = True, ordinal: int = 0, log=print) -> Level:
    """Rasterise labelled points into a single ground Level."""
    roles = np.array([int(role_of(k)) for k in range(int(max(Klass)) + 1)])[labels]
    is_ground = roles == int(Role.GROUND)
    is_obstacle = roles == int(Role.OBSTACLE)

    u_axis, v_axis = (a for a in (0, 1, 2) if a != up_axis)
    u = positions[:, u_axis]
    v = positions[:, v_axis]
    height = up_sign * positions[:, up_axis]

    # Drop SfM floaters: keep ground candidates within a robust global height band. Real
    # terrain relief spans the band; points far below/above it are mis-triangulations or
    # mislabels (e.g. a -79 m floater when the ground relief is ~11 m).
    if is_ground.any():
        lo_h, hi_h = np.quantile(height[is_ground], ground_height_clip)
        n_before = int(is_ground.sum())
        is_ground &= (height >= lo_h) & (height <= hi_h)
        dropped = n_before - int(is_ground.sum())
        if dropped:
            log(f"  dropped {dropped} ground floaters outside [{lo_h:.1f}, {hi_h:.1f}] m")

    keep = is_ground | is_obstacle
    if not keep.any():
        raise ValueError("no ground/obstacle points to build a level from")

    # Robust bounds from kept points only (outlier SfM points would otherwise inflate grid).
    lo_u, hi_u = np.quantile(u[keep], [bounds_pct, 1 - bounds_pct])
    lo_v, hi_v = np.quantile(v[keep], [bounds_pct, 1 - bounds_pct])
    cols = int(np.ceil((hi_u - lo_u) / cell_size)) + 1
    rows = int(np.ceil((hi_v - lo_v) / cell_size)) + 1
    spec = GridSpec(cell_size, up_axis, up_sign, float(lo_u), float(lo_v), cols, rows)
    ncells = rows * cols
    log(f"  grid {cols}x{rows} @ {cell_size} m ({cols * cell_size:.0f}x{rows * cell_size:.0f} m)")

    def cell_of(mask):
        ix = np.clip(np.floor((u[mask] - lo_u) / cell_size).astype(np.int64), 0, cols - 1)
        iy = np.clip(np.floor((v[mask] - lo_v) / cell_size).astype(np.int64), 0, rows - 1)
        return iy * cols + ix

    # Ground height: low quantile per cell (ground is the low surface after up_sign).
    g_cells = cell_of(is_ground)
    h_flat, g_count = _grouped_reduce(g_cells, height[is_ground], ncells, "quantile", ground_quantile)
    height_raster = h_flat.reshape(rows, cols)
    coverage = (g_count.reshape(rows, cols) > 0)

    # Ground semantics: confidence-weighted majority per cell.
    n_klass = int(max(Klass)) + 1
    sem_flat = _grouped_majority(g_cells, labels[is_ground], confidence[is_ground], ncells, n_klass)
    semantic = sem_flat.reshape(rows, cols)

    if fill_holes:
        filled = _nearest_fill(np.nan_to_num(height_raster, nan=0.0).astype(np.float32), coverage)
        height_out = np.where(coverage, height_raster, filled).astype(np.float32)
        semantic = _nearest_fill(semantic, coverage)
    else:
        height_out = height_raster

    # Occupancy = walkability, not "is there anything above the ground". On narrow forest
    # paths, foliage/branches overhang walkable ground and drop obstacle points into the band,
    # so "any obstacle point -> blocked" false-blocks the paths the operator literally walked.
    # Instead block only where obstacle points *dominate* ground points: a trunk/wall/bush has
    # many obstacle points and little ground beneath it, whereas a path is ground-dominant with
    # only overhang above. Then require a minimum count and drop isolated specks.
    o_cells = cell_of(is_obstacle)
    delta = height[is_obstacle] - height_out.reshape(-1)[o_cells]
    lo_b, hi_b = obstacle_band
    in_band = o_cells[(delta >= lo_b) & (delta <= hi_b)]
    obs_count = np.bincount(in_band, minlength=ncells).reshape(rows, cols)
    grd_count = g_count.reshape(rows, cols)
    blocked = (obs_count >= min_obstacle_count) & (obs_count >= obstacle_dominance * grd_count)
    raw_blocked = int(blocked.sum())

    if occupancy_morph and blocked.any():
        b = blocked.astype(np.float32)
        neighbours = cv2.filter2D(b, -1, np.ones((3, 3), np.float32),
                                  borderType=cv2.BORDER_CONSTANT) - b
        b = np.where(blocked & (neighbours >= 1), np.uint8(1), np.uint8(0))  # drop lone specks
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # fill pinholes
        blocked = b.astype(bool)

    occ = np.full((rows, cols), UNKNOWN_OCC, dtype=np.uint8)
    occ[coverage] = FREE
    occ[blocked] = BLOCKED
    occupancy = occ

    log(f"  ground cells {int(coverage.sum())}/{ncells} observed"
        f" ({100 * coverage.mean():.1f}%); blocked {raw_blocked} -> {int(blocked.sum())} after cleanup")
    return Level(ordinal, spec, height_out, semantic, occupancy, coverage)


# ----- export ------------------------------------------------------------------

def _colorize_height(h: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    valid = np.isfinite(h) & coverage
    img = np.zeros(h.shape, np.uint8)
    if valid.any():
        lo, hi = np.quantile(h[valid], [0.02, 0.98])
        norm = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    color[~coverage] = (30, 30, 30)
    return color


def save_level(level: Level, out_dir: str | Path, prefix: str = "level0") -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    s = level.spec

    np.savez_compressed(
        d / f"{prefix}.npz",
        height=level.height, semantic=level.semantic,
        occupancy=level.occupancy, coverage=level.coverage,
    )
    with open(d / f"{prefix}.meta.json", "w") as f:
        json.dump({
            "ordinal": level.ordinal, "cell_size": s.cell_size,
            "up_axis": s.up_axis, "up_sign": s.up_sign,
            "origin_u": s.origin_u, "origin_v": s.origin_v,
            "cols": s.cols, "rows": s.rows,
        }, f, indent=2)

    # Orientation: col = +u (=+X), row = +v (=+Z). Shown as-is this is a NON-mirrored
    # top-down (screen-right=+X, screen-up=-Z), verified against the ARKit trajectory
    # (Kabsch det=+1, RMSE~0). Do NOT flipud -- that mirrors the map left/right.
    def up(img):
        return img

    cv2.imwrite(str(d / f"{prefix}_height.png"), up(_colorize_height(level.height, level.coverage)))

    lut = np.array(color_lut(), dtype=np.uint8)
    sem_rgb = lut[level.semantic]
    sem_rgb[~level.coverage] = (30, 30, 30)
    cv2.imwrite(str(d / f"{prefix}_semantic.png"), up(cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR)))

    occ_rgb = np.zeros((*level.occupancy.shape, 3), np.uint8)
    occ_rgb[level.occupancy == FREE] = (60, 180, 60)
    occ_rgb[level.occupancy == BLOCKED] = (40, 40, 220)
    occ_rgb[level.occupancy == UNKNOWN_OCC] = (40, 40, 40)
    cv2.imwrite(str(d / f"{prefix}_occupancy.png"), up(occ_rgb))
