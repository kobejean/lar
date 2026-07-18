"""True-colour BEV orthophoto: drape the source RGB onto the DEM, top-down.

Where :mod:`dense_projection` reads the cached *semantic mask* at each ground cell's
reprojection, this reads the source *RGB image* there -- so the output is a photographic
overhead map instead of a class raster. Two differences make it a "high-res" map:

  * **Sub-cell grid.** The DEM `Level` is coarse (e.g. 0.5 m, single-valued height is fine
    at that scale). The orthophoto is rendered at `sub`x finer (e.g. 0.05 m), with each fine
    cell's ground height *bilinearly sampled from the DEM*. Height varies slowly, so a coarse
    DEM upsamples cleanly; colour is where we want the resolution.
  * **RGB accumulation.** Each view contributes an inverse-depth-weighted colour; the per-cell
    result is the weighted mean (`best_view=False`) or the single closest view's colour
    (`best_view=True`, crisper but with exposure seams).

Occlusion trick (same as dense_projection): a canopy-covered path cell reprojects to *tree*
pixels in top-down-ish views and *path* pixels in along-the-path views. When a MaskStore is
given, a view's colour is accepted only where its pixel is a **ground class**, so tree/sky
colours are dropped and any clear ground view wins. Without masks, all in-view colours count.

Geometry (height) comes from the DEM; this only produces the colour raster.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from colmap_io import Reconstruction, qvec2rotmat
from grid_transform import cell_to_world
from ground_model import GridSpec, Level, _nearest_fill
from labeling import MaskStore
from taxonomy import Klass, Role, role_of


def _fine_spec(spec: GridSpec, sub: int) -> GridSpec:
    """A grid over the same footprint/origin as `spec` but `sub`x finer cells."""
    return GridSpec(
        cell_size=spec.cell_size / sub,
        up_axis=spec.up_axis, up_sign=spec.up_sign,
        origin_u=spec.origin_u, origin_v=spec.origin_v,
        cols=spec.cols * sub, rows=spec.rows * sub, v_sign=spec.v_sign,
    )


def project_rgb_ortho(recon: Reconstruction, level: Level, image_ids: list[int],
                      image_dir: str | Path, *, store: MaskStore | None = None,
                      sub: int = 5, max_depth: float = 20.0, best_view: bool = False,
                      log=print) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb (H,W,3) uint8 BGR, observed (H,W) bool) -- a top-down RGB orthophoto.

    `sub` is the super-resolution factor over the DEM grid (fine cell = cell_size/sub).
    If `store` is given, each view's colour is gated to ground-class pixels (occlusion-robust).
    """
    image_dir = Path(image_dir)
    fspec = _fine_spec(level.spec, sub)
    rows, cols = fspec.rows, fspec.cols
    u_axis, v_axis = fspec.horiz_axes
    ncells = rows * cols
    log(f"  ortho grid {cols}x{rows} @ {fspec.cell_size:.3f} m "
        f"({sub}x the {level.spec.cell_size:.2f} m DEM)")

    # Height for every fine cell: bilinear upsample of the DEM (height varies slowly).
    h_dem = np.nan_to_num(level.height, nan=0.0).astype(np.float32)
    h_fine = cv2.resize(h_dem, (cols, rows), interpolation=cv2.INTER_LINEAR)

    # World point at each fine cell centre (u,v from the grid, up-coord from the DEM height).
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    P = np.zeros((ncells, 3))
    u_world, v_world = cell_to_world(jj.ravel() + 0.5, ii.ravel() + 0.5, fspec)
    P[:, u_axis] = u_world
    P[:, v_axis] = v_world
    P[:, fspec.up_axis] = h_fine.ravel() * fspec.up_sign

    ground_cls = np.array([role_of(k) == Role.GROUND for k in range(int(max(Klass)) + 1)])

    csum = np.zeros((ncells, 3), dtype=np.float32)  # weighted colour accumulator (BGR)
    wsum = np.zeros(ncells, dtype=np.float32)
    best_w = np.zeros(ncells, dtype=np.float32) if best_view else None

    for n, img_id in enumerate(image_ids):
        im = recon.images[img_id]
        img = cv2.imread(str(image_dir / im.name), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            continue
        H, W = img.shape[:2]

        R = qvec2rotmat(im.qvec)
        cam = P @ R.T + im.tvec
        z = cam[:, 2]
        near = (z > 0.1) & (z < max_depth)
        if not near.any():
            continue
        fx, fy, cx, cy = recon.cameras[im.camera_id].params[:4]
        px = fx * cam[:, 0] / z + cx
        py = fy * cam[:, 1] / z + cy
        inb = near & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = np.nonzero(inb)[0]
        if len(idx) == 0:
            continue
        xs = px[idx].astype(np.int64)
        ys = py[idx].astype(np.int64)

        if store is not None:
            mask = store.load(im.name)
            if mask.shape[:2] != (H, W):  # mask cached at a different scale than the image
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            keep = ground_cls[mask[ys, xs]]
            if not keep.any():
                continue
            idx, xs, ys = idx[keep], xs[keep], ys[keep]

        w = (1.0 / z[idx]).astype(np.float32)  # inverse-depth: near, reliable views weigh more
        colours = img[ys, xs].astype(np.float32)
        if best_view:
            take = w > best_w[idx]
            sel = idx[take]
            best_w[sel] = w[take]
            csum[sel] = colours[take]
        else:
            np.add.at(csum, idx, colours * w[:, None])
            np.add.at(wsum, idx, w)
        if (n + 1) % 100 == 0 or n + 1 == len(image_ids):
            log(f"  projected {n + 1}/{len(image_ids)} views")

    observed = (best_w > 0) if best_view else (wsum > 0)
    rgb = np.zeros((ncells, 3), dtype=np.float32)
    if best_view:
        rgb[observed] = csum[observed]
    else:
        rgb[observed] = csum[observed] / wsum[observed, None]
    log(f"  ortho: {int(observed.sum())}/{ncells} cells coloured ({100 * observed.mean():.1f}%)")

    rgb = rgb.reshape(rows, cols, 3)
    obs2d = observed.reshape(rows, cols)
    # Fill gaps (unobserved / occluded cells) with the nearest coloured cell, per channel.
    for c in range(3):
        rgb[..., c] = _nearest_fill(rgb[..., c].astype(np.uint8), obs2d)
    return rgb.astype(np.uint8), obs2d


def save_ortho(rgb: np.ndarray, observed: np.ndarray, out_dir: str | Path,
               prefix: str = "level0") -> None:
    """Write the orthophoto (BGR) and an alpha-masked version showing only observed cells."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{prefix}_rgb.png"), rgb)  # rgb is BGR (cv2 convention)
    bgra = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = np.where(observed, 255, 0).astype(np.uint8)
    cv2.imwrite(str(d / f"{prefix}_rgb_masked.png"), bgra)
