"""CLI + diagnostic renders for the pure-geometry ground model (geometry.py).

Run on a COLMAP text model; emits a DEM hillshade, structure-height map, ground/vertical
density, camera-trajectory overlay, and vertical cross-sections so ground extraction
quality can be judged by eye.

    uv run python geometry_cli.py \
        --model ../../output/maguro-park-after-itchy-refined/sparse/0 \
        --out   ../../output/maguro-park-after-itchy-geom \
        --cell-size 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from colmap_io import read_model
import geometry as G


def _colorize(field, valid, cmap=cv2.COLORMAP_TURBO, lo=None, hi=None):
    v = field[valid]
    if lo is None:
        lo, hi = np.percentile(v, [2, 98]) if valid.any() else (0.0, 1.0)
    n = np.clip((field - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    img[~valid] = (35, 35, 35)
    return img


def _hillshade(dem, cell, az=315.0, alt=45.0):
    gy, gx = np.gradient(dem, cell)
    normal = np.dstack([-gx, -gy, np.ones_like(dem)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    a, e = np.radians(az), np.radians(alt)
    light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return np.clip(normal @ light, 0, 1)


def render(gf: G.GroundField, local, hag, label, cams_local, out: Path, log=print):
    out.mkdir(parents=True, exist_ok=True)
    s = gf.spec
    rows, cols = s.rows, s.cols
    cid = gf.cell_of(local[:, :2])
    flip = np.flipud  # north-up: +v downward in array -> flip for display

    # 1. DEM height, hill-shaded, masked to covered area (extrapolation dimmed).
    shade = _hillshade(gf.dem, s.cell_size)
    dem_rgb = _colorize(gf.dem, gf.coverage)
    dem_rgb = (dem_rgb.astype(np.float32) * (0.35 + 0.65 * shade[..., None])).astype(np.uint8)
    dem_rgb[~gf.coverage] = (dem_rgb[~gf.coverage] * 0.35).astype(np.uint8)
    cv2.imwrite(str(out / "dem_hillshade.png"), flip(dem_rgb))

    # 2. Structure height = 95th pct HAG of vertical points per cell.
    vmask = label == G.VERTICAL
    sh, _ = G._cell_quantile(cid[vmask], hag[vmask], rows * cols, 0.95, 1)
    sh = np.nan_to_num(sh).reshape(rows, cols).clip(0, 8)
    cv2.imwrite(str(out / "structure_height.png"),
                flip(_colorize(sh, sh > 0, cv2.COLORMAP_MAGMA, 0, 6)))

    # 3. Ground (green) vs vertical (red) point density.
    gcnt = np.bincount(cid[label == G.GROUND], minlength=rows * cols).reshape(rows, cols)
    vcnt = np.bincount(cid[vmask], minlength=rows * cols).reshape(rows, cols)
    gv = np.zeros((rows, cols, 3), np.uint8)
    gv[..., 1] = np.clip(gcnt * 40, 0, 255)
    gv[..., 2] = np.clip(vcnt * 40, 0, 255)
    cv2.imwrite(str(out / "ground_vs_vertical.png"), flip(gv))

    # 4. Camera trajectory over the DEM (orientation / registration sanity).
    traj = dem_rgb.copy()
    px = np.clip(((cams_local[:, 0] - s.origin_u) / s.cell_size).astype(int), 0, cols - 1)
    py = np.clip(((cams_local[:, 1] - s.origin_v) / s.cell_size).astype(int), 0, rows - 1)
    for x, y in zip(px, py):
        cv2.circle(traj, (x, y), 1, (0, 255, 255), -1)
    cv2.imwrite(str(out / "camera_trajectory.png"), flip(traj))

    # 5. Vertical cross-sections at 3 u-slices (the real ground-quality test).
    #    Rendered with cv2 (no matplotlib dep): points coloured by class, DEM overlaid
    #    solid where observed / faint where extrapolated.
    u, v, h = local[:, 0], local[:, 1], local[:, 2]
    panels = []
    W, H, pad = 1200, 300, 40
    for uc in np.percentile(u, [35, 50, 65]):
        m = np.abs(u - uc) < 1.0
        canvas = np.full((H, W, 3), 255, np.uint8)
        vmin, vmax = np.percentile(v[m], [0, 100])
        h0 = np.percentile(h[m], 1)
        hmin, hmax = h0 - 1.0, h0 + 8.0

        def to_px(vv, hh):
            x = pad + (vv - vmin) / max(vmax - vmin, 1e-6) * (W - 2 * pad)
            y = H - pad - (hh - hmin) / max(hmax - hmin, 1e-6) * (H - 2 * pad)
            return x.astype(int), y.astype(int)

        # gridlines every 1 m in height
        for hh in np.arange(np.ceil(hmin), hmax, 1.0):
            _, y = to_px(np.array([vmin]), np.array([hh]))
            cv2.line(canvas, (pad, int(y[0])), (W - pad, int(y[0])), (235, 235, 235), 1)
        for lb, col in [(G.GROUND, (34, 187, 34)), (G.VERTICAL, (34, 34, 204)),
                        (G.FLOATER, (204, 136, 34))]:  # BGR
            mm = m & (label == lb)
            xs, ys = to_px(v[mm], h[mm])
            ok = (ys >= 0) & (ys < H)
            canvas[np.clip(ys[ok], 0, H - 1), np.clip(xs[ok], 0, W - 1)] = col
        # DEM polyline for this slice
        ii = int(np.clip((uc - s.origin_u) / s.cell_size, 0, cols - 1))
        vs = s.origin_v + np.arange(rows) * s.cell_size
        inside = (vs >= vmin) & (vs <= vmax)
        xd, yd = to_px(vs, gf.dem[:, ii])
        covc = gf.coverage[:, ii]
        for j in range(rows - 1):
            if not (inside[j] and inside[j + 1]):
                continue
            trusted = covc[j] and covc[j + 1]
            cv2.line(canvas, (xd[j], yd[j]), (xd[j + 1], yd[j + 1]),
                     (0, 0, 0) if trusted else (180, 180, 180), 2 if trusted else 1)
        cv2.putText(canvas, f"u={uc:.1f}m (+/-1m)  black=observed DEM  grey=extrap",
                    (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        panels.append(canvas)
    cv2.imwrite(str(out / "cross_sections.png"), np.vstack(panels))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="COLMAP text model dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-size", type=float, default=0.5)
    args = ap.parse_args()

    print(f"reading {args.model} ...")
    recon = read_model(args.model)
    print(f"  images={len(recon.images)} points={recon.num_points}")
    gf, local, hag, label = G.from_reconstruction(recon, cell_size=args.cell_size)

    cams = np.array([-G._qvec2rotmat(im.qvec).T @ im.tvec for im in recon.images.values()])
    cams_local = (gf.R @ cams.T).T

    out = Path(args.out)
    render(gf, local, hag, label, cams_local, out)
    np.savez_compressed(out / "ground_field.npz", dem=gf.dem, coverage=gf.coverage,
                        R=gf.R, up=gf.up,
                        origin=np.array([gf.spec.origin_u, gf.spec.origin_v]),
                        cell_size=gf.spec.cell_size)
    print(f"wrote renders + ground_field.npz to {out}")


if __name__ == "__main__":
    main()
