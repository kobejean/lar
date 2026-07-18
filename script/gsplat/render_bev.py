#!/usr/bin/env python3
"""Top-down orthographic BEV render from a trained 3DGS point_cloud.ply.

The `--rgb-ortho` drape in `script/semantic_bev` paints the source RGB onto a single-surface
ground DEM. That smears anything above the ground (benches, walls, trees) and ghosts under the
grazing capture views. Gaussian splatting sidesteps both: rasterise the trained model from a
**virtual orthographic top-down camera** and occlusion/height fall out of the render itself —
a true photographic BEV, not a ground drape.

gsplat's `rasterization(camera_model="ortho")` gives real orthographic projection
(`px = fx*Xc + cx`, no Z-divide), so `fx = fy = 1/cell_size` px-per-metre and the whole
footprint maps in with the camera centred over it, looking straight down gravity.

Alignment: pass `--meta <level0.meta.json>` and the BEV lands on the **same grid** as the
semantic/height/occupancy rasters (same origin, cell size, gravity-canonical row axis), so the
layers overlay pixel-for-pixel. Without `--meta`, the footprint + gravity are inferred from the
COLMAP cameras (`--model`) and the Gaussian extent.

  uv run --extra gsplat python render_bev.py \
      --ply   ../../output/<run>/point_cloud.ply \
      --meta  ../../output/<run>-sbev/level0.meta.json \
      --out   ../../output/<run>-sbev --cell-size 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gsplat import rasterization
from colmap_dataset import read_cameras, read_images, camera_center


# ----- load the exported Gaussians (INRIA PLY layout) --------------------------
def load_ply(path: Path, device: str) -> dict:
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fields, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                fields.append(line.split()[-1])
        data = np.fromfile(f, dtype="<f4", count=n * len(fields)).reshape(n, len(fields))

    col = {name: i for i, name in enumerate(fields)}
    means = data[:, [col["x"], col["y"], col["z"]]].copy()
    f_dc = data[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]
    rest_names = sorted((k for k in col if k.startswith("f_rest_")),
                        key=lambda s: int(s.split("_")[-1]))
    f_rest = data[:, [col[k] for k in rest_names]]
    k_minus1 = f_rest.shape[1] // 3
    shN = f_rest.reshape(n, 3, k_minus1).transpose(0, 2, 1)
    sh = np.concatenate([f_dc[:, None, :], shN], axis=1)
    sh_degree = int(round(sh.shape[1] ** 0.5)) - 1

    opacities = torch.sigmoid(torch.from_numpy(data[:, [col["opacity"]]].copy()).squeeze(-1))
    scales = torch.exp(torch.from_numpy(
        data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]].copy()))
    quats = torch.from_numpy(data[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]].copy())
    return dict(
        means_np=means, means=torch.from_numpy(means).to(device),
        colors=torch.from_numpy(sh.copy()).to(device),
        opacities=opacities.to(device), scales=scales.to(device), quats=quats.to(device),
        sh_degree=sh_degree, n=n,
    )


# ----- gravity from the COLMAP cameras (same rule as semantic_bev.pipeline) -----
def detect_gravity(model_dir: Path) -> tuple[int, float]:
    cams = read_cameras(model_dir / "cameras.txt")
    views = read_images(model_dir / "images.txt", cams)
    # image-up in world = -R[1,:] (COLMAP: +Y_cam points down); phones held upright -> true up.
    ups = np.array([-v.viewmat[:3, :3][1, :] for v in views])
    m = ups.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    axis = int(np.argmax(np.abs(m)))
    return axis, float(np.sign(m[axis]))


def main() -> None:
    p = argparse.ArgumentParser(description="Orthographic top-down BEV render of a 3DGS model")
    p.add_argument("--ply", required=True)
    p.add_argument("--meta", default=None,
                   help="level0.meta.json to align the BEV to the semantic/DEM raster grid")
    p.add_argument("--model", default=None,
                   help="COLMAP model dir for gravity (needed when --meta is absent)")
    p.add_argument("--out", default=None, help="output dir (default: alongside the ply)")
    p.add_argument("--prefix", default="level0")
    p.add_argument("--cell-size", type=float, default=0.05, help="metres per output pixel")
    p.add_argument("--pad", type=float, default=2.0, help="footprint padding (m) when auto (no --meta)")
    p.add_argument("--bounds-pct", type=float, default=0.01,
                   help="robust Gaussian-extent quantile for auto footprint")
    p.add_argument("--up-axis", type=int, default=None, choices=[0, 1, 2])
    p.add_argument("--up-sign", type=float, default=None, choices=[1.0, -1.0])
    p.add_argument("--bg", type=float, default=0.0, help="background grey level [0,1]")
    p.add_argument("--min-opacity", type=float, default=0.15,
                   help="drop Gaussians below this opacity before rendering. A top-down view is "
                        "out-of-distribution for a ground-captured model, so low-opacity floaters "
                        "show as needle artifacts; culling them cleans the BEV. 0 = keep all")
    p.add_argument("--max-pixels", type=int, default=80_000_000,
                   help="guard: refuse a grid larger than this (W*H)")
    args = p.parse_args()

    device = "cuda"
    ply = Path(args.ply)
    out = Path(args.out) if args.out else ply.parent
    out.mkdir(parents=True, exist_ok=True)
    g = load_ply(ply, device)
    print(f"loaded {g['n']} Gaussians (sh_degree={g['sh_degree']}) from {ply}")

    if args.min_opacity > 0:
        keep = g["opacities"] >= args.min_opacity
        nk = int(keep.sum())
        for k in ("means", "colors", "opacities", "scales", "quats"):
            g[k] = g[k][keep]
        g["means_np"] = g["means_np"][keep.cpu().numpy()]
        print(f"  kept {nk}/{g['n']} Gaussians with opacity >= {args.min_opacity}")
        g["n"] = nk

    cs = args.cell_size
    meta = json.loads(Path(args.meta).read_text()) if args.meta else None

    # --- gravity + canonical row sign ---
    if args.up_axis is not None and args.up_sign is not None:
        up_axis, up_sign = args.up_axis, args.up_sign
    elif meta is not None:
        up_axis, up_sign = int(meta["up_axis"]), float(meta["up_sign"])
    elif args.model:
        up_axis, up_sign = detect_gravity(Path(args.model))
    else:
        raise SystemExit("need --meta, --model, or explicit --up-axis/--up-sign for gravity")
    u_axis, v_axis = (a for a in (0, 1, 2) if a != up_axis)
    v_sign = float(meta["v_sign"]) if meta else -up_sign  # matches ground_model.build_level

    means = g["means_np"]

    # --- footprint (u,v world extent) + output resolution ---
    if meta is not None:
        cols, rows = int(meta["cols"]), int(meta["rows"])
        origin_u, origin_v, dem_cs = meta["origin_u"], meta["origin_v"], meta["cell_size"]
        # Align to the DEM grid exactly, but render at the finer --cell-size.
        u_lo, u_hi = origin_u, origin_u + cols * dem_cs
        # canonical v span -> world[v_axis] span (v = v_sign * world[v_axis]).
        cv_lo, cv_hi = origin_v, origin_v + rows * dem_cs
        v_lo_w, v_hi_w = sorted((cv_lo * v_sign, cv_hi * v_sign))
    else:
        u = means[:, u_axis]
        w = means[:, v_axis]
        u_lo, u_hi = np.quantile(u, [args.bounds_pct, 1 - args.bounds_pct])
        v_lo_w, v_hi_w = np.quantile(w, [args.bounds_pct, 1 - args.bounds_pct])
        u_lo, u_hi = u_lo - args.pad, u_hi + args.pad
        v_lo_w, v_hi_w = v_lo_w - args.pad, v_hi_w + args.pad

    W = int(np.ceil((u_hi - u_lo) / cs))
    H = int(np.ceil((v_hi_w - v_lo_w) / cs))
    if W * H > args.max_pixels:
        raise SystemExit(f"grid {W}x{H} = {W * H} px exceeds --max-pixels {args.max_pixels}; "
                         f"raise --cell-size")
    print(f"BEV grid {W}x{H} @ {cs} m  (footprint {u_hi - u_lo:.1f} x {v_hi_w - v_lo_w:.1f} m, "
          f"up=axis{up_axis}{up_sign:+.0f}, v_sign={v_sign:+.0f})")

    # --- virtual ortho camera: right=+u_axis, look straight down gravity ---
    e = np.eye(3)
    right = e[u_axis].copy()
    forward = -up_sign * e[up_axis]            # look down: Z_cam along -up
    down = np.cross(forward, right)            # right-handed OpenCV cam frame (X right, Y down, Z fwd)
    R_wc = np.stack([right, down, forward], axis=0)  # world->camera rotation (rows are cam axes)

    # Camera centre: over the footprint centre, above the highest Gaussian.
    up_hi = up_sign * float(means[:, up_axis].max())      # highest height in the scene
    C = np.zeros(3)
    C[u_axis] = 0.5 * (u_lo + u_hi)
    C[v_axis] = 0.5 * (v_lo_w + v_hi_w)
    C[up_axis] = up_sign * (up_hi + 5.0)                  # 5 m above the top
    tvec = -R_wc @ C
    viewmat = np.eye(4)
    viewmat[:3, :3] = R_wc
    viewmat[:3, 3] = tvec

    # Intrinsics: 1/cs px per metre; principal point centres the footprint (camera is at centre).
    K = np.array([[1.0 / cs, 0, W / 2.0], [0, 1.0 / cs, H / 2.0], [0, 0, 1.0]])

    vt = torch.from_numpy(viewmat).float().to(device)[None]
    Kt = torch.from_numpy(K).float().to(device)[None]
    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"],
            opacities=g["opacities"], colors=g["colors"],
            viewmats=vt, Ks=Kt, width=W, height=H,
            sh_degree=g["sh_degree"], packed=True, render_mode="RGB",
            camera_model="ortho",
        )
    # renders are alpha-composited over black; place them over the --bg grey via the alpha.
    rgb = renders[0].clamp(0, 1).cpu().numpy()
    alpha = alphas[0, ..., 0].cpu().numpy()
    rgb = (rgb + (1.0 - alpha)[..., None] * args.bg).clip(0, 1)

    # Orient to the gravity-canonical raster: render row axis is `down`; raster row axis is
    # v_sign*e[v_axis]. Flip vertically iff they oppose so the BEV overlays the other layers.
    raster_row = v_sign * e[v_axis]
    if np.dot(down, raster_row) < 0:
        rgb = rgb[::-1].copy()
        alpha = alpha[::-1].copy()

    img = (rgb * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / f"{args.prefix}_rgb_bev.png"), bgr)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = (alpha.clip(0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / f"{args.prefix}_rgb_bev_masked.png"), bgra)
    print(f"done -> {out}/{args.prefix}_rgb_bev.png  ({W}x{H}, "
          f"mean alpha {alpha.mean():.2f})")


if __name__ == "__main__":
    main()
