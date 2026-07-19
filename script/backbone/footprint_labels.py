"""Footprint / free-space supervision generator for a per-frame ground head.

Produces the training targets a *footprint / free-space* head (à la Niantic's
"Footprints and Free Space from a Single Color Image", CVPR 2020) needs on top of the
frozen LingBot backbone — the one signal plain semantic segmentation can't give you:
**where the ground continues *behind* objects**, and **which ground cells an object's
footprint occupies**. That's exactly what tells the BEV projector which pixels map onto
the ground plane (incl. hidden ground) and which map onto occupied area.

Why this is cheaper/cleaner than Niantic's method:
Niantic had no global model, so they *manufactured* hidden-ground labels by projecting the
ground observed in neighbouring frames into the target view. We already reconstruct a
gravity-aligned **global ground DEM** (`geometry.from_reconstruction`) that aggregates every
frame's ground. Rendering that DEM into a target camera yields the full ground extent —
visible *and* occluded — in one shot, with far less noise than pairwise reprojection. The
neighbour-frame trick is subsumed by "render the global DEM".

Per target frame we emit, at a downscaled render resolution:
  - `<stem>.png`        uint8 class map, one of:
        0 NON_GROUND      (sky / above / unobserved)
        1 VISIBLE_GROUND  (ground directly visible — project it)
        2 HIDDEN_GROUND   (traversable ground occluded by an object — project it too)
        3 FOOTPRINT       (ground cell occupied by a vertical object — its contact/occupied area)
        4 OBJECT          (the occluder body standing above the ground)
  - `<stem>_depth.npy`  float32 metric depth from the camera to the DEM ground surface,
                        valid wherever the ground extent is (visible|hidden|footprint) —
                        the Niantic "depth to (hidden) ground" regression target.
  - `<stem>_cover.npy`  float32 in {0,1}: was the ground cell DEM-*observed* (trust) vs
                        extrapolated — use as a per-pixel loss weight.
  - `<stem>_preview.png` RGB blended with the colourised class map (eyeball check).

Geometry-only for now (footprint is class-agnostic): the *what-class* layer (tree / building
/ wall / furniture per footprint cell, "occupied areas WITH semantic labels") drops in next by
voting each VERTICAL point's class from the cached Mask2Former masks — see FOOTPRINT-SEMANTICS
hook below.

Run (from repo root):
  uv run python script/backbone/footprint_labels.py --session maguro-park-after-itchy --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# --- make sibling script packages importable (geometry, colmap_io, lar_session) ---
_HERE = Path(__file__).resolve().parent
_SCRIPT_ROOT = _HERE.parent
sys.path.insert(0, str(_SCRIPT_ROOT))
sys.path.insert(0, str(_SCRIPT_ROOT / "semantic_bev"))

import geometry  # noqa: E402
from colmap_io import qvec2rotmat, read_model  # noqa: E402
from lar_session import Session  # noqa: E402

# class ids (keep in sync with the module docstring / meta.json legend)
NON_GROUND, VISIBLE_GROUND, HIDDEN_GROUND, FOOTPRINT, OBJECT = 0, 1, 2, 3, 4
CLASS_NAMES = ["non_ground", "visible_ground", "hidden_ground", "footprint", "object"]
# BGR palette for previews
PALETTE = np.array([
    [0, 0, 0],        # non_ground   black
    [80, 200, 80],    # visible      green
    [40, 120, 220],   # hidden       orange
    [40, 40, 220],    # footprint    red
    [200, 200, 60],   # object       cyan-ish
], dtype=np.uint8)


# --------------------------------------------------------------------------- #
# intrinsics
# --------------------------------------------------------------------------- #
def pinhole_params(cam) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) from a PINHOLE / SIMPLE_PINHOLE / SIMPLE_RADIAL camera.

    Distortion (SIMPLE_RADIAL's k) is ignored — the DEM render only needs the pinhole
    part, and these ARKit-derived cameras are near-pinhole after undistortion.
    """
    p = cam.params
    if cam.model == "PINHOLE":
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    raise ValueError(f"camera model {cam.model!r} not supported")


# --------------------------------------------------------------------------- #
# ground mesh (built once from the DEM, reused for every frame)
# --------------------------------------------------------------------------- #
class GroundMesh:
    """The DEM as a triangle mesh of world-space vertices + per-face attributes.

    Vertices sit at cell centres of the gravity-aligned DEM; two triangles per cell quad.
    ``footprint`` / ``coverage`` are per-face booleans so a rasterised pixel resolves
    straight to "is this ground under an object?" / "was this ground observed?".
    """

    def __init__(self, gf: geometry.GroundField, footprint_cell: np.ndarray):
        self.gf = gf  # kept so downstream (base_points.py) can reuse the DEM/grid + gravity R
        spec, dem = gf.spec, gf.dem
        rows, cols, cs = spec.rows, spec.cols, spec.cell_size
        # vertex grid at cell centres, local (gravity-aligned) frame
        cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
        u = spec.origin_u + (cc + 0.5) * cs
        v = spec.origin_v + (rr + 0.5) * cs
        vert_local = np.stack([u.ravel(), v.ravel(), dem.ravel()], axis=1)  # (V,3)
        # local -> world: world_row = local_row @ R  (R maps world->local, orthonormal)
        self.verts_world = vert_local @ gf.R                                 # (V,3)

        cov = gf.coverage.ravel()
        foot = footprint_cell.reshape(rows, cols)
        # two triangles per quad (r,c)-(r+1,c+1); vertex index = r*cols + c
        r0, c0 = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
        r0, c0 = r0.ravel(), c0.ravel()
        i00 = r0 * cols + c0
        i01 = r0 * cols + (c0 + 1)
        i10 = (r0 + 1) * cols + c0
        i11 = (r0 + 1) * cols + (c0 + 1)
        faces = np.concatenate([
            np.stack([i00, i01, i11], axis=1),
            np.stack([i00, i11, i10], axis=1),
        ], axis=0)
        self.faces = faces                                                  # (F,3)
        # per-face attrs, indexed by the quad's min corner cell (r0,c0)
        quad_cell = r0 * cols + c0
        f_foot = foot.ravel()[quad_cell]
        f_cov = cov[quad_cell]
        self.face_footprint = np.tile(f_foot, 2)                            # (F,)
        self.face_coverage = np.tile(f_cov, 2)                              # (F,)


def backproject_positions(recon, depth_dir: Path, image_ids, stride: int, voxel: float,
                          depth_max: float = 0.0, log=print) -> np.ndarray:
    """Fused world point cloud from per-view metric depth (positions only, voxel-deduped).

    Positions-only twin of ``depth_backproject.backproject_labeled_points`` — no MaskStore,
    since the DEM only needs geometry. Same depth-map contract: ``<stem>.npy`` float32 (H,W)
    metric z-depth, intrinsics scaled to depth resolution.
    """
    depth_dir = Path(depth_dir)
    all_pos, n, raw = [], 0, 0
    for img_id in image_ids:
        im = recon.images[img_id]
        dp = depth_dir / f"{Path(im.name).stem}.npy"
        if not dp.exists():
            continue
        depth = np.load(dp).astype(np.float32)
        h, w = depth.shape
        cam = recon.cameras[im.camera_id]
        fx, fy, cx, cy = pinhole_params(cam)
        sx, sy = w / cam.width, h / cam.height           # scale intrinsics to depth res
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        ys, xs = np.arange(0, h, stride), np.arange(0, w, stride)
        gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
        d = depth[gy, gx]
        ok = (d > 0) & np.isfinite(d)
        if depth_max > 0:
            ok &= d < depth_max          # far/sky mono depth explodes (1/(a·disp+b)); cull it
        if not ok.any():
            continue
        u, v, dd = gx[ok].astype(np.float32), gy[ok].astype(np.float32), d[ok]
        cam_pts = np.stack([(u - cx) / fx * dd, (v - cy) / fy * dd, dd], axis=1)
        R = qvec2rotmat(im.qvec)
        world = cam_pts @ R + (-R.T @ im.tvec)           # world = R^T @ cam + C
        all_pos.append(world.astype(np.float32))
        n += 1
        raw += len(world)
    if not all_pos:
        raise SystemExit(f"no depth maps found in {depth_dir}")
    pos = np.concatenate(all_pos)
    vox = np.floor(pos / voxel).astype(np.int64)
    uniq, inv = np.unique(vox, axis=0, return_inverse=True)
    sums = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sums, inv, pos)
    positions = (sums / np.bincount(inv)[:, None]).astype(np.float32)
    log(f"  mono depth: {raw} pts from {n} maps -> {len(positions)} voxels @ {voxel} m")
    return positions


def build_ground_mesh(recon, cell_size: float, clearance: tuple[float, float],
                      min_count: int, solid_gap: float, *, dem_world: np.ndarray | None = None,
                      log=print) -> tuple[GroundMesh, np.ndarray]:
    """DEM mesh (from ``dem_world`` or COLMAP points) + COLMAP footprint/occluders.

    The ground **surface** (DEM) can come from a denser source via ``dem_world`` (e.g. fused
    mono depth) while the **footprint/occupancy stays on the sparse-but-accurate COLMAP
    obstacle cloud** — the depth-bench hybrid verdict (mono's ~20% vertical noise scatters
    canopy into the band and over-blocks; SfM is geometrically exact). Both share one gravity
    frame derived from the cameras, so the mono DEM and the COLMAP footprint cells line up.

    A cell is a **footprint** (occupied) only if it passes the *solid-to-ground* test the
    validated occupancy layer uses (`ground_model.build_level`): ≥ ``min_count`` obstacle
    points whose low (0.1) height-above-ground quantile ≤ ``solid_gap`` — the column reaches
    the ground (trunk/wall/bush/building). Canopy over a path floats above the gap → the path
    stays walkable ground. Occluders for the visible/hidden split are the body-height **band**
    points (canopy over a path still occludes the ground below).
    """
    qvecs = np.array([im.qvec for im in recon.images.values()])
    up = geometry.gravity_up(qvecs)
    R = geometry.align_rotation(up)
    log(f"gravity up = [{up[0]:+.4f} {up[1]:+.4f} {up[2]:+.4f}] (axis {int(np.argmax(np.abs(up)))})")

    colmap_xyz = np.array([p.xyz for p in recon.points3d.values()])
    dem_local = (colmap_xyz if dem_world is None else dem_world) @ R.T
    gf = geometry.build_dem(dem_local[:, :2], dem_local[:, 2], R, up, cell_size=cell_size, log=log)

    # footprint + occluders: COLMAP obstacle points classified against this DEM
    colmap_local = colmap_xyz @ R.T
    hag, label = geometry.classify_points(colmap_local[:, :2], colmap_local[:, 2], gf)
    ncells = gf.spec.rows * gf.spec.cols
    obstacle = label == geometry.VERTICAL
    base_hag, _ = geometry._cell_quantile(gf.cell_of(colmap_local[obstacle, :2]),
                                          hag[obstacle], ncells, 0.1, min_count)
    footprint_cell = np.isfinite(base_hag) & (base_hag <= solid_gap)
    lo, hi = clearance
    band = (hag > lo) & (hag < hi)
    occ_world = colmap_xyz[band]  # already world coords
    log(f"  obstacles {int(obstacle.sum())}, band {int(band.sum())}, "
        f"footprint cells {int(footprint_cell.sum())} / {ncells}")
    return GroundMesh(gf, footprint_cell), occ_world


# --------------------------------------------------------------------------- #
# projection + rasterisation
# --------------------------------------------------------------------------- #
def project(world: np.ndarray, R_wc: np.ndarray, tvec: np.ndarray,
            fx, fy, cx, cy) -> tuple[np.ndarray, np.ndarray]:
    """world (N,3) -> pixel (N,2) + camera-space depth z (N,). Mirrors dense_projection."""
    cam = world @ R_wc.T + tvec
    z = cam[:, 2]
    px = np.full(len(z), -1.0)
    py = np.full(len(z), -1.0)
    front = z > 1e-6
    px[front] = fx * cam[front, 0] / z[front] + cx
    py[front] = fy * cam[front, 1] / z[front] + cy
    return np.stack([px, py], axis=1), z


def rasterize(verts_px: np.ndarray, verts_z: np.ndarray, faces: np.ndarray,
              h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer a triangle mesh -> (depth[h,w] float32, face_id[h,w] int32, -1 empty).

    Plain per-face barycentric fill over the face's pixel bounding box. Face count is
    already culled to the frustum+range slice, so the python loop stays small.
    """
    zbuf = np.full((h, w), np.inf, np.float32)
    fbuf = np.full((h, w), -1, np.int32)
    for fi in range(len(faces)):
        a, b, c = faces[fi]
        pa, pb, pc = verts_px[a], verts_px[b], verts_px[c]
        za, zb, zc = verts_z[a], verts_z[b], verts_z[c]
        minx = max(int(np.floor(min(pa[0], pb[0], pc[0]))), 0)
        maxx = min(int(np.ceil(max(pa[0], pb[0], pc[0]))), w - 1)
        miny = max(int(np.floor(min(pa[1], pb[1], pc[1]))), 0)
        maxy = min(int(np.ceil(max(pa[1], pb[1], pc[1]))), h - 1)
        if maxx < minx or maxy < miny:
            continue
        denom = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(denom) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        w0 = ((pb[1] - pc[1]) * (gx - pc[0]) + (pc[0] - pb[0]) * (gy - pc[1])) / denom
        w1 = ((pc[1] - pa[1]) * (gx - pc[0]) + (pa[0] - pc[0]) * (gy - pc[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * za + w1 * zb + w2 * zc
        sub_z = zbuf[miny:maxy + 1, minx:maxx + 1]
        sub_f = fbuf[miny:maxy + 1, minx:maxx + 1]
        m = inside & (z < sub_z)
        sub_z[m] = z[m]
        sub_f[m] = fi
    return zbuf, fbuf


def splat_occluders(px: np.ndarray, py: np.ndarray, z: np.ndarray,
                    h: int, w: int, radius: int) -> np.ndarray:
    """Nearest-depth buffer of the (sparse) vertical points, dilated by ``radius`` px.

    A min-filter (cv2.erode on depth) grows each point into a small disk so the sparse
    SfM occluders form connected blobs to test ground pixels against.
    """
    BIG = np.float32(1e9)
    buf = np.full(h * w, BIG, np.float32)
    ix = np.round(px).astype(np.int64)
    iy = np.round(py).astype(np.int64)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (z > 1e-6)
    np.minimum.at(buf, iy[ok] * w + ix[ok], z[ok].astype(np.float32))
    buf = buf.reshape(h, w)
    if radius > 0:
        k = 2 * radius + 1
        buf = cv2.erode(buf, np.ones((k, k), np.uint8))  # grayscale erode = local min
    return buf


# --------------------------------------------------------------------------- #
# per-frame label assembly
# --------------------------------------------------------------------------- #
def make_frame_labels(mesh: GroundMesh, occ_world: np.ndarray, im, cam,
                      render_size: int, max_range: float, occ_margin: float,
                      splat_radius: int) -> dict | None:
    """Render the DEM (+ occluders) into one frame -> class map, ground depth, coverage."""
    scale = render_size / max(cam.width, cam.height)
    w = int(round(cam.width * scale))
    h = int(round(cam.height * scale))
    fx, fy, cx, cy = (v * scale for v in pinhole_params(cam))
    R_wc = qvec2rotmat(im.qvec)
    tvec = im.tvec

    verts_px, verts_z = project(mesh.verts_world, R_wc, tvec, fx, fy, cx, cy)

    # cull faces: all 3 verts in front, and the face's nearest vertex within range
    fz = verts_z[mesh.faces]                      # (F,3)
    keep = (fz > 1e-6).all(axis=1) & (fz.min(axis=1) < max_range)
    if not keep.any():
        return None
    kept = np.flatnonzero(keep)
    ground_depth, fbuf = rasterize(verts_px, verts_z, mesh.faces[kept], h, w)

    # occluder depth from the band points, culled to render range (far points can't
    # meaningfully occlude the ground we actually render)
    opx, oz = project(occ_world, R_wc, tvec, fx, fy, cx, cy)
    near = oz < max_range
    obj_depth = splat_occluders(opx[near, 0], opx[near, 1], oz[near], h, w, splat_radius)

    ground = np.isfinite(ground_depth)
    face_at = fbuf.copy()
    face_at[~ground] = -1
    is_foot = np.zeros((h, w), bool)
    cover = np.zeros((h, w), np.float32)
    valid = face_at >= 0
    if valid.any():
        # map rasterised local-face-index -> global face index -> attrs
        gfi = kept[face_at[valid]]
        is_foot[valid] = mesh.face_footprint[gfi]
        cover[valid] = mesh.face_coverage[gfi].astype(np.float32)

    has_occ = obj_depth < 1e8
    occluded = ground & has_occ & (obj_depth < ground_depth - occ_margin)

    # A pixel that visually shows an occluder but has DEM ground behind it is labelled by that
    # ground (HIDDEN, or FOOTPRINT if the cell is occupied) — the Niantic "predict the ground
    # behind the object" target. OBJECT is reserved for occluders standing over sky (above the
    # horizon), i.e. structure that is genuinely not ground.
    label = np.full((h, w), NON_GROUND, np.uint8)
    label[ground] = VISIBLE_GROUND
    label[occluded] = HIDDEN_GROUND
    label[ground & is_foot] = FOOTPRINT           # ground-under-object = the contact/occupied area
    label[has_occ & ~ground] = OBJECT             # occluder over sky = above-horizon structure

    depth_out = np.where(ground, ground_depth, np.nan).astype(np.float32)
    return {"label": label, "depth": depth_out, "cover": cover, "size": (w, h)}


def colourise(label: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
    vis = PALETTE[label]
    if rgb is not None:
        rgb = cv2.resize(rgb, (label.shape[1], label.shape[0]))
        vis = cv2.addWeighted(rgb, 0.5, vis, 0.5, 0)
    return vis


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session")
    ap.add_argument("--model", help="COLMAP text model dir (default: session.best_model())")
    ap.add_argument("--images", help="image dir (default: session.images)")
    ap.add_argument("--out", help="output dir (default: output/<session>-footprint)")
    ap.add_argument("--dem-source", choices=["colmap", "mono"], default="colmap",
                    help="ground DEM from COLMAP points (default) or fused mono depth (denser)")
    ap.add_argument("--depth-dir", help="per-view depth .npy dir (default: session.depth_dir('mono'))")
    ap.add_argument("--depth-stride", type=int, default=8, help="pixel stride when back-projecting depth")
    ap.add_argument("--depth-voxel", type=float, default=0.1, help="voxel size (m) for depth dedup")
    ap.add_argument("--depth-max", type=float, default=40.0,
                    help="drop mono-depth pixels beyond this range (m); far/sky depth explodes")
    ap.add_argument("--cell-size", type=float, default=0.5, help="DEM cell size (m)")
    ap.add_argument("--clearance-lo", type=float, default=0.4,
                    help="body-height band lower bound above ground (m)")
    ap.add_argument("--clearance-hi", type=float, default=2.0,
                    help="body-height band upper bound above ground (m); canopy above is ignored")
    ap.add_argument("--min-count", type=int, default=3,
                    help="min obstacle points in a cell to call it a footprint")
    ap.add_argument("--solid-gap", type=float, default=0.8,
                    help="max low-quantile height-above-ground for a footprint (solid-to-ground test)")
    ap.add_argument("--size", type=int, default=512, help="render long-side resolution")
    ap.add_argument("--max-range", type=float, default=30.0, help="cull ground beyond this (m)")
    ap.add_argument("--occ-margin", type=float, default=0.5,
                    help="depth margin (m) for an occluder to count as in-front of ground")
    ap.add_argument("--splat-radius", type=int, default=2, help="occluder splat radius (px)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N frames (0=all)")
    ap.add_argument("--sample", type=int, default=0,
                    help="process N frames spaced evenly across the capture (overrides --limit)")
    ap.add_argument("--no-preview", action="store_true", help="skip RGB preview blends")
    args = ap.parse_args()

    s = Session(args.session) if args.session else None
    if s:
        model = Path(args.model) if args.model else s.best_model()
        images = Path(args.images) if args.images else s.images
        out = Path(args.out) if args.out else s.root / "output" / f"{s.name}-footprint"
    else:
        if not (args.model and args.images and args.out):
            ap.error("without --session, pass --model, --images and --out")
        model, images, out = Path(args.model), Path(args.images), Path(args.out)

    depth_dir = None
    if args.dem_source == "mono":
        depth_dir = Path(args.depth_dir) if args.depth_dir else (s.depth_dir("mono") if s else None)
        if depth_dir is None:
            ap.error("--dem-source mono needs --depth-dir (or --session)")
    out.mkdir(parents=True, exist_ok=True)

    print(f"model  : {model}")
    print(f"images : {images}")
    print(f"out    : {out}")
    recon = read_model(str(model))
    print(f"loaded {len(recon.images)} images, {recon.num_points} points")

    dem_world = None
    if args.dem_source == "mono":
        print(f"DEM source: mono depth <- {depth_dir}")
        dem_world = backproject_positions(recon, depth_dir, list(recon.images),
                                          args.depth_stride, args.depth_voxel, args.depth_max)

    mesh, occ_world = build_ground_mesh(
        recon, args.cell_size, (args.clearance_lo, args.clearance_hi),
        args.min_count, args.solid_gap, dem_world=dem_world)
    print(f"mesh: {len(mesh.verts_world)} verts, {len(mesh.faces)} faces, "
          f"{len(occ_world)} occluder points")

    ims = sorted(recon.images.values(), key=lambda im: im.name)
    if args.sample and args.sample < len(ims):
        idx = np.linspace(0, len(ims) - 1, args.sample).round().astype(int)
        ims = [ims[i] for i in idx]
    elif args.limit:
        ims = ims[: args.limit]

    counts = np.zeros(len(CLASS_NAMES), np.int64)
    done = skipped = 0
    for im in ims:
        cam = recon.cameras[im.camera_id]
        res = make_frame_labels(mesh, occ_world, im, cam, args.size,
                                args.max_range, args.occ_margin, args.splat_radius)
        if res is None:
            skipped += 1
            continue
        stem = Path(im.name).stem
        cv2.imwrite(str(out / f"{stem}.png"), res["label"])
        np.save(out / f"{stem}_depth.npy", res["depth"])
        np.save(out / f"{stem}_cover.npy", res["cover"])
        if not args.no_preview:
            rgb = cv2.imread(str(images / im.name))
            cv2.imwrite(str(out / f"{stem}_preview.png"), colourise(res["label"], rgb))
        counts += np.bincount(res["label"].ravel(), minlength=len(CLASS_NAMES))
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(ims)}")

    total = counts.sum()
    print(f"\nwrote {done} frames ({skipped} skipped, no ground in view) -> {out}")
    if total:
        for name, c in zip(CLASS_NAMES, counts):
            print(f"  {name:14s} {c / total * 100:5.1f}%")

    meta = {
        "classes": {i: n for i, n in enumerate(CLASS_NAMES)},
        "dem_source": args.dem_source, "cell_size": args.cell_size, "render_size": args.size,
        "clearance_band": [args.clearance_lo, args.clearance_hi], "min_count": args.min_count,
        "solid_gap": args.solid_gap,
        "max_range": args.max_range, "occ_margin": args.occ_margin,
        "splat_radius": args.splat_radius, "frames": done,
        "note": "FOOTPRINT-SEMANTICS hook: vote each VERTICAL point's Mask2Former class "
                "(MaskStore) to give footprint cells a class id.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
