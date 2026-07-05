"""End-to-end driver: COLMAP model + images -> semantic ground Level.

    read model -> segment images -> vote labels onto points -> rasterise -> export

The geometry source (COLMAP points) and the segmenter are both swappable; this file just
wires them together and handles frame orientation (which world axis is "up", and its sign)
since a raw COLMAP frame is not guaranteed to be gravity-aligned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from colmap_io import Reconstruction, read_model
from ground_model import build_level, save_level
from labeling import MaskStore, accumulate_votes, resolve_labels, segment_and_cache
from segmentation import SEGMENTER_KINDS, make_segmenter
from taxonomy import Klass


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def detect_gravity_up(recon: Reconstruction) -> tuple[int, float, np.ndarray]:
    """Authoritative gravity-up from camera orientations (not a geometry heuristic).

    Phones are held roughly upright, so the world-space direction of image-up -- which is
    ``-R[1,:]`` (COLMAP cameras look +Z with +Y_cam pointing *down*) -- averaged over all
    frames points along true up. Guards against an upside-down / rotated reconstruction.
    Returns (axis, sign, mean_up_vector).
    """
    ups = np.array([-_qvec2rotmat(im.qvec)[1, :] for im in recon.images.values()])
    m = ups.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    axis = int(np.argmax(np.abs(m)))
    return axis, float(np.sign(m[axis])), m


def camera_centers(recon: Reconstruction) -> np.ndarray:
    return np.array([-_qvec2rotmat(im.qvec).T @ im.tvec for im in recon.images.values()])


def run(model_dir: str | None, image_dir: str | None, out_dir: str, *,
        source: str = "colmap", gsplat_dir: str | None = None, min_opacity: float = 0.1,
        segmenter_kind: str = "heuristic", cell_size: float = 0.5,
        up_axis: int | None = None, up_sign: float | None = None,
        limit: int | None = None, clip_threshold: float = 0.30,
        overwrite_masks: bool = False, semantic_mode: str = "vote", log=print) -> None:
    out = Path(out_dir)
    recon = None   # COLMAP model: geometry+labels in colmap mode; gravity only in gsplat mode
    store = None
    image_ids = None

    if source == "gsplat":
        log(f"[1/4] loading semantic 3DGS points: {gsplat_dir}")
        from gsplat_source import load_gsplat_points
        positions, labels, conf = load_gsplat_points(gsplat_dir, min_opacity=min_opacity, log=log)
        labeled = int((labels != int(Klass.UNKNOWN)).sum())
        log(f"      {labeled}/{len(labels)} points labelled "
            f"({100 * labeled / max(len(labels), 1):.1f}%)")
        _log_class_histogram(labels, log)
        # Gravity/up needs camera orientations, which the Gaussians don't carry -- read the
        # COLMAP model the gsplat run was trained from (same world coords).
        if (up_axis is None or up_sign is None):
            if not model_dir:
                raise SystemExit("--source gsplat needs --model (for gravity) or explicit "
                                 "--up-axis/--up-sign")
            recon = read_model(model_dir)
        if semantic_mode == "project":
            log("      note: --semantic-mode project is COLMAP-only; using per-Gaussian labels")
            semantic_mode = "vote"
    else:
        mask_dir = out / "masks"
        log(f"[1/5] reading COLMAP model: {model_dir}")
        recon = read_model(model_dir)
        log(f"      {len(recon.images)} images, {recon.num_points} points")

        image_ids = sorted(recon.images)
        if limit is not None:
            image_ids = image_ids[:limit]
            log(f"      limiting to first {len(image_ids)} images")

        log(f"[2/5] segmenting ({segmenter_kind}) -> {mask_dir}")
        seg_kw = {"threshold": clip_threshold} if segmenter_kind == "clipseg" else {}
        segmenter = make_segmenter(segmenter_kind, **seg_kw)
        store = MaskStore(mask_dir)
        segment_and_cache(recon, image_dir, segmenter, store, image_ids,
                          overwrite=overwrite_masks, log=log)

        log("[3/5] voting labels onto points")
        point_ids, votes, _ = accumulate_votes(recon, store, image_ids)
        labels, conf = resolve_labels(votes)
        positions = np.array([recon.points3d[int(pid)].xyz for pid in point_ids])
        labeled = int((labels != int(Klass.UNKNOWN)).sum())
        log(f"      {labeled}/{len(labels)} points labelled "
            f"({100 * labeled / len(labels):.1f}%)")
        _log_class_histogram(labels, log)

    if up_axis is None or up_sign is None:
        a, s, vec = detect_gravity_up(recon)
        up_axis = a if up_axis is None else up_axis
        up_sign = s if up_sign is None else up_sign
        cu = camera_centers(recon)[:, up_axis] * up_sign
        pu = positions[:, up_axis] * up_sign
        above = float(np.quantile(cu, 0.5) - np.quantile(pu, 0.1))
        log(f"      gravity up: axis={'xyz'[up_axis]} sign={up_sign:+.0f} "
            f"(mean cam-up {np.round(vec, 2).tolist()}); cameras ~{above:.1f} m above ground")
        if above < 0:
            log("      WARNING: cameras sit BELOW ground along up-axis -- frame may be upside down!")

    log("[build] building ground level")
    level = build_level(positions, labels, conf, cell_size=cell_size,
                        up_axis=up_axis, up_sign=up_sign, log=log)

    if semantic_mode == "project":
        log("      dense-mask projection (semantic raster from all pixels)")
        from dense_projection import project_dense_semantics
        sem, proj_cov = project_dense_semantics(recon, store, level, image_ids, log=log)
        level.semantic = sem
        level.coverage = level.coverage | proj_cov  # projection reaches cells sparse points miss

    log(f"[export] -> {out}")
    save_level(level, out, prefix="level0")
    log("done.")


def _log_class_histogram(labels: np.ndarray, log) -> None:
    counts = np.bincount(labels, minlength=int(max(Klass)) + 1)
    parts = [f"{Klass(k).name}={c}" for k, c in enumerate(counts) if c > 0]
    log("      classes: " + ", ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic BEV ground model from COLMAP points "
                                             "or a semantic 3DGS export")
    ap.add_argument("--session", default=None,
                    help="LAR session name: fills --model/--images/--gsplat-dir/--out from the "
                         "canonical layout (script/lar_session.py). Explicit flags override.")
    ap.add_argument("--source", default="colmap", choices=["colmap", "gsplat"],
                    help="geometry source: 'colmap' sparse points + segmentation (default), or "
                         "'gsplat' a trained semantic 3DGS export (denser, pre-labelled)")
    ap.add_argument("--gsplat-dir", default=None,
                    help="semantic 3DGS export dir (script/gsplat --out); required for --source gsplat")
    ap.add_argument("--min-opacity", type=float, default=0.1,
                    help="drop Gaussians below this opacity when using --source gsplat")
    ap.add_argument("--model", default=None,
                    help="COLMAP text model dir. Required for --source colmap; for --source gsplat "
                         "it supplies camera orientations for gravity (unless --up-axis/--up-sign given)")
    ap.add_argument("--images", default=None, help="directory of source images (--source colmap)")
    ap.add_argument("--out", default=None, help="output directory (derived from --session if unset)")
    ap.add_argument("--segmenter", default="heuristic", choices=list(SEGMENTER_KINDS))
    ap.add_argument("--cell-size", type=float, default=0.5, help="metres per grid cell")
    ap.add_argument("--up-axis", type=int, default=None, choices=[0, 1, 2], help="0=x,1=y,2=z (auto if unset)")
    ap.add_argument("--up-sign", type=float, default=None, choices=[1.0, -1.0], help="auto if unset")
    ap.add_argument("--limit", type=int, default=None, help="only use first N images")
    ap.add_argument("--clip-threshold", type=float, default=0.30)
    ap.add_argument("--overwrite-masks", action="store_true")
    ap.add_argument("--semantic-mode", default="vote", choices=["vote", "project"],
                    help="vote: sparse point votes (fast); project: dense-mask projection (cleaner)")
    args = ap.parse_args()
    if args.session:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from lar_session import Session
        s = Session(args.session)
        args.model = args.model or str(s.best_model())  # geometry (colmap) or gravity (gsplat)
        args.out = args.out or str(s.sbev_out(args.source))
        if args.source == "gsplat":
            args.gsplat_dir = args.gsplat_dir or str(s.gsplat_out(semantic=True))
        else:
            args.images = args.images or str(s.images)
        print(f"session '{args.session}': source={args.source} out={args.out}")

    if not args.out:
        ap.error("need --session or --out")
    if args.source == "colmap" and (not args.model or not args.images):
        ap.error("--source colmap requires --model and --images (or --session)")
    if args.source == "gsplat" and not args.gsplat_dir:
        ap.error("--source gsplat requires --gsplat-dir (or --session)")

    run(args.model, args.images, args.out, source=args.source, gsplat_dir=args.gsplat_dir,
        min_opacity=args.min_opacity, segmenter_kind=args.segmenter,
        cell_size=args.cell_size, up_axis=args.up_axis, up_sign=args.up_sign,
        limit=args.limit, clip_threshold=args.clip_threshold,
        overwrite_masks=args.overwrite_masks, semantic_mode=args.semantic_mode)


if __name__ == "__main__":
    main()
