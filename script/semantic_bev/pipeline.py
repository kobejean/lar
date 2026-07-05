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


def run(model_dir: str, image_dir: str, out_dir: str, *,
        segmenter_kind: str = "heuristic", cell_size: float = 0.5,
        up_axis: int | None = None, up_sign: float | None = None,
        limit: int | None = None, clip_threshold: float = 0.30,
        overwrite_masks: bool = False, semantic_mode: str = "vote", log=print) -> None:
    out = Path(out_dir)
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
    log(f"      {labeled}/{len(labels)} points labelled ({100 * labeled / len(labels):.1f}%)")
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

    log("[4/5] building ground level")
    level = build_level(positions, labels, conf, cell_size=cell_size,
                        up_axis=up_axis, up_sign=up_sign, log=log)

    if semantic_mode == "project":
        log("      dense-mask projection (semantic raster from all pixels)")
        from dense_projection import project_dense_semantics
        sem, proj_cov = project_dense_semantics(recon, store, level, image_ids, log=log)
        level.semantic = sem
        level.coverage = level.coverage | proj_cov  # projection reaches cells sparse points miss

    log(f"[5/5] exporting -> {out}")
    save_level(level, out, prefix="level0")
    log("done.")


def _log_class_histogram(labels: np.ndarray, log) -> None:
    counts = np.bincount(labels, minlength=int(max(Klass)) + 1)
    parts = [f"{Klass(k).name}={c}" for k, c in enumerate(counts) if c > 0]
    log("      classes: " + ", ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic BEV ground model from COLMAP + segmentation")
    ap.add_argument("--model", required=True, help="COLMAP text model dir (cameras/images/points3D.txt)")
    ap.add_argument("--images", required=True, help="directory of source images")
    ap.add_argument("--out", required=True, help="output directory")
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

    run(args.model, args.images, args.out, segmenter_kind=args.segmenter,
        cell_size=args.cell_size, up_axis=args.up_axis, up_sign=args.up_sign,
        limit=args.limit, clip_threshold=args.clip_threshold,
        overwrite_masks=args.overwrite_masks, semantic_mode=args.semantic_mode)


if __name__ == "__main__":
    main()
