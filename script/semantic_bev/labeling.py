"""Turn per-image segmentation masks into a semantic label per 3D point.

Because a COLMAP point's track stores the exact keypoint it was observed at in every
image, labelling is just: for each observing image, sample that image's mask at the stored
pixel and cast a vote. No reprojection, no pose maths.

Masks are cached to disk as single-channel (grayscale) PNGs of class ids, so segmentation
(the expensive step) runs once and labelling/tuning can re-run freely. Voting streams one
mask at a time, so memory stays flat regardless of image count.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from colmap_io import Reconstruction
from segmentation import Segmenter, save_mask_png
from taxonomy import Klass


class MaskStore:
    """On-disk cache of class-id masks, keyed by image name."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _id_path(self, name: str) -> Path:
        return self.dir / f"{Path(name).stem}.png"

    def _preview_path(self, name: str) -> Path:
        return self.dir / f"{Path(name).stem}_preview.png"

    def has(self, name: str) -> bool:
        return self._id_path(name).exists()

    def save(self, name: str, mask: np.ndarray) -> None:
        cv2.imwrite(str(self._id_path(name)), mask)  # single channel = exact ids
        save_mask_png(mask, self._preview_path(name))

    def load(self, name: str) -> np.ndarray:
        m = cv2.imread(str(self._id_path(name)), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"no cached mask for {name}")
        return m


def segment_and_cache(recon: Reconstruction, image_dir: str | Path,
                      segmenter: Segmenter, store: MaskStore,
                      image_ids: list[int] | None = None,
                      overwrite: bool = False, log=print) -> list[int]:
    """Segment each (selected) image once and cache its mask. Returns processed image ids."""
    image_dir = Path(image_dir)
    ids = image_ids if image_ids is not None else sorted(recon.images)
    done = []
    for n, img_id in enumerate(ids):
        img = recon.images[img_id]
        if not overwrite and store.has(img.name):
            done.append(img_id)
            continue
        mask = segmenter.segment_file(image_dir / img.name)
        store.save(img.name, mask)
        done.append(img_id)
        if (n + 1) % 25 == 0 or n + 1 == len(ids):
            log(f"  segmented {n + 1}/{len(ids)} images")
    return done


def accumulate_votes(recon: Reconstruction, store: MaskStore,
                     image_ids: list[int]) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Stream masks and tally per-point class votes.

    Returns ``(point_ids, votes, index)`` where ``votes`` is (P, num_classes) uint16 and
    ``index`` maps a COLMAP point id to its row.
    """
    point_ids = np.array(sorted(recon.points3d), dtype=np.int64)
    index = {int(pid): i for i, pid in enumerate(point_ids)}
    n_classes = int(max(Klass)) + 1
    votes = np.zeros((len(point_ids), n_classes), dtype=np.uint16)

    id_set = set(image_ids)
    for img_id in image_ids:
        img = recon.images[img_id]
        mask = store.load(img.name)
        H, W = mask.shape
        pid = img.point3d_ids
        valid = pid >= 0
        if not valid.any():
            continue
        xy = img.xys[valid]
        pids = pid[valid]
        xs = np.clip(np.round(xy[:, 0]).astype(np.int64), 0, W - 1)
        ys = np.clip(np.round(xy[:, 1]).astype(np.int64), 0, H - 1)
        klasses = mask[ys, xs]
        for pid_val, k in zip(pids, klasses):
            row = index.get(int(pid_val))
            if row is not None:
                votes[row, k] += 1
    return point_ids, votes, index


def resolve_labels(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote -> (label per point, confidence in [0,1]).

    UNKNOWN votes are ignored unless a point has *only* unknown/no votes, so a single
    confident ground/obstacle observation beats many unknowns.
    """
    labels = np.full(votes.shape[0], int(Klass.UNKNOWN), dtype=np.uint8)
    conf = np.zeros(votes.shape[0], dtype=np.float32)

    known = votes.copy()
    known[:, int(Klass.UNKNOWN)] = 0
    known_total = known.sum(axis=1)
    has_known = known_total > 0

    labels[has_known] = known[has_known].argmax(axis=1).astype(np.uint8)
    conf[has_known] = known[has_known].max(axis=1) / known_total[has_known]
    return labels, conf
