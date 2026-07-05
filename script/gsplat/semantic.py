"""Semantic supervision for the semantic 3DGS head.

We distil a 2D image segmenter into the Gaussians: run the segmenter once per training
image, cache the class-id masks, and cross-entropy the rendered per-Gaussian logits
against them during training. The 3D Gaussians become the view-consistent fusion of the
2D masks -- exactly the "geometry source behind (positions, labels)" that
``script/semantic_bev`` is designed to consume.

The segmenters, taxonomy (class enum + colours), and on-disk mask cache are reused
verbatim from ``script/semantic_bev`` so there is a single semantic contract across the
BEV and 3DGS pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the semantic_bev semantic stack (single source of truth for the taxonomy).
_SBEV = Path(__file__).resolve().parent.parent / "semantic_bev"
if str(_SBEV) not in sys.path:
    sys.path.insert(0, str(_SBEV))

from labeling import MaskStore          # noqa: E402
from segmentation import make_segmenter  # noqa: E402
from taxonomy import Klass, color_lut    # noqa: E402

NUM_CLASSES = int(max(Klass)) + 1
IGNORE_INDEX = int(Klass.UNKNOWN)        # unlabeled pixels excluded from the CE loss
SEMANTIC_COLOR_LUT = color_lut()


def build_mask_cache(
    image_names: list[str],
    image_dir: str | Path,
    cache_dir: str | Path,
    segmenter_kind: str = "oneformer",
    overwrite: bool = False,
    log=print,
) -> MaskStore:
    """Segment each image once (full-res) and cache the class-id mask. Idempotent."""
    image_dir = Path(image_dir)
    store = MaskStore(cache_dir)
    todo = [n for n in image_names if overwrite or not store.has(n)]
    if not todo:
        log(f"semantic: {len(image_names)} masks already cached ({segmenter_kind})")
        return store

    log(f"semantic: segmenting {len(todo)} images with {segmenter_kind} ...")
    seg = make_segmenter(segmenter_kind)  # lazy: loads the model only when needed
    for i, name in enumerate(todo):
        store.save(name, seg.segment_file(image_dir / name))
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            log(f"  segmented {i + 1}/{len(todo)}")

    # Release the segmenter's GPU memory before training. Matters on an 8 GB card for the
    # heavier backends (mask2former-large is a swin-large model).
    del seg
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return store


def load_mask(store: MaskStore, name: str, width: int, height: int) -> np.ndarray:
    """Load a cached class-id mask resized (nearest) to the training resolution."""
    m = store.load(name)
    if m.shape[1] != width or m.shape[0] != height:
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m
