"""Presentation-grade BEV map from a footprint2d raster.

`footprint2d.npz` is *evidence*: every cell is whatever the multi-view votes actually support,
speckle and all. That is the right thing for a raster you compute against, and the wrong thing
to put in a UI -- a map that renders a lone mis-voted cell as confidently as a corridor seen
101 times reads as broken, and an object drawn as a hollow ring of base-contacts reads as a
hole in the world.

So this is a separate, deliberately **lossy** layer. Nothing here feeds back into the npz; the
evidence stays untouched and re-rendering is instant instead of an hour.

Three things it does that the evidence raster must not:

1. **Says "I don't know" out loud.** A cell seen twice and a cell seen 101 times are both just
   FREE in the raster. Here anything below `--min-views` renders **grey** rather than
   committing to a class, so thin evidence looks uncertain instead of authoritative. This is
   what stops the coverage-frontier speckle from reading as real structure.
2. **Fills occupied interiors.** FOOTPRINT marks *base contacts* -- the camera-facing rim of an
   object -- so a building or hedge comes out as an outline around unobserved middle. Small
   enclosed holes are filled so objects render solid. Only *small* ones: a courtyard ringed by
   trees is a real hole and stays open (`--max-fill`).
3. **Drops speckle.** Connected components below `--min-blob` cells are demoted to grey, not to
   a neighbouring class -- an isolated cell is unsupported, not evidence for whatever surrounds
   it.

Run (from repo root):
  uv run python script/backbone/bev_render.py --npz output/<session>-footprint2d/footprint2d.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "semantic_bev"))

from taxonomy import Klass  # noqa: E402

UNKNOWN, FREE, FOOTPRINT, HIDDEN = 0, 1, 2, 3

# presentation labels (priority order matters when classes overlap after morphology)
L_UNSURVEYED, L_LOWCONF, L_GRASS, L_TERRAIN, L_PATH, L_HIDDEN, L_OCCUPIED = range(7)

PALETTE = {                      # RGB
    L_UNSURVEYED: (24, 26, 30),
    L_LOWCONF:    (105, 108, 115),
    L_GRASS:      (104, 152, 92),
    L_TERRAIN:    (150, 132, 106),
    L_PATH:       (238, 220, 170),
    L_HIDDEN:     (72, 88, 104),
    L_OCCUPIED:   (188, 74, 62),
}
NAMES = {
    L_UNSURVEYED: "not surveyed", L_LOWCONF: "low confidence", L_GRASS: "grass",
    L_TERRAIN: "terrain", L_PATH: "path / paved", L_HIDDEN: "occluded ground",
    L_OCCUPIED: "occupied",
}


def remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than min_area cells."""
    if min_area <= 1 or not mask.any():
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill enclosed holes up to max_area cells; larger voids are real and stay open.

    Base contacts trace the camera-facing rim of an object, so its middle is never observed.
    Filling that reads correctly. Filling a tree-ringed courtyard would not -- hence the cap,
    and hence skipping any component touching the border (that is outside, not a hole).
    """
    if max_area <= 0 or not mask.any():
        return mask
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = mask.shape
    out = mask.copy()
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    for i in range(1, n):
        if i in border:
            continue
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            out[lab == i] = True
    return out


def close_only(mask: np.ndarray, k: int) -> np.ndarray:
    """Dilate-then-erode: knits a broken rim into a closed outline WITHOUT shedding small
    blobs. `smooth` follows its close with an open, which erases anything smaller than the
    kernel -- fatal for occupied, where a trunk or bench is one or two cells at 0.5 m."""
    if k < 3 or not mask.any():
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)


def smooth(mask: np.ndarray, k: int) -> np.ndarray:
    """Close then open with the same kernel: knits ragged edges, then sheds the nubs."""
    if k < 3 or not mask.any():
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    return m.astype(bool)


def build_labels(z, args) -> tuple[np.ndarray, dict]:
    state = z["state"]
    surface = z["surface"] if "surface" in z else np.zeros_like(state)
    obs = z["obs_count"] if "obs_count" in z else np.full(state.shape, 99, np.int32)
    bearings = z["foot_bearings"] if "foot_bearings" in z else np.zeros_like(state, np.int32)

    # The survey frontier is dithered at cell level (one ray lands, its neighbour misses), which
    # renders as salt-and-pepper along every edge. Smooth the *masks* so the boundary reads as a
    # boundary; this changes only where we admit to knowing, never what we claim to know.
    surveyed = smooth(obs > 0, args.smooth)
    confident = smooth(obs >= args.min_views, args.smooth)

    occupied = (state == FOOTPRINT) & (bearings >= args.min_bearings)
    hidden = (state == HIDDEN) & confident
    free = (state == FREE) & confident

    path = free & np.isin(surface, [int(Klass.PATH), int(Klass.PAVEMENT), int(Klass.STAIRS)])
    grass = free & (surface == int(Klass.GRASS))
    terrain = free & (surface == int(Klass.TERRAIN))
    # FREE but unclassified surface -> walkable yet we cannot say what it is: that is
    # low-confidence information, not terrain. Rendering it as terrain would be a guess.
    unclassified = free & ~(path | grass | terrain)

    layers = {L_OCCUPIED: occupied, L_PATH: path, L_TERRAIN: terrain,
              L_GRASS: grass, L_HIDDEN: hidden}

    stats = {}
    for lid, m in layers.items():
        before = int(m.sum())
        if lid == L_OCCUPIED:
            # Occupied is the sparsest layer by construction -- FOOTPRINT marks base contacts,
            # a camera-facing rim, and at 0.5 m a bench or trunk is one or two cells. The blob
            # filter tuned for area classes deletes exactly those, so occupied gets its own
            # (small) threshold, is closed first so a broken rim becomes a ring, and only then
            # has its interior filled.
            m = remove_small_blobs(m, args.min_blob_occupied)
            m = close_only(m, args.occupied_close)
            m = fill_small_holes(m, args.max_fill)
        else:
            m = remove_small_blobs(m, args.min_blob)
            m = smooth(m, args.smooth)
        layers[lid] = m
        stats[NAMES[lid]] = (before, int(m.sum()))

    out = np.full(state.shape, L_UNSURVEYED, np.uint8)
    out[surveyed] = L_LOWCONF                      # surveyed but not confident -> grey
    out[unclassified] = L_LOWCONF
    for lid in (L_HIDDEN, L_GRASS, L_TERRAIN, L_PATH, L_OCCUPIED):   # last wins
        out[layers[lid]] = lid
    return out, stats


def colorize(labels: np.ndarray, scale: int, legend: bool) -> np.ndarray:
    lut = np.zeros((len(PALETTE), 3), np.uint8)
    for k, v in PALETTE.items():
        lut[k] = v
    rgb = lut[labels]
    rgb = np.flipud(rgb)                                  # north-up, as footprint2d renders
    if scale > 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if not legend:
        return rgb
    pad, sw, lh = 14, 18, 26
    bar = np.full((rgb.shape[0], 210, 3), PALETTE[L_UNSURVEYED], np.uint8)
    for i, (lid, name) in enumerate(NAMES.items()):
        y = pad + i * lh
        bar[y:y + sw, pad:pad + sw] = PALETTE[lid]
        cv2.putText(bar, name, (pad + sw + 9, y + sw - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 238, 242), 1, cv2.LINE_AA)
    return np.hstack([rgb, bar])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None, help="output png (default: <npz dir>/bev_clean.png)")
    ap.add_argument("--min-views", type=int, default=4,
                    help="views a cell needs before it is drawn as a class at all; below this "
                         "it renders grey instead of committing")
    ap.add_argument("--min-bearings", type=int, default=2,
                    help="distinct bearings required to draw a cell as occupied")
    ap.add_argument("--min-blob", type=int, default=4,
                    help="drop connected components smaller than this many cells (speckle)")
    ap.add_argument("--min-blob-occupied", type=int, default=2,
                    help="separate, smaller speckle threshold for occupied cells (base "
                         "contacts are sparse; the area-class threshold would erase objects)")
    ap.add_argument("--occupied-close", type=int, default=5,
                    help="morphological kernel used to knit broken base-contact rims into a "
                         "closed outline before the interior is filled")
    ap.add_argument("--max-fill", type=int, default=40,
                    help="fill enclosed holes in occupied regions up to this many cells; "
                         "larger voids are real (a courtyard) and stay open")
    ap.add_argument("--smooth", type=int, default=3, help="morphological kernel (cells), <3 off")
    ap.add_argument("--scale", type=int, default=4, help="upscale factor")
    ap.add_argument("--no-legend", action="store_true")
    args = ap.parse_args()

    npz = Path(args.npz).expanduser().resolve()
    z = np.load(npz)
    labels, stats = build_labels(z, args)

    total = labels.size
    print(f"[bev_render] {npz}")
    for name, (before, after) in stats.items():
        d = after - before
        print(f"  {name:<16s} {before:>6d} -> {after:>6d} cells ({d:+d})")
    for lid, name in NAMES.items():
        n = int((labels == lid).sum())
        print(f"  final {name:<16s} {100*n/total:5.1f}%")

    out = Path(args.out) if args.out else npz.parent / "bev_clean.png"
    cv2.imwrite(str(out), colorize(labels, args.scale, not args.no_legend)[:, :, ::-1])
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
