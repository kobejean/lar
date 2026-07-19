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



def surroundedness(mask: np.ndarray, radius: int) -> np.ndarray:
    """For each cell, how many of the 8 compass directions hit `mask` within `radius`.

    `fill_small_holes` only closes *fully enclosed* voids, so an object whose base contacts
    were seen from one side stays a C-shaped rim with its middle open to the outside -- the
    hollow-object look. "Nearly surrounded" is the missing test: a cell walled in on 6 of 8
    sides is inside the object even though a gap technically connects it to the exterior.

    Direction-counting rather than a bigger morphological close, because a close with a kernel
    wide enough to bridge the gap also welds together neighbouring objects that merely pass
    near each other.
    """
    h, w = mask.shape
    hits = np.zeros((h, w), np.uint8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        seen = np.zeros((h, w), bool)
        cur = mask.copy()
        for _ in range(radius):
            cur = np.roll(cur, (dy, dx), (0, 1))
            # a roll wraps; kill the wrapped edge so the far side of the map cannot vote
            if dy > 0:
                cur[0, :] = False
            elif dy < 0:
                cur[-1, :] = False
            if dx > 0:
                cur[:, 0] = False
            elif dx < 0:
                cur[:, -1] = False
            seen |= cur
        hits += seen.astype(np.uint8)
    return hits


def fill_surrounded(mask: np.ndarray, radius: int, min_dirs: int, rounds: int = 3) -> np.ndarray:
    """Grow `mask` into cells walled in on >= min_dirs of 8 sides. Iterated, because each
    round makes the next concavity shallower."""
    if min_dirs > 8 or radius < 1:
        return mask
    out = mask.copy()
    for _ in range(rounds):
        grow = (surroundedness(out, radius) >= min_dirs) & ~out
        if not grow.any():
            break
        out |= grow
    return out


def range_limit_true(near: np.ndarray, max_m: float) -> np.ndarray:
    """Keep cells the camera actually came within `max_m` of (from `near_range` in the npz).

    Strictly better than the distance-from-core proxy below: view COUNT cannot distinguish a
    cell seen 60 times from 30 m away at a grazing angle from one seen 6 times at 4 m, and it
    is the former that is noisy. -1 marks cells never raycast."""
    if max_m <= 0:
        return np.ones_like(near, bool)
    return (near >= 0) & (near <= max_m)


def range_limit(confident: np.ndarray, cell_size: float, max_m: float) -> np.ndarray:
    """Cells within `max_m` of the well-observed core.

    Noise is not spread evenly -- it lives at the coverage frontier, far from where the camera
    actually went, where a cell has one or two grazing observations. Rather than trusting a
    vote threshold to sort that out, cut on distance from the confident core outright. The
    trajectory itself is not in the npz (no DEM origin is stored), but the set of
    well-observed cells is a faithful stand-in for where the camera was.
    """
    if max_m <= 0 or not confident.any():
        return np.ones_like(confident, bool)
    d = cv2.distanceTransform((~confident).astype(np.uint8), cv2.DIST_L2, 5)
    return d * cell_size <= max_m


def crop_to_content(labels: np.ndarray, margin: int = 3) -> np.ndarray:
    """Trim the dead border so the map fills its frame."""
    ys, xs = np.where(labels != L_UNSURVEYED)
    if not len(ys):
        return labels
    y0, y1 = max(0, ys.min() - margin), min(labels.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(labels.shape[1], xs.max() + margin + 1)
    return labels[y0:y1, x0:x1]


def absorb_enclosed(occluded: np.ndarray, occupied: np.ndarray,
                    min_frac: float, max_area: int, log=None) -> np.ndarray:
    """Occluded pockets whose BORDER is mostly occupied are interior, not walkable ground.

    HIDDEN means "in frustum but never seen walkable" -- which covers two physically different
    things. A pocket ringed by base contacts is the middle of a tree clump or hedge mass: the
    ground is invisible because the object stands on it, and a router must treat it as solid.
    A pocket open on one side is merely that object's occlusion shadow, and the ground there is
    genuinely unknown -- claiming it is occupied would over-block, the failure this pipeline has
    rejected repeatedly (it is why mono depth was dropped for occupancy).

    So enclosure is measured per connected component as the fraction of its dilated border that
    is occupied, and only components above `min_frac` are absorbed. A convex hull would not
    distinguish the two cases: the hull of a curved hedge sweeps straight across the path it
    borders.
    """
    if min_frac <= 0 or not occluded.any():
        return occupied
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occluded.astype(np.uint8), 8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = occupied.copy()
    absorbed = 0
    for i in range(1, n):
        comp = lab == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_area:
            continue                                  # too big to be an object interior
        border = cv2.dilate(comp.astype(np.uint8), ker).astype(bool) & ~comp
        nb = int(border.sum())
        if nb == 0:
            continue
        if int((border & occupied).sum()) / nb >= min_frac:
            out |= comp
            absorbed += 1
    if log:
        log(f"  absorbed {absorbed} enclosed occluded pockets into occupied")
    return out


def absorb_walled(occluded: np.ndarray, occupied: np.ndarray, radius: int,
                  min_dirs: int) -> np.ndarray:
    """Absorb occluded cells walled in by occupied on >= min_dirs of 8 compass directions.

    Looser than the component test: it fires per cell, so it catches the deep part of a
    concave shadow without needing the whole pocket to be ringed. Measured on the full park
    run at radius 6: >=5 dirs absorbs 179 of 4414 occluded cells (4%), >=6 only 41 (0.9%) --
    small, because occupied is a CRESCENT not a ring. Base contacts come from the path-facing
    side, so red rarely encircles anything, and most blue is genuine occlusion shadow whose
    ground is unknown rather than object interior. Kept conservative for that reason.
    """
    if min_dirs > 8 or min_dirs <= 0 or not occluded.any() or not occupied.any():
        return occupied
    return occupied | (occluded & (surroundedness(occupied, radius) >= min_dirs))


def build_labels(z, args) -> tuple[np.ndarray, dict]:
    state = z["state"]
    surface = z["surface"] if "surface" in z else np.zeros_like(state)
    obs = z["obs_count"] if "obs_count" in z else np.full(state.shape, 99, np.int32)
    bearings = z["foot_bearings"] if "foot_bearings" in z else np.zeros_like(state, np.int32)

    # The survey frontier is dithered at cell level (one ray lands, its neighbour misses), which
    # renders as salt-and-pepper along every edge. Smooth the *masks* so the boundary reads as a
    # boundary; this changes only where we admit to knowing, never what we claim to know.
    # De-speckle BEFORE smoothing: a lone surveyed cell survives the close as a plus-shaped
    # nub (the kernel's own footprint), which is why crosses were littering the border.
    surveyed = smooth(remove_small_blobs(obs > 0, args.min_blob), args.smooth)
    confident = smooth(remove_small_blobs(obs >= args.min_views, args.min_blob), args.smooth)

    # Range is measured from the WELL-travelled core, not merely the confident set: obs peaks
    # along the walked route, so a high threshold approximates the trajectory the npz does not
    # store. Measuring from `confident` (>=4 views) made the core so broad that 25 m reached
    # the frontier it was meant to exclude.
    if "near_range" in z:
        in_range = range_limit_true(z["near_range"], args.max_range)
    else:
        # Older npz: fall back to distance from the well-travelled core. Weak here, because the
        # park is uniformly well-observed -- obs>=12 covers half the grid and every surveyed
        # cell sits within 12 m of it, so the proxy has almost nothing to cut.
        core = remove_small_blobs(obs >= args.core_views, args.min_blob)
        if not core.any():
            core = confident
        in_range = range_limit(core, float(z["cell_size"]), args.max_range)
    surveyed &= in_range
    confident &= in_range

    occupied = (state == FOOTPRINT) & (bearings >= args.min_bearings) & in_range
    hidden = (state == HIDDEN) & confident
    free = (state == FREE) & confident

    path = free & np.isin(surface, [int(Klass.PATH), int(Klass.PAVEMENT), int(Klass.STAIRS)])
    grass = free & (surface == int(Klass.GRASS))
    terrain = free & (surface == int(Klass.TERRAIN))
    # FREE but unclassified surface -> walkable yet we cannot say what it is: that is
    # low-confidence information, not terrain. Rendering it as terrain would be a guess.
    unclassified = free & ~(path | grass | terrain)

    # do this before the per-layer morphology, so absorbed pockets get the same
    # close/fill treatment as the rest of the occupied mask
    occupied = absorb_enclosed(hidden, occupied, args.enclose_frac, args.enclose_max_area)
    occupied = absorb_walled(hidden, occupied, args.surround_radius, args.enclose_dirs)
    hidden = hidden & ~occupied

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
            m = fill_surrounded(m, args.surround_radius, args.surround_dirs)
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


def colorize(labels, scale, legend, cell_size=0.5, outline=True):
    lut = np.zeros((len(PALETTE), 3), np.uint8)
    for k, v in PALETTE.items():
        lut[k] = v
    total = labels.size
    pct = {lid: 100.0 * float((labels == lid).sum()) / total for lid in NAMES}

    rgb = np.flipud(lut[labels])                          # north-up, as footprint2d renders
    occ = np.flipud(labels == L_OCCUPIED)
    if scale > 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        occ = cv2.resize(occ.astype(np.uint8), None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    if outline:
        # A dark keyline round occupied regions. Obstacles are the one class a user must not
        # misread as ground, and at small sizes a fill alone reads as a colour blotch.
        cnts, _ = cv2.findContours(occ.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, (70, 22, 18), max(1, scale // 3))

    # scale bar: a map without one cannot be measured, and this is metric
    h, w = rgb.shape[:2]
    px_per_m = scale / cell_size
    target = max(5.0, round((w * 0.18) / px_per_m / 5.0) * 5.0)      # ~18% of width, round 5 m
    bar = int(target * px_per_m)
    x0, y0 = 18, h - 26
    cv2.rectangle(rgb, (x0, y0), (x0 + bar, y0 + 6), (245, 245, 245), -1)
    cv2.rectangle(rgb, (x0, y0), (x0 + bar, y0 + 6), (20, 20, 20), 1)
    cv2.putText(rgb, f"{target:.0f} m", (x0, y0 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (245, 245, 245), 1, cv2.LINE_AA)
    # north arrow (rows increase northward before the flip, so up is north here)
    nx, ny = w - 34, 34
    cv2.arrowedLine(rgb, (nx, ny + 20), (nx, ny - 12), (245, 245, 245), 2, tipLength=0.4)
    cv2.putText(rgb, "N", (nx - 6, ny + 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (245, 245, 245), 1, cv2.LINE_AA)

    if not legend:
        return rgb
    pad, sw, lh = 14, 18, 27
    bar_w = 232
    panel = np.full((h, bar_w, 3), PALETTE[L_UNSURVEYED], np.uint8)
    for i, (lid, name) in enumerate(NAMES.items()):
        y = pad + i * lh
        panel[y:y + sw, pad:pad + sw] = PALETTE[lid]
        cv2.rectangle(panel, (pad, y), (pad + sw, y + sw), (20, 20, 20), 1)
        cv2.putText(panel, name, (pad + sw + 9, y + sw - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 238, 242), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{pct[lid]:4.1f}%", (pad + sw + 9, y + sw + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 155, 165), 1, cv2.LINE_AA)
    return np.hstack([rgb, panel])


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
    ap.add_argument("--core-views", type=int, default=12,
                    help="views defining the well-travelled core that --max-range is measured "
                         "from; obs peaks along the walked route, so this approximates the "
                         "trajectory (which the npz does not store)")
    ap.add_argument("--max-range", type=float, default=18.0,
                    help="metres from the well-observed core beyond which cells are dropped "
                         "entirely. Noise lives at the coverage frontier, so cut on distance "
                         "rather than hoping a vote threshold sorts it out. 0 disables.")
    ap.add_argument("--enclose-frac", type=float, default=0.65,
                    help="fraction of an occluded pocket's border that must be occupied before "
                         "the pocket is absorbed as occupied (it is the inside of an object, "
                         "not walkable ground). 0 disables; high values keep occlusion "
                         "shadows -- which are genuinely unknown -- out of it.")
    ap.add_argument("--enclose-dirs", type=int, default=5,
                    help="absorb an occluded cell into occupied when this many of 8 compass "
                         "directions hit occupied within --surround-radius. 9 disables. "
                         "Conservative on purpose: most occluded ground is a one-sided "
                         "occlusion shadow, and calling that solid would over-block.")
    ap.add_argument("--enclose-max-area", type=int, default=400,
                    help="occluded pockets larger than this many cells are never absorbed")
    ap.add_argument("--surround-radius", type=int, default=6,
                    help="how far (cells) the 8 direction probes look when deciding a cell is "
                         "inside an object")
    ap.add_argument("--surround-dirs", type=int, default=6,
                    help="of 8 directions, how many must hit occupied before an interior cell "
                         "is filled. 8 = fully enclosed only; 6 tolerates a C-shaped rim. "
                         "Lower bleeds objects outward.")
    ap.add_argument("--no-crop", action="store_true", help="keep the empty border")
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

    if not args.no_crop:
        labels = crop_to_content(labels)
    out = Path(args.out) if args.out else npz.parent / "bev_clean.png"
    img = colorize(labels, args.scale, not args.no_legend, float(z["cell_size"]))
    cv2.imwrite(str(out), img[:, :, ::-1])
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
