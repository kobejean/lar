"""Path centerlines: turn blob walkway regions into a smooth centerline graph + clean paths.

A human draws a path as a centerline with a width, not a wiggly polygon. So we skeletonise the
walkway mask, trace it into a graph (nodes = junctions/endpoints, edges = polylines), prune
skeleton spurs, smooth each edge, and re-buffer by the measured width into clean constant-width
paths. The graph doubles as a routing network.

    walkway mask --skeletonize--> graph --prune+smooth--> centerlines (+width) --buffer--> paths
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
from skimage.morphology import skeletonize

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_STEP = {o: (2 ** 0.5 if o[0] and o[1] else 1.0) for o in _NB}


def _trace(skel: np.ndarray):
    """1px skeleton -> (edges, degree). Each edge is a pixel polyline between non-degree-2 nodes."""
    pts = set(map(tuple, np.argwhere(skel)))

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in _NB if (y + dy, x + dx) in pts]

    deg = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    edges, seen = [], set()
    for n in nodes:
        for m in nbrs(n):
            if (n, m) in seen:
                continue
            path, prev, cur = [n], n, m
            while True:
                path.append(cur)
                seen.add((prev, cur))
                seen.add((cur, prev))
                if cur in nodes:
                    break
                nxt = [q for q in nbrs(cur) if q != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
            edges.append(path)
    return edges, deg


def _chaikin(pts: np.ndarray, iters: int) -> np.ndarray:
    pts = np.asarray(pts, float)
    for _ in range(iters):
        if len(pts) < 3:
            break
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append(0.75 * a + 0.25 * b)
            out.append(0.25 * a + 0.75 * b)
        out.append(pts[-1])
        pts = np.array(out)
    return pts


def _pix_len(path) -> float:
    return sum(_STEP[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path[:-1], path[1:]))


def walkway_centerlines(mask: np.ndarray, meta: dict, prune_m: float = 1.5,
                        min_len_m: float = 1.0, smooth_iters: int = 2,
                        min_component_m2: float = 10.0) -> list[dict]:
    """Walkway mask -> centerline segments. Each: {'line': (N,2) world XZ, 'width': metres}."""
    cs = meta["cell_size"]
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # heal gaps before skeleton

    # drop disconnected speckle islands (only skeletonise real path components)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] * cs * cs >= min_component_m2:
            keep[lbl == i] = 1
    m = keep
    skel = skeletonize(m.astype(bool))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)                      # px to boundary
    edges, deg = _trace(skel)

    out = []
    for path in edges:
        length_m = _pix_len(path) * cs
        is_spur = deg[path[0]] == 1 or deg[path[-1]] == 1
        if (is_spur and length_m < prune_m) or length_m < min_len_m:
            continue
        world = np.array([[meta["origin_u"] + x * cs, meta["origin_v"] + y * cs] for y, x in path])
        world = _chaikin(world, smooth_iters)
        halfw = np.array([dist[y, x] for y, x in path]) * cs            # dist = half-width
        out.append({"line": world, "width": float(2 * np.median(halfw))})
    return out


def centerlines_to_polygons(centerlines: list[dict], min_width_m: float = 0.6):
    """Re-buffer each smooth centerline by half its width -> clean constant-width path polygons."""
    polys = []
    for c in centerlines:
        if len(c["line"]) < 2:
            continue
        r = max(c["width"] / 2, min_width_m / 2)
        polys.append(LineString(c["line"]).buffer(r, cap_style=1, join_style=1))
    return unary_union(polys) if polys else None
