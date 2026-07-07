"""Single source of truth for grid-cell <-> world-metre conversion.

The raster's row axis is **gravity-canonical**: `row = v_sign * world[v_axis]` (set once in
`ground_model.build_level`, so a Y-up and a Y-down source model rasterise the same way). Every
converter that turns a `(col, row)` into world metres — or back — MUST go through here, so the
sign lives in exactly one place. A forgotten sign silently mirrors the map (that is precisely
the bug this module exists to make unrepresentable).

Accepts either the `meta` dict (from `level0.meta.json`) or a `GridSpec` object.
"""

from __future__ import annotations

import numpy as np


def _get(spec, key: str):
    return spec[key] if isinstance(spec, dict) else getattr(spec, key)


def _v_sign(spec) -> float:
    try:
        return float(_get(spec, "v_sign"))
    except (KeyError, AttributeError):
        return 1.0  # pre-canonical metas default to no flip


def cell_to_world(col, row, spec):
    """(col, row) grid indices -> (u, v) world metres along the two horizontal axes.

    Scalars or numpy arrays. For a cell *centre* pass ``col + 0.5``, ``row + 0.5``.
    """
    cs = _get(spec, "cell_size")
    u = _get(spec, "origin_u") + np.asarray(col, dtype=np.float64) * cs
    v = _v_sign(spec) * (_get(spec, "origin_v") + np.asarray(row, dtype=np.float64) * cs)
    return u, v


def world_to_cell(u, v, spec):
    """(u, v) world metres -> fractional (col, row) grid indices (inverse of cell_to_world)."""
    cs = _get(spec, "cell_size")
    col = (np.asarray(u, dtype=np.float64) - _get(spec, "origin_u")) / cs
    row = (_v_sign(spec) * np.asarray(v, dtype=np.float64) - _get(spec, "origin_v")) / cs
    return col, row


def cells_to_world(colrow, spec):
    """(N,2) array of (col,row) -> (N,2) array of (u,v) world metres."""
    colrow = np.asarray(colrow, dtype=np.float64)
    u, v = cell_to_world(colrow[:, 0], colrow[:, 1], spec)
    return np.stack([u, v], axis=1)
