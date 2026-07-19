#!/usr/bin/env python3
"""Score the depth bake-off: compare BEV levels from each geometry backend side by side.

Loads each backend's ``level0.npz`` (from ``pipeline.py`` with the corresponding source)
and reports, per backend:

  - coverage %       -- fraction of grid cells with observed ground geometry
  - ground cells     -- absolute observed count (denser depth -> more)
  - blocked cells    -- occupancy footprints (obstacles/objects)
  - height roughness -- median |∇height| over observed cells (proxy for surface noise;
                        lower = smoother, so MVS/2DGS should beat mono/3DGS if cleaner)
  - class mix        -- semantic distribution over observed ground

and renders a combined semantic+height comparison image. Footprint IoU vs. hand-marked
objects is a later addition (needs instance segmentation), flagged as TODO.

  uv run python eval_depth_bev.py --session maguro-park-after-itchy \
      --backends colmap,mvs,mono,3dgs,2dgs --out ../../output/<name>-bakeoff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from taxonomy import Klass, color_lut

BLOCKED = 1


def _sbev_dir(session, backend):
    if backend in ("colmap", "gsplat"):
        return session.sbev_out(backend)
    return session.sbev_out("depth", tag=backend)


def load_level(d: Path):
    npz = np.load(d / "level0.npz")
    meta = json.loads((d / "level0.meta.json").read_text())
    return dict(height=npz["height"], semantic=npz["semantic"],
                occupancy=npz["occupancy"], coverage=npz["coverage"], meta=meta)


def metrics(lv) -> dict:
    cov = lv["coverage"]
    h = lv["height"].astype(np.float32)
    obs = cov & np.isfinite(h)
    gy, gx = np.gradient(np.nan_to_num(h))
    grad = np.sqrt(gy ** 2 + gx ** 2)
    rough = float(np.median(grad[obs])) if obs.any() else float("nan")
    classes = np.bincount(lv["semantic"][cov], minlength=int(max(Klass)) + 1)
    top = sorted(((c, int(n)) for c, n in enumerate(classes) if n), key=lambda x: -x[1])[:4]
    return {
        "cells": int(cov.size),
        "coverage_%": round(100 * cov.mean(), 1),
        "ground_cells": int(cov.sum()),
        "blocked_cells": int((lv["occupancy"] == BLOCKED).sum()),
        "height_roughness_m": round(rough, 3),
        "top_classes": ", ".join(f"{Klass(c).name}:{n}" for c, n in top),
    }


def render_comparison(levels: dict, out: Path):
    lut = np.array(color_lut(), np.uint8)
    tiles = []
    for name, lv in levels.items():
        sem = lut[lv["semantic"]]
        sem[~lv["coverage"]] = (30, 30, 30)
        sem = cv2.cvtColor(sem, cv2.COLOR_RGB2BGR)
        sem = cv2.copyMakeBorder(sem, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(sem, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(sem)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    cv2.imwrite(str(out / "bakeoff_semantic.png"), np.hstack(tiles))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--backends", default="colmap,mvs,mono,3dgs,2dgs",
                    help="comma list; each resolves to its sbev output dir")
    ap.add_argument("--out", default=None, help="scorecard dir (default: <name>-bakeoff)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lar_session import Session
    s = Session(args.session)
    out = Path(args.out) if args.out else s.root / "output" / f"{s.name}-bakeoff"
    out.mkdir(parents=True, exist_ok=True)

    levels, rows = {}, []
    for b in [x.strip() for x in args.backends.split(",") if x.strip()]:
        d = _sbev_dir(s, b)
        if not (d / "level0.npz").exists():
            print(f"skip {b}: no level0.npz at {d}")
            continue
        lv = load_level(d)
        levels[b] = lv
        m = metrics(lv)
        rows.append((b, m))
        print(f"\n[{b}]  {d.name}")
        for k, v in m.items():
            print(f"    {k:20} {v}")

    if not levels:
        raise SystemExit("no backend outputs found — run pipeline.py per backend first")

    # Markdown scorecard.
    cols = ["coverage_%", "ground_cells", "blocked_cells", "height_roughness_m", "top_classes"]
    md = ["| backend | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for b, m in rows:
        md.append(f"| {b} | " + " | ".join(str(m[c]) for c in cols) + " |")
    (out / "scorecard.md").write_text("\n".join(md) + "\n")
    render_comparison(levels, out)
    print(f"\nscorecard -> {out}/scorecard.md + bakeoff_semantic.png")
    print("TODO: footprint IoU vs hand-marked objects (needs instance segmentation)")


if __name__ == "__main__":
    main()
