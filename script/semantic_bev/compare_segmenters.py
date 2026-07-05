"""Side-by-side comparison of segmentation backends on the same images.

Runs each `--segmenter` over the same sample images and writes one grid PNG
(rows = images, cols = source + one overlay per model) plus a timing / class-mix table,
so we can pick a backend before committing to a full-park run.

    uv run python compare_segmenters.py \
        --images ../../input/maguro-park-after-itchy \
        --segmenters clipseg oneformer --indices 120 300 480 660 840
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from segmentation import make_segmenter
from taxonomy import Klass, color_lut


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def _dist(mask: np.ndarray, top: int = 4) -> str:
    u, c = np.unique(mask, return_counts=True)
    tot = mask.size
    pairs = sorted(zip(u, c), key=lambda x: -x[1])
    return " ".join(f"{Klass(int(k)).name}:{100 * n / tot:.0f}" for k, n in pairs[:top] if 100 * n / tot >= 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--segmenters", nargs="+", default=["clipseg", "oneformer"])
    ap.add_argument("--indices", nargs="+", type=int, default=[120, 300, 480, 660, 840])
    ap.add_argument("--out", default="../../output/segmodel_compare.png")
    ap.add_argument("--col-width", type=int, default=760)
    args = ap.parse_args()

    lut = np.array(color_lut(), dtype=np.uint8)
    image_dir = Path(args.images)
    srcs = {i: cv2.imread(str(image_dir / f"{i:08d}_image.jpeg")) for i in args.indices}

    # column 0 = source; then one column per segmenter
    columns: dict[str, dict[int, np.ndarray]] = {"source": srcs}
    timing: dict[str, list[float]] = {}
    for kind in args.segmenters:
        print(f"loading {kind} ...")
        t0 = time.time()
        seg = make_segmenter(kind)
        load_t = time.time() - t0
        col, times = {}, []
        for i in args.indices:
            t1 = time.time()
            m = seg.segment(srcs[i])
            dt = time.time() - t1
            times.append(dt)
            overlay = cv2.addWeighted(srcs[i], 0.45, cv2.cvtColor(lut[m], cv2.COLOR_RGB2BGR), 0.55, 0)
            col[i] = _label(overlay, f"{kind} | {_dist(m)}")
            print(f"  {kind} img {i}: {dt:.2f}s  {_dist(m)}")
        columns[kind] = col
        timing[kind] = times
        print(f"  {kind}: load {load_t:.1f}s, mean {np.mean(times):.2f}s/img")

    # assemble grid
    def fit(img, w):
        return cv2.resize(img, (w, int(w * img.shape[0] / img.shape[1])))

    rows = []
    for i in args.indices:
        cells = [fit(_label(columns["source"][i], f"src {i}") if k == "source" else columns[k][i], args.col_width)
                 for k in ["source", *args.segmenters]]
        h = min(c.shape[0] for c in cells)
        rows.append(np.hstack([c[:h] for c in cells]))
    grid = np.vstack(rows)
    out = Path(args.out)
    cv2.imwrite(str(out), grid)
    print(f"\nwrote {out}  ({grid.shape[1]}x{grid.shape[0]})")
    print("timing (s/img):", {k: round(float(np.mean(v)), 2) for k, v in timing.items()})


if __name__ == "__main__":
    main()
