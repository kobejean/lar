#!/usr/bin/env python3
"""Commercial-friendly monocular-depth bake-off, scored by held-out SfM depth accuracy.

The question this answers: *which license-clean depth model gives the most metrically
accurate depth on our park imagery?* — without running the whole BEV pipeline.

Per sampled frame we have sparse but trustworthy metric depth at the SfM keypoints. We
split those points 50/50 (deterministic per frame), fit the model's standard scale+shift
alignment on one half, and measure error on the *held-out* half. That isolates depth
quality from the alignment fit (a model can't cheat by being easy to scale). Metrics are
the usual monocular-depth ones, aggregated (median, robust) over frames:

    AbsRel   mean |pred - z| / z            (lower better)
    RMSE     sqrt(mean (pred - z)^2)  [m]    (lower better)
    delta1   frac( max(pred/z, z/pred) < 1.25 )   (higher better)

Backends (all verified commercial-friendly on their HF model card, 2026-07):

    da2-small   depth-anything/Depth-Anything-V2-Small-hf   apache-2.0   (rel. disparity)
    dpt-large   Intel/dpt-large                             apache-2.0   (rel. disparity)
    dpt-beit-l  Intel/dpt-beit-large-512                    mit          (rel. disparity)
    marigold    prs-eth/marigold-depth-v1-0                 apache-2.0   (affine-inv depth)

NOTE: Depth-Anything-V2 *Base/Large* are CC-BY-NC (non-commercial) — excluded on purpose.

Run (transformers for the DPT/DA models; diffusers for Marigold):

    uv run --extra segmentation --with diffusers --with accelerate \
        python script/depth/depth_bench.py --session maguro-park-after-itchy --frames 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "semantic_bev"))
from colmap_io import read_model                         # noqa: E402
from mono_depth import sparse_samples                     # noqa: E402  (reuse SfM sampling)

# name -> (hf/diffusers id, runner kind, output space, license)
BACKENDS = {
    "da2-small":  ("depth-anything/Depth-Anything-V2-Small-hf", "hf",       "disparity", "apache-2.0"),
    "dpt-large":  ("Intel/dpt-large",                           "hf",       "disparity", "apache-2.0"),
    "dpt-beit-l": ("Intel/dpt-beit-large-512",                  "hf",       "disparity", "mit"),
    "marigold":   ("prs-eth/marigold-depth-v1-0",               "marigold", "depth",     "apache-2.0"),
}


def load_runner(kind: str, model_id: str):
    """Return f(pil_rgb) -> HxW float32 raw prediction (disparity: high=near; depth: high=far)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "hf":
        from transformers import pipeline
        pipe = pipeline("depth-estimation", model=model_id, device=0 if dev == "cuda" else -1)

        def run(pil):
            pred = pipe(pil)["predicted_depth"]
            return pred.squeeze().float().cpu().numpy()
        return run

    if kind == "marigold":
        from diffusers import MarigoldDepthPipeline
        dt = torch.float16 if dev == "cuda" else torch.float32
        try:
            pipe = MarigoldDepthPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=dt)
        except Exception:
            pipe = MarigoldDepthPipeline.from_pretrained(model_id, torch_dtype=dt)
        pipe = pipe.to(dev)
        pipe.set_progress_bar_config(disable=True)

        def run(pil):
            out = pipe(pil, num_inference_steps=4, ensemble_size=1, output_type="np")
            return np.asarray(out.prediction).squeeze().astype(np.float32)
        return run

    raise ValueError(kind)


def robust_fit(x: np.ndarray, y: np.ndarray, trim: float = 0.1):
    """Least-squares y ≈ a·x + b with one robust trim of the worst `trim` residuals."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    ab, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = np.abs(A @ ab - y)
    keep = resid <= np.quantile(resid, 1 - trim)
    if keep.sum() >= 2:
        ab, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return float(ab[0]), float(ab[1])


# Distance bands (metres). Ground BEV cares about near/mid; far is mostly background
# trees where both mono depth and sparse SfM triangulation are unreliable.
BANDS = [("near", 0.0, 8.0), ("mid", 8.0, 20.0), ("far", 20.0, np.inf)]


def _band_metrics(pred, zt):
    out = {}
    absrel = np.abs(pred - zt) / zt
    ratio = np.maximum(pred / zt, zt / pred)
    out["absrel"] = float(np.mean(absrel))
    out["rmse"] = float(np.sqrt(np.mean((pred - zt) ** 2)))
    out["delta1"] = float(np.mean(ratio < 1.25))
    for name, lo, hi in BANDS:
        m = (zt >= lo) & (zt < hi)
        if m.sum() >= 5:
            out[f"{name}_absrel"] = float(np.mean(absrel[m]))
            out[f"{name}_delta1"] = float(np.mean(ratio[m] < 1.25))
    return out


def eval_frame(raw: np.ndarray, px, py, z, space: str, seed: int, min_test: int = 10):
    """Fit scale+shift on half the SfM points, score metric error on the held-out half.

    Metrics are reported overall and per distance band (near/mid/far) so the ranking
    reflects the near-to-mid regime that a walkable-ground BEV actually depends on.
    """
    H, W = raw.shape
    ix = np.clip(np.round(px).astype(int), 0, W - 1)
    iy = np.clip(np.round(py).astype(int), 0, H - 1)
    r = raw[iy, ix].astype(np.float64)
    z = z.astype(np.float64)
    good = np.isfinite(r) & np.isfinite(z) & (z > 0)
    r, z = r[good], z[good]
    n = len(z)
    if n < 2 * min_test:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]

    if space == "disparity":                         # 1/z ≈ a·disp + b
        a, b = robust_fit(r[tr], 1.0 / z[tr])
        denom = a * r[te] + b
        pred = np.where(denom > 1e-6, 1.0 / denom, np.nan)
    else:                                            # depth: z ≈ a·d + b
        a, b = robust_fit(r[tr], z[tr])
        pred = a * r[te] + b

    zt = z[te]
    ok = np.isfinite(pred) & (pred > 0)
    if ok.sum() < min_test:
        return None
    return _band_metrics(pred[ok], zt[ok])


def colorize(raw, space):
    v = raw.astype(np.float32)
    lo, hi = np.percentile(v, [2, 98])
    n = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    if space == "depth":                              # show near=warm for both
        n = 1.0 - n
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None)
    ap.add_argument("--model", default=None, help="COLMAP text model with tracks (poses_txt)")
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backends", default=",".join(BACKENDS), help="comma list from " + ", ".join(BACKENDS))
    ap.add_argument("--frames", type=int, default=40, help="evenly-spaced frames to sample")
    ap.add_argument("--min-pts", type=int, default=60, help="min SfM points a frame needs to be used")
    ap.add_argument("--data-factor", type=int, default=2)
    args = ap.parse_args()

    if args.session:
        sys.path.insert(0, str(_ROOT))
        from lar_session import Session
        s = Session(args.session)
        # Mono alignment needs SfM *tracks*: use the raw COLMAP model, not the refined one.
        args.model = args.model or str(s.colmap_model)
        args.images = args.images or str(s.images)
        args.out = args.out or str(s.root / "output" / f"{s.name}-depthbench")
    if not (args.model and args.images and args.out):
        ap.error("need --session or explicit --model/--images/--out")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            ap.error(f"unknown backend {b!r}; choose from {list(BACKENDS)}")

    print(f"reading tracked model {args.model} ...")
    recon = read_model(args.model)
    images_dir = Path(args.images)
    f = args.data_factor

    # Sample frames evenly across the trajectory, keeping only well-observed ones.
    ordered = sorted(recon.images.values(), key=lambda im: im.name)
    picks, seen = [], 0
    for im in ordered[:: max(1, len(ordered) // (args.frames * 3))]:
        cam = recon.cameras[im.camera_id]
        sc = (cam.width // f) / cam.width
        px, py, z = sparse_samples(recon, im, sc)
        if len(z) >= args.min_pts:
            picks.append((im, px, py, z))
            if len(picks) >= args.frames:
                break
    print(f"using {len(picks)} frames (>= {args.min_pts} SfM pts each), data-factor {f}")
    if len(picks) < 3:
        raise SystemExit("too few usable frames; lower --min-pts or --data-factor")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Cache resized RGB once (shared across backends).
    frames = []
    for im, px, py, z in picks:
        bgr = cv2.imread(str(images_dir / im.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[0] // f, bgr.shape[1] // f
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)
        frames.append((im, rgb, px, py, z))

    from PIL import Image as PILImage
    results = {}
    sample_tiles = []
    for b in backends:
        model_id, kind, space, lic = BACKENDS[b]
        print(f"\n=== {b}  ({model_id}, {lic}) ===")
        run = load_runner(kind, model_id)
        per, t0, nt = [], time.time(), 0
        for i, (im, rgb, px, py, z) in enumerate(frames):
            raw = run(PILImage.fromarray(rgb))
            H, W = rgb.shape[:2]
            if raw.shape != (H, W):
                raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
            m = eval_frame(raw, px, py, z, space, seed=1000 + i)
            if m:
                per.append(m)
            nt += 1
            if i == 0:  # one sample depth preview per backend
                tile = colorize(raw, space)
                cv2.putText(tile, b, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                sample_tiles.append(tile)
        dt = (time.time() - t0) / max(nt, 1)
        if not per:
            print("  no usable frames"); continue
        # Median across frames for every metric a frame reported (bands may be absent).
        keys = set().union(*(p.keys() for p in per))
        agg = {k: float(np.median([p[k] for p in per if k in p])) for k in keys}
        agg["sec_per_frame"] = round(dt, 3)
        agg["frames"] = len(per)
        agg["license"] = lic
        results[b] = agg
        print(f"  near AbsRel {agg.get('near_absrel', float('nan')):.3f} "
              f"(d1 {agg.get('near_delta1', float('nan')):.2f}) | "
              f"mid {agg.get('mid_absrel', float('nan')):.3f} | "
              f"overall {agg['absrel']:.3f} | {dt:.2f}s/frame")

    if not results:
        raise SystemExit("no results")

    # Rank by near-field AbsRel (the regime a walkable-ground BEV depends on); fall back
    # to overall for any backend that never had enough near points.
    def rank_key(b):
        return results[b].get("near_absrel", results[b]["absrel"])
    order = sorted(results, key=rank_key)
    cols = ["license", "near_absrel", "near_delta1", "mid_absrel",
            "far_absrel", "absrel", "sec_per_frame", "frames"]
    hdr = ["backend"] + [c.replace("absrel", "AbsRel").replace("delta1", "d1") for c in cols]
    md = ["# Depth bake-off — held-out SfM depth accuracy",
          f"\nSession `{Path(args.model).parent.parent.name}`, {len(frames)} frames, "
          f"data-factor {f}. Ranked by **near-field** AbsRel (z<8 m; lower = better). "
          f"Bands: near <8 m, mid 8–20 m, far >20 m.\n",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for b in order:
        r = results[b]
        cells = [b] + [(f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "—")))
                       for c in cols]
        md.append("| " + " | ".join(cells) + " |")
    (out / "scorecard.md").write_text("\n".join(md) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    if sample_tiles:
        h = max(t.shape[0] for t in sample_tiles)
        sample_tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=0)
                        for t in sample_tiles]
        cv2.imwrite(str(out / "sample_depths.png"), np.hstack(sample_tiles))

    print(f"\nRanking by AbsRel: {' > '.join(order)}")
    print(f"scorecard -> {out}/scorecard.md  (+ results.json, sample_depths.png)")


if __name__ == "__main__":
    main()
