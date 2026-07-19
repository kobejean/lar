# script/depth — monocular metric depth (commercial-friendly)

Per-view metric depth for the semantic-BEV depth bake-off. Every backend writes the same
contract consumed by `../semantic_bev/depth_backproject.py`:

    <out>/<image_stem>.npy        float32 (H, W) metric z-depth (m); <=0 / NaN = invalid
    <out>/<image_stem>.conf.npy   optional float32 (H, W) confidence in [0, 1]

## Files

| file | role |
|------|------|
| `mono_depth.py`  | export dense metric depth for **all** frames (mono disparity → scale+shift fit to SfM) |
| `depth_bench.py` | **rank depth models** by held-out SfM depth accuracy (no full pipeline) |

## Commercial-friendly backends (licenses verified on the HF model card, 2026-07)

| key | model | license | type | speed* |
|-----|-------|---------|------|--------|
| `da2-small`  | depth-anything/Depth-Anything-V2-Small-hf | **Apache-2.0** | rel. disparity | 0.10 s |
| `dpt-large`  | Intel/dpt-large                           | **Apache-2.0** | rel. disparity | 0.22 s |
| `dpt-beit-l` | Intel/dpt-beit-large-512                  | **MIT**        | rel. disparity | 0.59 s |
| `marigold`   | prs-eth/marigold-depth-v1-0               | **Apache-2.0** | affine-inv depth | 1.61 s |

\* per 960×720 frame on the local RTX 5060.

> ⚠️ **Depth-Anything-V2 Base/Large are CC-BY-NC-4.0 (non-commercial)** — excluded. Only the
> *Small* checkpoint is Apache-2.0. `mono_depth.py` defaults to it for that reason.

## Rank the models (held-out SfM accuracy)

Fits each model's scale+shift on half a frame's SfM points and scores metric error on the
held-out half, aggregated over sampled frames. Metrics are stratified by distance because a
walkable-ground BEV only depends on the **near/mid** regime (far = background trees, where
mono depth *and* sparse SfM are both unreliable).

```sh
uv run --extra segmentation --with diffusers --with accelerate \
  python script/depth/depth_bench.py --session maguro-park-after-itchy --frames 40
```

Writes `output/<name>-depthbench/`: `scorecard.md`, `results.json`, `sample_depths.png`.
(`diffusers`/`accelerate` only needed if `marigold` is in `--backends`.)

### Result on `maguro-park-after-itchy` (40 frames)

Ranked by near-field (z<8 m) accuracy. δ1 = frac within 25 %; more robust than AbsRel,
which is mean-of-ratio and tail-heavy on near foreground edges.

| backend | license | near δ1 | near AbsRel | mid AbsRel | overall AbsRel | speed |
|---------|---------|---------|-------------|------------|----------------|-------|
| dpt-beit-l | MIT | **0.930** | 0.304 | 0.198 | **0.280** | 0.59 s |
| da2-small | Apache-2.0 | 0.927 | 0.335 | 0.201 | 0.313 | **0.10 s** |
| dpt-large | Apache-2.0 | 0.910 | 0.354 | 0.214 | 0.323 | 0.22 s |
| marigold | Apache-2.0 | 0.680 | 0.390 | 0.206 | 0.319 | 1.61 s |

**Takeaways.** (1) All models are best in the **mid-field** (~0.20 AbsRel) and tail-heavy
near-field. (2) `dpt-beit-l` is marginally most accurate but 6× slower; `da2-small` is
statistically tied on near-δ1, Apache-2.0, and fastest → the practical default. (3) Marigold
in its fast 4-step config is worst here and slowest — not worth it. (4) Model choice matters
less than the depth regime; none give better than ~20 % mid-field AbsRel on this park, so
per-frame depth should be **fused across many views** (the back-projection harness does this)
rather than trusted per-frame.

## Export depth for the winner (feeds the BEV pipeline)

```sh
uv run --extra segmentation python script/depth/mono_depth.py \
  --session maguro-park-after-itchy --data-factor 2                 # da2-small (default)
# or a different backend:
#   --model-name Intel/dpt-beit-large-512
```

Then run the BEV pipeline with `--source depth` and score with `eval_depth_bev.py`
(see `../semantic_bev/README.md`).
