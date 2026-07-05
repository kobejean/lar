# semantic_bev — cross-sectional semantic ground model

Offline pipeline that turns a COLMAP reconstruction + park imagery into a **2.5D semantic
ground model**: a top-down, georeferenceable grid of ground height + surface class +
occupancy. This is the front-end for both a 2D semantic occupancy grid and, later, IMDF
export — and the semantic labels are intended to feed back into localization.

## Design (why it's shaped this way)

- **Semantics defines the ground, not geometry heuristics.** We don't RANSAC-guess which
  points are ground; we segment `path/grass/terrain/...` and the ground surface is simply
  the geometry carrying those labels. Removes the brittle geometric ground-finding.
- **Geometry and semantics have different resolutions.** Height comes from sparse points
  (~25–50 cm ceiling); class *boundaries* come from dense per-pixel image masks. Two
  independent sources, each as good as itself.
- **Pluggable geometry source.** The grid builder only consumes
  `(positions, labels, confidence)`. COLMAP sparse points today; 2DGS/MVS surfels later
  behind the same interface — no downstream change.
- **Pluggable segmenter.** `Segmenter` interface: heuristic (no deps, plumbing only) vs.
  open-vocab CLIPSeg (real). Grounded-SAM-2 can slot in later.
- **Levels, IMDF-style.** A `GroundModel` is a list of single-valued 2D `Level`s (one for
  now). Slopes are a smooth per-cell height field; bridges/overpasses later = another
  `Level`. No multi-surface-per-cell complexity.

## Modules

| file | role |
|------|------|
| `colmap_io.py`   | read COLMAP text model; tracks give exact observed pixel per point |
| `taxonomy.py`    | **the semantic contract**: prompts → internal class → IMDF category |
| `segmentation.py`| `Segmenter` interface + `Heuristic`/`ClipSeg`/`OneFormer` backends |
| `labeling.py`    | cache masks, vote track pixels → per-point class (no reprojection) |
| `ground_model.py`| `GroundModel`/`Level`, rasterize labelled points, export npz+PNG |
| `dense_projection.py` | dense-mask projection: semantic raster from all mask pixels (CPU fallback) |
| `imdf_export.py` | vectorise BEV -> schema-valid IMDF archive (WGS84) + self-validator |
| `pipeline.py`    | end-to-end driver + CLI (auto up-axis/sign detection) |
| `compare_segmenters.py` | run backends side-by-side on sample images → comparison grid |
| `verify_orientation.py` | overlay camera trajectory on the BEV (orientation sanity) |

## Semantic raster modes (`--semantic-mode`)

- `vote` (default) — sparse point votes: fast, but speckly and low-coverage (limited by point density).
- `project` — **dense-mask projection**: project every cell centre into all views, read the
  cached masks, keep only ground-class votes (occlusion-robust), inverse-depth weighted. Much
  cleaner + higher coverage, CPU-only. This is the fallback for when semantic 3DGS is too heavy
  (low-VRAM / on-device). Height + occupancy still come from the point path.

## Segmentation backends (all commercial-license friendly)

| backend | model | license | speed | notes |
|---------|-------|---------|-------|-------|
| `heuristic` | HSV colour rules | — | instant | plumbing only, not real |
| `clipseg`   | CIDAS/clipseg-rd64-refined | Apache-2.0 | ~2.3 s/img | open-vocab, custom prompts, coarse edges |
| `oneformer` | oneformer_ade20k_swin_tiny | MIT | ~0.25 s/img | crisp edges, fixed ADE20k vocab |
| `oneformer-large` | oneformer_ade20k_swin_large | MIT | ~0.42 s/img | marginally better grass; dominated on speed |
| `mask2former-large` | mask2former-swin-large-ade-semantic | **MIT** | **~0.12 s/img** | **recommended** — top quality + fastest |

Benched tiny vs large on sample park images: quality is close across all three closed-set
models (large is marginally better at grass/undergrowth). The deciding factor is speed, and
**`mask2former-large` wins** — best-tier quality *and* ~2× faster than OneFormer-tiny (no
text-task branch). ADE20k's 150 classes are remapped to our taxonomy by
`taxonomy.keyword_klass`. Keep `clipseg` for custom open-vocab classes ADE20k lacks.
(SegFormer excluded — NVlabs weights are non-commercial. Grounded-SAM-2 (Apache-2.0) is a
future add for discrete-object precision, benches/poles → IMDF amenities.)

## Run

Heuristic backend (no GPU, validates plumbing/geometry):

```sh
cd script/semantic_bev
uv run python pipeline.py \
  --model  ../../input/<session>/colmap/poses_txt \
  --images ../../input/<session> \
  --out    ../../output/<session>-sbev \
  --segmenter heuristic --cell-size 0.5
```

Real open-vocab backend (CLIPSeg; needs the `segmentation` extra):

```sh
uv sync --extra segmentation          # one-time: installs torch + transformers
uv run python pipeline.py ... --segmenter clipseg --clip-threshold 0.3
```

Add `--limit N` to process only the first N images while iterating.

### Geometry source: COLMAP points (default) or semantic 3DGS

`--source` selects where `(positions, labels, confidence)` come from. The default
`colmap` path is above. `--source gsplat` instead consumes a trained **semantic 3DGS**
export ([`../gsplat`](../gsplat/) `train.py --semantic`): every Gaussian already carries a
class, so this skips segmentation/voting entirely and feeds a much denser, pre-labelled
cloud straight into the *same* `build_level`. `--model` is still used (cameras only) to
detect gravity/up, unless you pass `--up-axis`/`--up-sign`.

```sh
uv run python pipeline.py --source gsplat \
  --gsplat-dir ../../output/<session>-gsplat-sem \
  --model      ../../output/<session>-refined/colmap/sparse/0 \
  --out        ../../output/<session>-sbev-gsplat \
  --cell-size 0.5 --min-opacity 0.1
```

`--min-opacity` drops faint Gaussians (3DGS floaters) before rasterising. This is the
optional "3DGS geometry source" tier: denser height/occupancy than sparse COLMAP tracks,
at the cost of first training a semantic splat model.

## Output (`<out>/`)

- `level0.npz` — `height` (m, nan=unobserved), `semantic` (Klass id), `occupancy`
  (free/blocked/unknown), `coverage` (observed vs interpolated)
- `level0.meta.json` — grid spec (cell size, origin, up-axis/sign, dims)
- `level0_{height,semantic,occupancy}.png` — previews (north-up)
- `masks/` — cached per-image class-id PNGs (+ colour previews); segmentation runs once

## Status / next

- [x] end-to-end chain validated on `maguro-park-after-itchy` (970 imgs, 429k pts)
- [x] park-scale semantic BEV with visible path/grass structure (heuristic backend)
- [ ] CLIPSeg open-vocab pass (real semantics)
- [ ] dense-mask semantic projection (crisp class boundaries, not point-limited)
- [ ] per-cell height outlier rejection (SfM floaters)
- [ ] metric/gravity frame confirmation before trusting absolute heights
- [ ] 2DGS geometry source; multi-level; IMDF export; localization feedback
