# backbone — LingBot-Vision heads for the semantic-BEV front-end

Frozen [LingBot-Vision](https://github.com/robbyant/lingbot-vision) ViT (Apache-2.0 code +
weights) as a **unified feature source** for the two things the BEV front-end needs per frame:
dense semantics and a ground/footprint model. Cheap heads on the same patch grid replace the
two separate off-the-shelf models (DA2 depth + Mask2Former seg) with one backbone.

| file | role |
|------|------|
| `probe.py`            | is the frozen backbone worth building on? `pca` (feature viz) + `linprobe` (linear head vs Mask2Former pseudo-labels). Verdict: yes — ViT-S/16 hits 0.87 patch-acc on our taxonomy. |
| `footprint_labels.py` | **generate footprint / free-space supervision** (below) for a ground head, no manual labels. |
| `base_points.py`      | **deduped discrete object footprints** — open-vocab detection → foot points → BEV clustering (below). |

## `footprint_labels.py` — footprint / free-space supervision generator

Produces the per-frame training targets a ground head needs à la Niantic's *Footprints and
Free Space from a Single Color Image* (CVPR 2020) — the one signal plain segmentation can't
give: **where the ground continues behind objects**, and **which ground cells an object's
footprint occupies**. That's what tells the BEV projector which pixels map onto the ground
plane (incl. hidden ground) and which map onto occupied area.

**Why it's cleaner than Niantic's method.** They had no global model, so they manufactured
hidden-ground labels by projecting the ground seen in *neighbouring* frames into the target
view. We already reconstruct a gravity-aligned **global ground DEM**
(`semantic_bev/geometry.from_reconstruction`). Rendering that DEM into a target camera yields
the full ground extent — visible *and* occluded — in one shot, less noisy than pairwise
reprojection. The neighbour-frame trick is subsumed by "render the global DEM".

**How.** The DEM is rasterised as a triangle mesh into each posed camera (z-buffered → dense
metric ground-depth). Body-height band points splat in as occluders (visible vs hidden split).
Footprint cells reuse the **validated occupancy rule** from `ground_model.build_level`:
solid-to-ground obstacles only (≥ `min_count` points whose low height-above-ground quantile
≤ `solid_gap`), so canopy overhanging an open path stays walkable ground rather than blocking
it.

### Outputs (`output/<session>-footprint/`)

Per frame, at a downscaled render resolution:
- `<stem>.png` — uint8 class map: `0` non-ground · `1` visible-ground · `2` hidden-ground ·
  `3` footprint (occupied) · `4` object (above-horizon structure)
- `<stem>_depth.npy` — float32 metric depth to the DEM ground (valid on the ground extent) —
  the "depth to (hidden) ground" regression target
- `<stem>_cover.npy` — float32 {0,1}: ground cell DEM-observed vs extrapolated → per-pixel loss weight
- `<stem>_preview.png` — RGB × colourised labels (eyeball check)
- `meta.json` — class legend + params

### Run

```sh
# from repo root; --sample spreads N frames across the whole capture for QA
uv run python script/backbone/footprint_labels.py --session maguro-park-after-itchy --sample 10
# denser ground surface from fused mono depth (needs script/depth/mono_depth.py output)
uv run python script/backbone/footprint_labels.py --session maguro-park-after-itchy --dem-source mono
# full run
uv run python script/backbone/footprint_labels.py --session maguro-park-after-itchy
```

**`--dem-source {colmap,mono}`** — where the ground *surface* comes from. `colmap` (default)
uses the sparse SfM points; `mono` back-projects the per-view mono depth
(`output/<session>-depth-<mono>/`, from `script/depth/mono_depth.py`) into a dense cloud for
the DEM. **Footprint/occupancy always stays on the accurate COLMAP obstacle cloud** (the
depth-bench hybrid: mono's vertical noise over-blocks). On `maguro-park-after-itchy`, `mono`
lifts DEM coverage 20%→43% and *trusted* depth-target coverage 75%→98% (near-zero extrapolated
fill under the loss) — for ~33 s reusing precomputed maps.

Key knobs: `--depth-dir/--depth-stride/--depth-voxel` (mono source), `--cell-size` (DEM),
`--size` (render res), `--max-range`, `--clearance-lo/-hi` (occluder band),
`--min-count`/`--solid-gap` (footprint solidity), `--occ-margin`, `--splat-radius`.

## `base_points.py` — deduped discrete object footprints

Semantic masks smear: the same tree, projected from 970 frames, becomes one un-separable BEV
blob. The multi-view fix is the **foot point** — reduce each *detected instance* to its
ground-contact (bottom-centre of the box), project to the ground, and **cluster across frames**.
Points from one object collapse to a single landmark; a stray frame is an outlier. Instance
detection (not semantic seg) is what makes objects separable to begin with.

Per sampled frame: **Grounding DINO** (Apache-2.0, open-vocab, via `transformers`) detects the
`--prompt` classes → foot point per box → **DEM ground depth** at that pixel (the trusted
surface, reusing this dir's DEM render — not mono depth at the noisy object edge) → world point.
Across frames: gravity-align → **DBSCAN per class** → one labelled footprint per object.

Outputs `output/<session>-basepoints/`: `bev_footprints.png` (top-down map, camera trajectory +
class-coloured markers sized by detection count), `objects.json` (`{class,u,v,count,score}` per
object), `detections/` (per-frame box + foot-point overlays for QA).

```sh
uv run --extra segmentation python script/backbone/base_points.py \
  --session maguro-park-after-itchy --dem-source mono --sample 250
```

On `maguro-park-after-itchy` (250 frames): 1364 foot points → **109 objects** — 47 tree, 22
bush, 12 bench, 11 pole, 8 sign, 5 trash-can, 4 rock; trees line the walked paths. Drops into
IMDF `amenity.landmark` (trees/benches) / `unit.structure`. Knobs: `--prompt`, `--box-thr/
--text-thr` (detector), `--eps/--min-samples` (cluster radius / min detections per object).

### Status / next

- [x] `base_points.py`: open-vocab foot-point → BEV clustering → 109 deduped labelled objects
  on the park (the "occupied areas *with semantic labels*" ask, as discrete landmarks)
- [ ] **lines** for extended objects (walls/hedges): SAM2/Grounded-SAM-2 mask → bottom contour
  → ground polyline (vs a single point)

- [x] geometry-only targets validated on `maguro-park-after-itchy` (open ground → visible,
  trunks/walls → footprint, distant occluded ground → hidden; metric depth sane 2–30 m)
- [x] `--dem-source mono` hybrid: dense mono DEM + COLMAP footprint → trusted depth-target
  coverage 75%→98% (reconstruction poses are fine; the ground was just sparsely sampled)
- [ ] **FOOTPRINT-SEMANTICS**: vote each obstacle point's Mask2Former class (the `MaskStore`
  cache) → footprint cells carry a class id ("occupied areas *with semantic labels*")
- [ ] denser *footprint*: 2DGS surfels (dense **and** surface-accurate — fixes the speckly
  forest-floor footprint that sparse COLMAP still leaves, unlike mono which over-blocks)
- [ ] train the head on frozen LingBot patch features (semantic head + this ground head)
