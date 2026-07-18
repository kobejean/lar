# backbone — LingBot-Vision heads for the semantic-BEV front-end

Frozen [LingBot-Vision](https://github.com/robbyant/lingbot-vision) ViT (Apache-2.0 code +
weights) as a **unified feature source** for the two things the BEV front-end needs per frame:
dense semantics and a ground/footprint model. Cheap heads on the same patch grid replace the
two separate off-the-shelf models (DA2 depth + Mask2Former seg) with one backbone.

| file | role |
|------|------|
| `probe.py`            | is the frozen backbone worth building on? `pca` (feature viz) + `linprobe` (linear head vs Mask2Former pseudo-labels). Verdict: yes — ViT-S/16 hits 0.87 patch-acc on our taxonomy. |
| `footprint_labels.py` | **generate footprint / free-space supervision** (below) for a ground head, no manual labels. |

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
# full run
uv run python script/backbone/footprint_labels.py --session maguro-park-after-itchy
```

Key knobs: `--cell-size` (DEM), `--size` (render res), `--max-range`, `--clearance-lo/-hi`
(occluder band), `--min-count`/`--solid-gap` (footprint solidity), `--occ-margin`,
`--splat-radius`.

### Status / next

- [x] geometry-only targets validated on `maguro-park-after-itchy` (open ground → visible,
  trunks/walls → footprint, distant occluded ground → hidden; metric depth sane 2–30 m)
- [ ] **FOOTPRINT-SEMANTICS**: vote each obstacle point's Mask2Former class (the `MaskStore`
  cache) → footprint cells carry a class id ("occupied areas *with semantic labels*")
- [ ] denser occluders for a stronger hidden-ground signal (extruded footprint columns, or
  mono-depth back-projection) — sparse COLMAP gives a speckly forest-floor footprint
- [ ] train the head on frozen LingBot patch features (semantic head + this ground head)
