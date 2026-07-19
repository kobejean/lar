# backbone — LingBot-Vision heads for the semantic-BEV front-end

Frozen [LingBot-Vision](https://github.com/robbyant/lingbot-vision) ViT (Apache-2.0 code +
weights) as a **unified feature source** for the two things the BEV front-end needs per frame:
dense semantics and a ground/footprint model. Cheap heads on the same patch grid replace the
two separate off-the-shelf models (DA2 depth + Mask2Former seg) with one backbone.

| file | role |
|------|------|
| `probe.py`            | is the frozen backbone worth building on? `pca` (feature viz) + `linprobe` (linear head vs Mask2Former pseudo-labels). Verdict: yes — ViT-S/16 hits 0.87 patch-acc on our taxonomy. **Caveat: that 0.87 is agreement with the teacher, not accuracy — see `ade20k_eval.py`.** |
| `footprint_labels.py` | **generate footprint / free-space supervision** (below) for a ground head, no manual labels. |
| `base_points.py`      | **deduped discrete object footprints** — open-vocab detection → foot points → BEV clustering (below). |
| `ade20k_eval.py`      | **score the frozen backbone against real ADE20K ground truth** — semantic linear probe + instance separability (below). |
| `relabel.py`          | **browser BEV relabeler** — hand-correct `footprint2d` rasters + place semantic ground-contact keypoints (below). |

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

### `--backend {moondream,gemini}` — pointing-VLM front-end

Same points pipeline, different instance detector. A **pointing VLM** emits *one 2D point per
instance* directly from a class name — no boxes. Because the VLM points at the object body
(trunk/seat), not its base, we **drop straight down the image column to the first ground
(DEM-finite) pixel** = the object's ground contact directly below. That contact feeds the
*identical* `world → DBSCAN` clustering as the box path, so the backends are directly comparable.

- **`gemini`** (cloud oracle) — Gemini's trained pointing via `google-genai`; needs
  `GEMINI_API_KEY` in the env. Default `gemini-flash-lite-latest` (free-tier accessible). One
  multi-class call per frame returns `{"point":[y,x],"label"}` at the ground contact.
- **`moondream`** (local, Apache-2.0) — Moondream2's native `.point()`. **Currently blocked**:
  its HF remote code is incompatible with transformers ≥ 5 (`all_tied_weights_keys`); needs a
  transformers-4 env or the standalone `moondream` package. Weights download fine.

```sh
# cloud oracle (quality ceiling); GEMINI_API_KEY must be set, never commit it
GEMINI_API_KEY=... uv run --extra segmentation --with google-genai python \
  script/backbone/base_points.py --session maguro-park-after-itchy --backend gemini --sample 40
```

**Gemini vs Grounding DINO, apples-to-apples (40 frames, `--dem-source mono --min-samples 2`):**
DINO 231 pts → **29 objects**; Gemini 828 pts → **115 objects** (77 tree, 16 bush, 9 rock, 5
pole, 5 bench, 2 sign, 1 trash can). ~4× the recall — Gemini catches background trees, saplings,
and occluded contacts a tiny box detector misses, and its points land on the ground contact
without a prompt trick. Caveats: cloud API (not shippable offline; ~1 paid call/frame, free-tier
daily caps), and `drop_to_ground` can bias a distant object's contact slightly toward the camera
(the first ground pixel below the trunk). Natural next step: use Gemini as an **oracle to distill
labels** into a shippable local head.

## Depth contact gate — "touches the ground in 2D" ≠ "touches it in 3D"

Both footprint paths reduce an object to its **bottom-most pixel** and ray-DEM that to the
ground. `footprint_instances`' *canopy guard* checks the pixels below are `Role.GROUND`, but
that is a **2D** test and passes in exactly the case it needs to catch: a signboard panel or
bench seat silhouetted against open lawn. The pixels below genuinely are ground — just ground
ten metres behind — so the contact lands metres past the object.

`--contact-depth-tol` (default `0.15`, `0` disables) settles it geometrically: if the object
really stands there, its measured mono depth equals the range at which that ray meets the DEM;
if it floats, it is much *nearer* than the ground its ray hits.

Compared against the **DEM hit**, not the neighbouring ground pixel, on purpose — mono depth
smooths across object boundaries, so a boundary-crossing comparison washes out the very jump
it looks for, and spends two noisy samples instead of one. And mono depth only ever *vetoes*
a contact; placement stays ray-DEM. A relative comparison at one pixel is what mono depth is
reliable for; metric placement at an object edge is what it is not.

**It is measured against the frame's median depth ratio, not against 1.0.** `MonoDepth.metric`
fits scale+shift per frame from that frame's SfM points, and a poor fit skews the whole map by
a constant. On maguro-park frame 0 the fit is off 2.3× (d_mono median 1.31 m vs z_dem 3.04 m):
an absolute test keeps **4 of 960** contacts and silently deletes the frame; the median-relative
test keeps **693**. Healthy frames sit at 0.91–1.02, so normalising costs them nothing. Same
trick as `base_points.py`'s relative ground datum — absorb the systematic bias, test the
outlier. Frames whose median lands outside 0.7–1.4 are flagged `<< depth scale suspect` rather
than quietly normalised, since that indicates a depth fit worth fixing at the source.

Measured on `maguro-park-after-itchy` (40 frames, `--dem-source mono`):

| | contacts | FOOTPRINT cells |
|---|---|---|
| gate off | 24108 | 3.0% |
| gate on (`0.15`) | **18664** (−23%) | 2.2% |

Debug overlays mark kept contacts **red** and gate-rejected ones **magenta** — on frame 3 the
magenta traces the underside of a signboard panel while red sits on its two post bases and the
tree trunk. Eyeballing those is how you tune the tolerance.

Caveat: normalising assumes most bottom-most pixels in a frame are genuine contacts (true here
— ground is everywhere, floating silhouettes are the minority). A frame of nothing but floating
objects would normalise to its own wrong consensus.

## `ade20k_eval.py` — the frozen backbone vs real ground truth

`probe.py --task linprobe` trains **and** scores against Mask2Former output, so its 0.87 is
*agreement with the teacher*, silently capped by the teacher's own errors. ADE20K supplies
real labels in a matching label space (the cached Mask2Former is ADE-trained), so the two
can finally be separated.

Data — no registration, ~1 GB + 90 MB, extracted to `/home/play/Code/datasets/ade20k`:

```sh
curl -fLO http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip   # 20210 train / 2000 val
curl -fLO http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
```

```sh
uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
  python script/backbone/ade20k_eval.py semantic --num 120 --val-num 60
uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
  python script/backbone/ade20k_eval.py instance --num 40
```

**Measured (LingBot ViT-S/16, 512², 120 train / 60 val images):**

| | mIoU | pixel-acc |
|---|---|---|
| LingBot-small + **linear** head | **57.3** | 82.8 |
| Mask2Former-Swin-**L** (the `probe.py` teacher) | 65.6 | 88.9 |

A *linear* probe on a frozen ViT-S lands **8.4 mIoU** behind a fully-supervised Swin-Large —
good for the compute, and it reframes the 0.87: the teacher probe.py measured against is
itself only 65.6 mIoU vs GT. Per class, LingBot wins **FURNITURE 24.6 vs 9.0** — the class
holding our park objects (bench, pole, sign, trash can, streetlight). Distilling the teacher
would make the classes we care about *worse*, not better.

Caveats, stated plainly: teacher predictions are mapped to our taxonomy through
`taxonomy.keyword_klass` while GT uses this file's explicit table, so part of the 8.4 gap is
mapping mismatch, not model quality — the gap is an **upper bound** on the teacher's real
advantage. GRASS scores badly for both (7.8 / 9.2), which smells like a label-mapping
artifact (grass/field/flower folded together) rather than model failure.

**Instance separability** (`instance`, 40 val images) asks the question the BEV pipeline
chokes on — "the same tree from 970 frames becomes one un-separable blob". It scores how well
cosine similarity ranks same-instance patch pairs above different-instance-same-class pairs:

```
same-instance      mean cos 0.655   |   diff-instance same-class  mean cos 0.484
ROC AUC 0.734  -> instance identity is weakly present; a trained head may separate instances
```

So the frozen features do carry instance identity beyond semantics — enough to be worth a
head, not enough to expect it for free.

### ⚠ `taxonomy.keyword_klass` mislabels real classes

The mapper does unanchored substring matching, which produces genuine errors:

| ADE class | maps to | because |
|---|---|---|
| `seat` | WATER | contains "sea" |
| `streetlight` | TREE | contains "s**tree**t" |
| `carpet` | VEHICLE | contains "car" |
| `kitchen island` | TERRAIN | contains "**land**" |
| `skyscraper` | SKY | prefix match |
| `pool table` | WATER | contains "pool" |

This is **live**: `footprint2d.py`'s `SegKlass` builds its LUT the same way, and
`maguro-park-after-itchy`'s structure raster carries **9 WATER cells in a park with no
water**. `ade20k_eval.py` therefore uses a hand-written `_ADE_KLASS` table and offers
`--mapping keyword` to reproduce the buggy behaviour for comparison.

Note that **mIoU is the wrong instrument** for this bug — it barely moves (57.3 → 56.2),
because when GT and predictions share the same wrong LUT the head just learns the wrong label
consistently. The damage shows in the class populations: FURNITURE training patches collapse
**3031 → 815** (−73%, as bench/pole/sign/streetlight scatter) while WATER inflates
**1842 → 3776** (2×, absorbing seats and pool tables). Downstream consumers eat that, which is
exactly what the 9 phantom WATER cells are.

## `relabel.py` — BEV relabeler (hand-correct + keypoints)

`footprint2d.py` gets the broad BEV structure right and the *edges* wrong: footprint
boundaries bleed where the bottom-most obstacle pixel is a shadow or a leaf, HIDDEN
over/under-claims, thin structures come out speckled. Fixing that per-frame would mean
painting 970 images; **fixing it in BEV means painting once** — the park is a single
201×230 grid — and every frame that sees a cell inherits the correction when the raster is
rendered back into that camera. That asymmetry is why the editor works in BEV.

```sh
uv run python script/backbone/relabel.py \
  --npz output/maguro-park-after-itchy-footprint2d-mono2/footprint2d.npz
```

Opens a local browser editor (stdlib http.server, no new deps). DEM hillshade underneath for
terrain context — uncovered cells render near-black, so extrapolated guesswork is visibly
distinct from observed ground.

- **Ground** tool — paint FREE / FOOTPRINT / HIDDEN / UNKNOWN
- **Class** tool — assign a `Klass` to footprint cells (fixes the structure raster)
- **Keypoint** tool — drop semantic **ground-contact points**: the signal `base_points.py`
  triangulates for. Hand-placed points are both the ground truth to *score* that pipeline
  against and the target for a contact-point head on frozen LingBot features.
- wheel zoom, space/middle-drag pan, `1`/`2`/`3` tools, `[`/`]` brush, `z` undo,
  "Highlight my edits" to see the diff vs the original

Writes `relabel.npz` (corrected `state` + `structure`, with `state_orig`/`structure_orig`
kept for diffing) and `keypoints.json` next to the input. The original npz is never
overwritten and re-running resumes.

**Grid registration caveat:** `footprint2d`'s npz stores `cell_size` but not the DEM origin,
so edits are registered to *that grid*, not world coordinates. Consumers must rebuild the
GroundField with the same session/cell-size/dem-source (deterministic) and apply edits
cell-wise. Adding `origin_u`/`origin_v` to `footprint2d`'s `np.savez` would make this
self-describing, and let keypoints carry true world (u, v).

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
