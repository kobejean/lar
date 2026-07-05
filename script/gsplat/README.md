# gsplat — 3D Gaussian Splatting from LAR/COLMAP reconstructions

Trains a **3D Gaussian Splatting** model (optionally **semantic**) directly from the
COLMAP reconstruction the LAR pipeline already produces. Built for eventual **park
scale** on modest hardware: the coarse-but-complete model is the goal, not
photo-perfect detail.

## Why gsplat + MCMC (the scale story)

- **Engine: [gsplat](https://github.com/nerfstudio-project/gsplat)** (Apache-2.0 — matches
  this project's commercial-license discipline; supports Blackwell GPUs).
- **MCMC strategy with `--cap-max`** puts a *hard ceiling* on the number of Gaussians.
  VRAM is dominated by Gaussian count, so the cap — not the scene size — sets the memory
  budget. A park just gets a coarser model at the same cap. This is what makes an 8 GB
  laptop GPU viable for a ~80 × 80 m capture.
- Combine with **`--data-factor`** (image downscale) and a low **`--sh-degree`** to trade
  detail for coverage.

## Semantic 3DGS

With `--semantic`, each Gaussian additionally carries a **per-class logit vector**. We
render those logits (alpha-composited, view-consistently) and cross-entropy them against
2D masks from a segmenter — i.e. we **distil the 2D segmenter into the 3D Gaussians**.
The segmenters, class taxonomy, and colours are reused verbatim from
[`../semantic_bev`](../semantic_bev/), so there is one semantic contract across both
pipelines. Output per Gaussian: an argmax class id (`_labels.npy`) + a taxonomy-coloured
`.ply`. That labelled point set is exactly the `(positions, labels, confidence)` geometry
source `semantic_bev` is designed to consume, and a route into LAR localization.

**Two-phase training (the reliable, canonical recipe).** Semantics is *not* trained jointly
with geometry. Following LangSplat / Feature-3DGS / Gaussian Grouping:

1. **Phase 1 (`--max-steps`)** — train RGB + geometry with MCMC densification. This is
   identical to a plain RGB run; no semantic field exists yet.
2. **Phase 2 (`--sem-steps`)** — attach a fresh semantic field to the *converged*
   Gaussians and train only it, with the geometry **detached/frozen** (MCMC off). The 2D
   masks are distilled onto fixed supports.

Why: geometry from the dense photometric loss is far more reliable than the semantic CE,
and freezing it means labelling becomes a clean multi-view fusion problem that **cannot
move or degrade the reconstruction**. Phase 2 is cheap (a few thousand steps — no SSIM,
no SH, no densification).

## Modules

| file | role |
|------|------|
| `colmap_dataset.py` | read COLMAP **text** model (`poses_txt/`) → posed cameras + init points, RAM-cached at training resolution |
| `model.py`          | Gaussian init from sparse points (k-NN scale seed); PLY + semantic-label export |
| `semantic.py`       | cache per-image class masks via `semantic_bev` segmenters (shared taxonomy) |
| `train.py`          | MCMC trainer (L1+SSIM RGB, optional semantic CE), CLI, exports |

## Install

gsplat compiles CUDA kernels on **first import** (JIT via torch's `cpp_extension`, ~60 s,
cached afterwards). That build needs two things that aren't here by default: a `nvcc`
matching the installed torch, and Python **dev headers**. Recipe below is what works on
this box (RTX 5060 / Blackwell sm_120, no system CUDA toolkit, no `python3.12-dev`).

```sh
# 1. Python deps. Force a *managed* interpreter -- the system python3.12 has no
#    Python.h, which the kernel build needs. python-build-standalone ships headers.
uv python install 3.12
uv sync --extra gsplat --extra segmentation --python-preference only-managed

# 2. A CUDA toolchain (nvcc) matching torch's CUDA version. torch here is 2.12+cu130
#    -> CUDA 13.0 (check: uv run --extra gsplat python -c "import torch;print(torch.version.cuda)")
~/bin/micromamba create -y -n cudatk -c nvidia -c conda-forge cuda-toolkit=13.0

# 3. The nvidia conda layout splits headers/libs under targets/<arch>/, but torch's
#    cpp_extension expects them at $CUDA_HOME/{include,lib64}. Stitch a shim CUDA_HOME:
CT=~/micromamba/envs/cudatk; SHIM=~/cuda-shim
mkdir -p $SHIM
ln -sf $CT/bin $SHIM/bin
ln -sf $CT/nvvm $SHIM/nvvm
ln -sf $CT/targets/x86_64-linux/include $SHIM/include
ln -sf $CT/targets/x86_64-linux/lib $SHIM/lib
ln -sf $CT/targets/x86_64-linux/lib $SHIM/lib64
```

Then set these in any shell that trains (kernels are cached after the first build):

```sh
export CUDA_HOME=~/cuda-shim
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0     # Blackwell sm_120
```

> **8 GB VRAM.** Keep `--cap-max` ≤ ~300k and `--data-factor ≥ 2` for a park capture.
> The init cloud is auto-subsampled to `--init-points` (default = `--cap-max`), because
> MCMC's cap only bounds *growth* — without subsampling the ~429k-point COLMAP cloud you
> could never get a model coarser than it. Raise the cap on a bigger GPU. A full park
> train is a multi-hour offline job — smoke test with `--limit` / small `--max-steps` first.

## Run

**Just pass `--session <name>`** and paths are filled from the canonical layout
([`../lar_session.py`](../lar_session.py)): `--model` = the refined COLMAP model if it
exists else the raw one, `--images` = `input/<name>`, `--out` = `output/<name>-gsplat[-sem]`.
Any explicit flag overrides.

Semantic park model (the usual command):

```sh
cd script/gsplat
uv run --extra gsplat --extra segmentation python train.py \
  --session maguro-park-after-itchy \
  --data-factor 2 --cap-max 300000 --max-steps 30000 \
  --semantic --segmenter mask2former-large
```

RGB only — drop `--semantic` (writes to `output/<name>-gsplat`). Smoke test — add
`--limit 40 --max-steps 500` (and `--out` if you don't want to overwrite the real run).

Explicit paths still work instead of `--session`:

```sh
uv run --extra gsplat python train.py \
  --model ../../input/maguro-park-after-itchy/colmap/poses_txt \
  --out   ../../output/maguro-gsplat --data-factor 2 --cap-max 300000 --max-steps 30000
```

`--segmenter` accepts any `semantic_bev` backend: `oneformer` / `oneformer-large` /
`mask2former` / `mask2former-large` (all ADE20K, MIT) or `clipseg` (open-vocab). Masks
are cached per-segmenter under `<out>/masks/<segmenter>/`, so switching backends is safe;
pass `--overwrite-masks` only to re-segment the *same* backend.

## Output (`<out>/`)

- `point_cloud.ply` — the trained Gaussians (INRIA field layout; opens in SuperSplat, the
  gsplat viewer, etc.)
- `point_cloud_labels.npy` + `point_cloud_semantic.ply` — *(semantic only)* per-Gaussian
  class id (row-aligned with `point_cloud.ply`) and a taxonomy-coloured point cloud
- `preview_*.png` — RGB render previews for sanity
- `masks/` — cached per-image class-id masks (segmentation runs once)
- `config.json` — the run's arguments

## Status / next

- [x] COLMAP-text → posed-camera dataset + sparse-point init
- [x] MCMC RGB trainer, cap-bounded for 8 GB / park scale
- [x] semantic head (distil `semantic_bev` segmenter → per-Gaussian labels)
- [ ] validate coarse full-park RGB train (multi-hour offline)
- [ ] appearance/exposure modelling for outdoor lighting variation
- [ ] feed labelled Gaussians back into `semantic_bev` as a geometry source
- [ ] 2DGS/surfel variant for cleaner ground-surface height
```
