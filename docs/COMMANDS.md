# Handy Commands (Linux dev box)

Quick reference for running the COLMAP pipeline on this machine. See
[INSTALLATION.md](INSTALLATION.md) and [RECONSTRUCTION.md](RECONSTRUCTION.md) for the
full workflow.

> **This box has no system `colmap`.** COLMAP 4.1 (CUDA) lives in a user-space
> micromamba env named `colmap`. Every command that touches COLMAP must run through
> `~/bin/micromamba run -n colmap ...` so the env's binary is on `PATH`. Python deps
> are managed with `uv`, so the pipeline is wrapped as
> `micromamba run -n colmap uv run python ...`.

## Environment

```sh
# Verify COLMAP sees CUDA (should print "... with CUDA")
~/bin/micromamba run -n colmap colmap --version

# Verify the GPU (RTX 5060, 8 GB, Blackwell sm_120)
nvidia-smi
```

## Run the reconstruction pipeline

Run from the repo root (`~/Code/lar`). The script is not executable — call it through
`python` (running `script/colmap/colmap.py` directly gives "Permission denied").

```sh
# ARKit-covisibility matching (loop-closure connectivity)
~/bin/micromamba run -n colmap uv run python script/colmap/colmap.py \
    input/<session> --use_covisibility

# GLOMAP global SfM (faster reconstruction)
~/bin/micromamba run -n colmap uv run python script/colmap/colmap.py \
    input/<session> --use_glomap

# Sequential matching (best for ordered/video-like captures)
~/bin/micromamba run -n colmap uv run python script/colmap/colmap.py \
    input/<session> --use_sequential
```

`<session>` is a directory under `input/` containing `*_image.jpeg` files and
`frames.json`.

### GPU acceleration

All heavy stages run on the GPU automatically — no flag needed:

| Stage              | GPU flag (set by the script)      |
| ------------------ | --------------------------------- |
| SIFT extraction    | `--FeatureExtraction.use_gpu 1`   |
| Feature matching   | `--FeatureMatching.use_gpu 1`     |

**8 GB VRAM caveat:** on park-sized datasets GPU SIFT can OOM. If it crashes during
extraction/matching, lower the feature budget:

```sh
... script/colmap/colmap.py input/<session> --use_covisibility --max_num_features 8192
```

## Pipeline options

| Flag                     | Default | Purpose                                                        |
| ------------------------ | ------- | -------------------------------------------------------------- |
| `--use_colmap_sift`      | off     | Use COLMAP's built-in SIFT instead of OpenCV                   |
| `--max_num_features N`   | 16384   | Max features per image (lower to fit VRAM)                     |
| `--alignment_max_error`  | 0.1     | Max error threshold for model alignment                        |
| `--use_vocab_tree`       | off     | Vocab-tree matching instead of exhaustive                      |
| `--use_sequential`       | off     | Sequential/sliding-window matching + vocab-tree loop detection |
| `--sequential_overlap N` | 10      | Images matched ahead per frame (sequential)                    |
| `--use_glomap`           | off     | GLOMAP global SfM instead of incremental COLMAP                |
| `--use_arkit_poses`      | off     | Triangulate against fixed ARKit poses (wide-baseline captures) |
| `--use_covisibility`     | off     | Augment matching with ARKit-covisibility loop-closure pairs    |

### Covisibility tuning (`--use_covisibility`)

| Flag                   | Default | Purpose                                            |
| ---------------------- | ------- | -------------------------------------------------- |
| `--covis_base_radius`  | 8.0     | Covisibility radius (m) before drift slack         |
| `--covis_drift_rate`   | 0.02    | Normal-tracking drift as fraction of arc length    |
| `--covis_drift_cap`    | 15.0    | Max drift slack added to the radius (m)            |
| `--covis_max_angle`    | 45.0    | Max optical-axis angle between paired frames (deg) |
| `--covis_min_seq_gap`  | 10      | Skip pairs closer than this in capture order       |
| `--covis_max_pairs`    | 30      | Keep only the nearest N candidates per image       |

## Raw COLMAP commands

Anything else you'd normally run as `colmap <cmd>`:

```sh
~/bin/micromamba run -n colmap colmap gui
~/bin/micromamba run -n colmap colmap <subcommand> <args...>
```
