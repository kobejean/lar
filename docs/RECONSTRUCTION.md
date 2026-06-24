# Map Reconstruction

End-to-end workflow for building a map: **capture with LARScan → transfer to Mac →
reconstruct with COLMAP/GLOMAP**.

Prerequisites: complete [INSTALLATION.md](INSTALLATION.md) first (uv, COLMAP, GLOMAP).

```
┌──────────────┐   AirDrop/Finder   ┌──────────────┐   uv run colmap.py   ┌──────────┐
│  LARScan     │ ─────────────────► │  lar/input/  │ ───────────────────► │ map.json │
│  (iPhone/iPad)│   session folder   │  <session>/  │    GLOMAP/COLMAP     │          │
└──────────────┘                    └──────────────┘                      └──────────┘
```

## Step 1 — Capture data with LARScan

LARScan is the iOS capture app in the
[lar-swift](https://github.com/kobejean/lar-swift) repo (`Examples/LARScan`).

1. Open `Examples/LARScan/LARScan.xcodeproj` in Xcode, select your iPhone/iPad, and
   **Run**. (LiDAR is **not** required — depth is no longer used.)
2. The app opens in **Snap** mode (camera icon in the segmented control). Leave it
   there for mapping. (The location icon is **Localize** mode, used for testing
   localization against an existing map.)
3. Walk the area and tap **Snap** to capture frames from many viewpoints:
   - Aim for **generous overlap** between consecutive shots (~70%+).
   - Cover the space from varied angles and positions; avoid pure rotation in place.
   - Keep good, even lighting; avoid motion blur (pause briefly when you tap Snap).
   - Each Snap saves a **color** image plus its ARKit camera pose. GPS is recorded
     automatically in the background.
4. (Optional) Tap surfaces in the scene to place **anchors**, and tap two anchors in
   sequence to connect them into a navigation graph.
5. Tap **Save** to write the session metadata.

Each capture session is written to the app's Documents directory as a folder named
with a millisecond timestamp, e.g. `1782302260032/`, containing:

```
1782302260032/
  00000000_image.jpeg     # color frames, zero-padded index
  00000001_image.jpeg
  ...
  frames.json             # per-frame camera intrinsics + ARKit poses
  gps.json                # GPS observations matched to frames
  map.json                # anchors / map metadata
```

> Note: there are **no** `depth.pfm` / `confidence.pfm` files — the COLMAP pipeline
> derives depth geometrically, so LiDAR depth capture is disabled.

## Step 2 — Transfer the session folder to your Mac

LARScan exposes its Documents folder via both Finder file sharing and the Files app.
Pick whichever is convenient:

### Option A — Finder (USB cable)

1. Connect the device to your Mac with a cable.
2. Open **Finder** → select the device in the sidebar → **Files** tab.
3. Expand **LARScan** to see the session folders.
4. Drag a session folder (e.g. `1782302260032`) to your Mac.

### Option B — Files app + AirDrop / iCloud

1. On the device, open the **Files** app → **On My iPhone/iPad** → **LARScan**.
2. Long-press the session folder → **Share** → **AirDrop** to your Mac (or **Save to
   Files** in iCloud Drive, then pull it down on the Mac).

### Place it under `input/`

Move the transferred folder into the lar repo's `input/` directory:

```
lar/
  input/
    1782302260032/        # ← your session folder
      00000000_image.jpeg
      frames.json
      gps.json
      map.json
```

## Step 3 — Run the reconstruction

From the repo root, run the pipeline through `uv` (the standard runner — it uses the
locked environment, no manual venv activation):

```sh
cd /path/to/lar
uv run python script/colmap/colmap.py input/1782302260032 --use_glomap
```

Replace `1782302260032` with your session folder name. The pipeline:

1. Copies images into a `colmap/` working dir
2. Extracts SIFT features (OpenCV by default)
3. Matches features (exhaustive by default)
4. Inserts ARKit relative poses as odometry constraints
5. Runs sparse reconstruction (GLOMAP or COLMAP)
6. Aligns the model to ARKit metric scale + GPS
7. Exports `map.json`

### Options

| Flag | Effect |
| --- | --- |
| `--use_glomap` | Global SfM via GLOMAP — **faster**; good default first pass. Omit to use COLMAP's incremental mapper. |
| `--use_vocab_tree` | Vocabulary-tree matching instead of exhaustive (for large image sets). |
| `--use_colmap_sift` | Use COLMAP's SIFT instead of OpenCV's. |
| `--max_num_features N` | Max features per image (default `16384`). |
| `--alignment_max_error F` | Max error threshold for model alignment (default `0.1`). |

## Step 4 — Output

Results land in a `colmap/` subfolder of your session:

```
input/1782302260032/colmap/
  database.db        # COLMAP feature/match database
  sparse/0/          # reconstructed sparse model
  poses_txt/         # exported camera poses
  map.json           # ← metric-scaled map from COLMAP (refined further in Step 5)
```

Inspect the reconstruction visually with the `colmap gui …` command the script prints
on completion, e.g.:

```sh
colmap gui \
  --database_path input/1782302260032/colmap/database.db \
  --import_path   input/1782302260032/colmap/sparse/0 \
  --image_path    input/1782302260032/colmap
```

## Step 5 — Refine (global bundle adjustment)

The COLMAP `map.json` from Step 4 is a solid initialization, but a final global
bundle adjustment improves accuracy by jointly optimizing camera poses and
landmarks while folding ARKit's high-quality relative poses in as **odometry
constraints**. This is done by the native C++ tool `lar_refine_colmap`.

### Build the C++ tools (one-time)

This needs the native build toolchain — see
[INSTALLATION.md → C++ tools for refinement](INSTALLATION.md#c-tools-for-refinement).

```sh
cd /path/to/lar
make fast          # Release build (-j8); binaries land in ./bin/
```

### Run the refinement

```sh
# lar_refine_colmap <input_dir> [output_dir]
./bin/lar_refine_colmap input/1782302260032
# …or set an explicit output dir:
./bin/lar_refine_colmap input/1782302260032 ./output/my-refined-map
```

It reads the session metadata and the COLMAP reconstruction under
`<input_dir>/colmap/`, runs a staged optimization (pose-only, then full BA with
odometry), and writes the refined map. If `output_dir` is omitted it defaults to
`./output/<input-name>-refined`.

### Output

```
output/1782302260032-refined/
  map.json       # ← refined, metric-scaled map (the final deliverable)
  map.g2o        # the optimization graph (for inspection/debugging)
  frames.json
  gps.json
```

**Success signal:** the average reprojection error should drop noticeably. On the
sample 23-frame capture it went from ~1.3 px (COLMAP) to ~0.45 px after refinement,
culling ~21k landmark outliers.

## Tips & troubleshooting

- **Reconstruction fails / too few images registered:** capture more frames with
  greater overlap and viewpoint variety. Textureless walls and repetitive patterns
  are hard for SfM.
- **Wrong scale or alignment:** ensure GPS had a good fix during capture and that the
  scene had enough parallax (translation, not just rotation). Tune
  `--alignment_max_error`.
- **Slow on large captures:** start with `--use_glomap`, and add `--use_vocab_tree`
  for matching.
- **`colmap`/`glomap` not found:** confirm they're installed and on `PATH`
  (see [INSTALLATION.md](INSTALLATION.md)).
- **`make fast` can't find OpenCV/Eigen/g2o:** the native build needs the C++
  toolchain — see [INSTALLATION.md → C++ tools](INSTALLATION.md#c-tools-for-refinement).
  If CMake misses Homebrew packages, retry with
  `make fast CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$(brew --prefix)"`.
