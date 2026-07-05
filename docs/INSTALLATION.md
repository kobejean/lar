# Installation

Setup for the **map reconstruction pipeline** (COLMAP/GLOMAP + the Python scripts in
`script/colmap/`). [`uv`](https://docs.astral.sh/uv/) is the standard environment
manager for this project — it pins the Python version, manages dependencies, and
creates the virtualenv from the committed lockfile.

> Building the C++ core library itself (`make all`, `make frameworks`) is covered in
> the top-level [README](../README.md). This document covers the toolchain needed to
> turn LARScan captures into a `map.json`.

## Prerequisites

- **macOS** (Apple Silicon recommended)
- **Xcode Command Line Tools** — `xcode-select --install`
- **Homebrew** — https://brew.sh (Apple Silicon installs to `/opt/homebrew`)

Verify Homebrew is on your `PATH`:

```sh
brew --version
```

If `brew` is "command not found", add it to your shell and reload:

```sh
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## 1. Install uv

```sh
brew install uv
```

`uv` is the **standard** tool for this project. Don't use system `pip`/`venv`/`conda`
directly — everything goes through `uv` so environments stay reproducible across
machines.

## 2. Install COLMAP and GLOMAP

These are native binaries used for Structure-from-Motion reconstruction.

### macOS (Homebrew)

```sh
brew install colmap glomap
```

Verify:

```sh
colmap --help | head -1
glomap --help | head -1
```

> If you already have `colmap`/`glomap` installed elsewhere (e.g. a manual build in
> `/usr/local/bin`), that works too — the scripts just call them off your `PATH`.

### Linux / CUDA / no-sudo (conda-forge)

A pinned, CUDA-enabled COLMAP can be installed entirely in user space (no `sudo`)
via conda-forge — the package bundles the CUDA runtime, so only the NVIDIA **driver**
is needed. The exact pins (and the reasoning behind them — libfaiss variant, Blackwell
CUDA build) live in [`script/colmap/environment.colmap.yml`](../script/colmap/environment.colmap.yml).

```sh
# one-time: install micromamba (or use an existing mamba/conda)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# create the pinned env and verify
micromamba create -f script/colmap/environment.colmap.yml
micromamba run -n colmap colmap --version       # -> COLMAP 4.1.0 ... with CUDA
```

Then run the pipeline with this env's `colmap` on `PATH` (still driven by `uv`):

```sh
micromamba run -n colmap uv run python script/colmap/colmap.py input/<session> --use_arkit_poses --use_sequential
```

> Blackwell GPUs (sm_120, RTX 50-series) require the CUDA 12.9 build pinned in the
> env file. On a machine with no NVIDIA GPU, switch the colmap pin to the CPU build
> (`colmap=4.1.0=cpu_*`) — see the comments in the env file.

> ⚠️ This env pins **COLMAP 4.1** and deliberately omits the standalone `glomap`
> package (it would downgrade libcolmap). GLOMAP is integrated into 4.x as
> `colmap global_mapper`, so the standalone-`glomap` `--use_glomap` path won't be
> available here — use `--use_arkit_poses` / the default incremental mapper instead.

## 3. Set up the Python environment

From the repo root, `uv sync` reads `pyproject.toml` + `uv.lock` and builds the
virtualenv with the exact pinned dependencies (`numpy`, `opencv-python`):

```sh
cd /path/to/lar
uv sync
```

`uv` automatically downloads CPython **3.12** (pinned in `.python-version`) if you
don't already have it. No manual venv activation is needed — prefix commands with
`uv run` (see below).

## 4. Verify

```sh
uv run python -c "import cv2, numpy; print('cv2', cv2.__version__, '| numpy', numpy.__version__)"
```

Expected output (versions may differ):

```
cv2 4.13.0 | numpy 2.5.0
```

If that prints without error, you're ready to reconstruct maps — see
[RECONSTRUCTION.md](RECONSTRUCTION.md).

## Working with uv (cheat sheet)

| Task | Command |
| --- | --- |
| Install/refresh the env from the lockfile | `uv sync` |
| Run a script in the env | `uv run python script/colmap/colmap.py …` |
| Add a dependency | `uv add <package>` |
| Remove a dependency | `uv remove <package>` |
| Change the Python version | `uv python pin <version>` then `uv sync` |
| Update locked versions | `uv lock --upgrade && uv sync` |

The files `pyproject.toml`, `uv.lock`, and `.python-version` are committed to the
repo — they define the canonical environment. Commit changes to them whenever you
add/upgrade dependencies.

## C++ tools (for refinement)

The map **refinement** step (`lar_refine_colmap`, used in
[RECONSTRUCTION.md → Step 5](RECONSTRUCTION.md)) and the other native apps are built
from C++ via CMake. This is separate from the Python/uv setup above.

### Dependencies (Homebrew)

```sh
brew install cmake eigen opencv ceres-solver nlohmann-json
```

The build also needs **g2o 1.0.0** (graph optimization for bundle adjustment),
located via CMake `find_package`. It is vendored as source in `thirdparty/g2o` and
must be built + installed once — see [INSTALL_G2O_VIEWER.md](INSTALL_G2O_VIEWER.md)
for the build steps (the install lands under `/usr/local/lib/cmake/g2o`, where
`find_package` picks it up).

### Build

```sh
cd /path/to/lar
make fast          # Release build, parallel (-j8) → binaries in ./bin/
# make all         # single-threaded Release
# make debug       # Debug build
```

On Apple Silicon the Makefile pins arm64 + Apple Clang automatically. If CMake can't
find the Homebrew packages, pass the prefix explicitly:

```sh
make fast CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$(brew --prefix)"
```

### Verify

```sh
./bin/lar_refine_colmap     # prints usage (input/output args)
```

You're then ready to refine reconstructions — see
[RECONSTRUCTION.md → Step 5](RECONSTRUCTION.md).
