"""Canonical directory layout for a LAR capture session — one source of truth.

Every phase of the pipeline reads/writes paths derived from a single **session name**
(e.g. ``maguro-park-after-itchy``). Encoding the convention here means each tool can accept
just ``--session <name>`` instead of re-specifying ``--model``/``--images``/``--out`` every
time. Explicit flags still override whatever the session resolves.

Layout (relative to the repo root that contains ``input/`` and ``output/``):

    input/<name>/                         raw capture (images, frames.json, gps.json, map.json)
    input/<name>/colmap/poses_txt/        COLMAP text model from script/colmap/colmap.py
    output/<name>-refined/                lar_refine_colmap output
    output/<name>-refined/colmap/sparse/0/  refined COLMAP text model (bundle-adjusted)
    output/<name>-gsplat[-sem]/           script/gsplat/train.py output
    output/<name>-sbev[-gsplat]/          script/semantic_bev/pipeline.py output

``best_model()`` prefers the refined (bundle-adjusted) model when present, else the raw
COLMAP model — that's the sensible default input for the gsplat/BEV phases.
"""

from __future__ import annotations

from pathlib import Path

# script/lar_session.py -> repo root is two levels up (root/script/lar_session.py).
REPO_ROOT = Path(__file__).resolve().parent.parent


class Session:
    def __init__(self, name: str, root: Path | str = REPO_ROOT):
        self.name = name
        self.root = Path(root)

    # ---- inputs -------------------------------------------------------------
    @property
    def input_dir(self) -> Path:
        return self.root / "input" / self.name

    @property
    def images(self) -> Path:
        return self.input_dir

    @property
    def colmap_model(self) -> Path:
        """Raw COLMAP text model from the reconstruction phase."""
        return self.input_dir / "colmap" / "poses_txt"

    @property
    def refined_dir(self) -> Path:
        return self.root / "output" / f"{self.name}-refined"

    @property
    def refined_model(self) -> Path:
        """Bundle-adjusted COLMAP text model from lar_refine_colmap (the richer one)."""
        return self.refined_dir / "colmap" / "sparse" / "0"

    def best_model(self) -> Path:
        """Refined model if it exists, else the raw COLMAP model."""
        return self.refined_model if (self.refined_model / "images.txt").exists() else self.colmap_model

    # ---- outputs ------------------------------------------------------------
    def gsplat_out(self, semantic: bool = False) -> Path:
        return self.root / "output" / f"{self.name}-gsplat{'-sem' if semantic else ''}"

    def sbev_out(self, source: str = "colmap", tag: str | None = None) -> Path:
        suffix = f"-{tag}" if tag else ("-gsplat" if source == "gsplat" else "")
        return self.root / "output" / f"{self.name}-sbev{suffix}"

    def depth_dir(self, backend: str) -> Path:
        """Per-backend per-view depth maps for the depth bake-off (mvs/2dgs/mono/3dgs)."""
        return self.root / "output" / f"{self.name}-depth-{backend}"
