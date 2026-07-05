"""Gaussian parameters: init from a sparse point cloud + PLY / label export.

The Gaussian set is a ``torch.nn.ParameterDict`` so gsplat's strategies can grow/prune
it in place. Attributes:

  - ``means``     (N, 3)              world-space centres, seeded from COLMAP points
  - ``scales``    (N, 3)  log-space   isotropic init from k-NN spacing
  - ``quats``     (N, 4)              identity rotations
  - ``opacities`` (N,)    logit
  - ``sh0``       (N, 1, 3)           SH DC term (view-independent colour)
  - ``shN``       (N, K-1, 3)         higher SH bands (K = (sh_degree+1)^2)
  - ``sem``       (N, C)   logits      *optional* per-Gaussian class logits (semantic head)

``sem`` lives in the same ParameterDict so the densification strategy relocates it
alongside geometry -- a relocated/split Gaussian keeps its semantics for free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

# Inverse of the first (DC) real spherical-harmonic basis coefficient. Converts a
# linear RGB colour in [0,1] to the SH DC coefficient 3DGS stores.
_SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / _SH_C0


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    return sh0 * _SH_C0 + 0.5


def _knn_scale(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to the k nearest neighbours, per point (isotropic scale seed)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    # query k+1 because the first hit is the point itself (distance 0)
    d, _ = tree.query(xyz, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    return np.clip(mean_d, 1e-4, None).astype(np.float32)


def init_gaussians(
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    sh_degree: int = 1,
    init_opacity: float = 0.1,
    scale_factor: float = 1.0,
    num_classes: int | None = None,
    device: str = "cuda",
) -> nn.ParameterDict:
    n = len(points_xyz)
    means = torch.from_numpy(points_xyz).float()

    dist = _knn_scale(points_xyz) * scale_factor
    scales = torch.log(torch.from_numpy(dist)).unsqueeze(-1).repeat(1, 3)

    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0  # identity (w, x, y, z)

    opacities = torch.logit(torch.full((n,), init_opacity))

    rgb = torch.from_numpy(points_rgb).float()
    n_sh = (sh_degree + 1) ** 2
    sh0 = rgb_to_sh0(rgb).unsqueeze(1)          # (N, 1, 3)
    shN = torch.zeros(n, n_sh - 1, 3)           # (N, K-1, 3)

    params = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }
    if num_classes is not None:
        params["sem"] = torch.zeros(n, num_classes)

    return nn.ParameterDict(
        {k: nn.Parameter(v.to(device)) for k, v in params.items()}
    )


# ------------------------------------------------------------------------------ export

def _write_ply(path: Path, verts: np.ndarray, fields: list[str]) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [(f, "f4") for f in fields]
    arr = np.empty(verts.shape[0], dtype=dtype)
    for i, f in enumerate(fields):
        arr[f] = verts[:, i]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


@torch.no_grad()
def export_gaussian_ply(params: nn.ParameterDict, path: str | Path) -> int:
    """Write a standard 3DGS ``.ply`` (INRIA field layout, readable by common viewers)."""
    path = Path(path)
    means = params["means"].detach().cpu().numpy()
    n = means.shape[0]
    normals = np.zeros_like(means)
    f_dc = params["sh0"].detach().cpu().numpy().reshape(n, -1)          # (N, 3)
    # INRIA lays out f_rest channel-major: [all band coeffs for R, then G, then B].
    shN = params["shN"].detach().cpu().numpy()                          # (N, K-1, 3)
    f_rest = np.transpose(shN, (0, 2, 1)).reshape(n, -1)               # (N, 3*(K-1))
    opacities = params["opacities"].detach().cpu().numpy().reshape(n, 1)
    scales = params["scales"].detach().cpu().numpy()                    # (N, 3)
    quats = params["quats"].detach().cpu().numpy()                      # (N, 4)

    verts = np.concatenate(
        [means, normals, f_dc, f_rest, opacities, scales, quats], axis=1
    )
    fields = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    _write_ply(path, verts, fields)
    return n


@torch.no_grad()
def export_semantic(params: nn.ParameterDict, path_stem: str | Path, color_lut) -> np.ndarray:
    """Export per-Gaussian semantic labels: an ``.npy`` of class ids + a colour ``.ply``.

    ``path_stem`` -> ``<stem>_labels.npy`` and ``<stem>_semantic.ply``. Rows are aligned
    with the Gaussian ``.ply`` order (same ParameterDict order), so labels index directly
    into the exported splats. Returns the label array.
    """
    path_stem = Path(path_stem)
    means = params["means"].detach().cpu().numpy()
    labels = params["sem"].detach().argmax(dim=1).cpu().numpy().astype(np.uint8)

    np.save(path_stem.with_name(path_stem.name + "_labels.npy"), labels)

    lut = np.asarray(color_lut, dtype=np.float32) / 255.0
    rgb = lut[labels]
    verts = np.concatenate([means, rgb], axis=1)
    _write_ply(
        path_stem.with_name(path_stem.name + "_semantic.ply"),
        verts,
        ["x", "y", "z", "red", "green", "blue"],
    )
    return labels
