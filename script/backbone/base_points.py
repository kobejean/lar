"""Object footprint *points/lines* from open-vocab detection + BEV foot-point clustering.

The dedup problem with semantic masks: the same tree, projected from 970 frames, smears into
one un-separable blob. The fix multi-view BEV localization uses is the **foot point** — reduce
each detected object to a single ground-contact point (bottom-centre of its box), project that
to the ground plane, and **cluster across frames**. Points from the same object collapse to one
landmark; a stray bad frame is an outlier that doesn't cluster. Instance detection (not semantic
segmentation) is what makes objects separable in the first place.

Pipeline (per sampled frame):
  1. **Grounding DINO** (Apache-2.0, open-vocab) detects `tree / bench / pole / sign / ...` →
     one box per instance from a text prompt.
  2. **Foot point** = bottom-centre of the box.
  3. Look up the **DEM ground depth** at that pixel (the ground surface we trust — reuses the
     footprint_labels DEM render — instead of mono depth at the noisy object edge) and
     back-project → a single world point on the ground.
Then across all frames: gravity-align, **DBSCAN per class** → one deduped footprint per object,
with an open-vocab label. Drops straight into IMDF `amenity.landmark` (trees) / `unit.structure`.

Geometry-only DEM by default; pass `--dem-source mono` (same as footprint_labels) for a denser
ground surface so more foot pixels resolve to a ground depth.

Run (from repo root; needs the `segmentation` extra for transformers/torch):
  uv run --extra segmentation python script/backbone/base_points.py --session maguro-park-after-itchy --sample 80
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "semantic_bev"))

from colmap_io import qvec2rotmat, read_model  # noqa: E402
from lar_session import Session  # noqa: E402
from footprint_labels import (  # noqa: E402
    backproject_positions, build_ground_mesh, make_frame_labels, pinhole_params,
)

DEFAULT_PROMPT = "tree. bench. pole. sign. trash can. street lamp. rock. bush."
# stable BGR colours per canonical class (fallback cycles for unknowns)
CLASS_COLORS = {
    "tree": (60, 170, 60), "bench": (200, 150, 40), "pole": (40, 140, 220),
    "sign": (40, 40, 220), "trash can": (180, 60, 200), "street lamp": (60, 200, 200),
    "rock": (120, 120, 120), "bush": (80, 200, 140),
}
_FALLBACK = [(200, 100, 60), (60, 100, 200), (100, 200, 60), (200, 60, 160)]


# --------------------------------------------------------------------------- #
# 2DGS surfel ground surface: the trained surfels are surface-aligned, so their
# centres are a dense, accurate surface cloud (same COLMAP frame gsplat trained on).
# Feed as `dem_world` — build_ground_mesh extracts ground as the low per-cell quantile.
# --------------------------------------------------------------------------- #
def read_gsplat_ply(path: Path, opacity_min: float = 0.1, log=print) -> np.ndarray:
    """Read a 3DGS/2DGS point_cloud.ply -> (N,3) surfel centres, opacity-filtered.

    Minimal binary_little_endian reader: parses the property list (all float32 in this
    format) to locate x/y/z/opacity by name, so it survives field-count changes."""
    raw = path.read_bytes()
    hdr_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:hdr_end].decode("ascii", "replace").splitlines()
    props, n = [], 0
    for line in header:
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property"):
            props.append(line.split()[-1])          # property float <name>
    assert {"x", "y", "z"} <= set(props), f"ply missing xyz: {props[:6]}"
    arr = np.frombuffer(raw[hdr_end:hdr_end + n * len(props) * 4],
                        dtype=np.float32).reshape(n, len(props))
    xyz = arr[:, [props.index("x"), props.index("y"), props.index("z")]].astype(np.float64)
    if opacity_min and "opacity" in props:
        alpha = 1.0 / (1.0 + np.exp(-arr[:, props.index("opacity")]))   # gsplat stores logit
        keep = alpha >= opacity_min
        xyz = xyz[keep]
        log(f"  2DGS surfels: {n} -> {len(xyz)} kept (opacity sigmoid >= {opacity_min})")
    else:
        log(f"  2DGS surfels: {n}")
    return xyz


def adapt_cameras_to_images(recon, images_dir, log=print):
    """Rescale COLMAP intrinsics to the actual image resolution.

    Common gotcha (e.g. mip-NeRF 360): COLMAP is solved on full-res but the provided images
    are a downsampled ``images_N`` dir, so detection pixels (image space) and the projection
    intrinsics (full-res) disagree by the downscale factor. No-op when sizes already match."""
    for cam in recon.cameras.values():
        name = next((im.name for im in recon.images.values() if im.camera_id == cam.id), None)
        if not name:
            continue
        p = Path(images_dir) / name
        if not p.exists():
            continue
        with Image.open(p) as im0:
            w, h = im0.size
        if (w, h) == (cam.width, cam.height):
            continue
        rx, ry = w / cam.width, h / cam.height
        q = cam.params.astype(float).copy()
        if cam.model == "PINHOLE":              # [fx, fy, cx, cy]
            q[0] *= rx; q[1] *= ry; q[2] *= rx; q[3] *= ry
        elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):  # [f, cx, cy, ...]
            q[0] *= rx; q[1] *= rx; q[2] *= ry
        else:                                    # OPENCV etc.: scale the [fx,fy,cx,cy] head
            q[:4] = q[:4] * [rx, ry, rx, ry]
        cam.params, cam.width, cam.height = q, w, h
        log(f"  adapted camera {cam.id}: {int(cam.width/rx)}x{int(cam.height/ry)} -> {w}x{h}")
    return recon


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def load_detector(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def detect(proc, model, pil: Image.Image, prompt: str, device: str,
           box_thr: float, text_thr: float):
    """Return list of (box xyxy in original px, label str, score)."""
    inputs = proc(images=pil, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[pil.size[::-1]])[0]
    boxes = res["boxes"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    labels = res.get("text_labels", res.get("labels"))
    return list(zip(boxes, labels, scores))


def canonical(label: str, classes: list[str]) -> str:
    """Map a (possibly multi-word / partial) detection phrase to a prompt class."""
    lab = str(label).lower().strip()
    for c in classes:
        if c in lab or lab in c:
            return c
    return lab or "object"


# --------------------------------------------------------------------------- #
# VLM pointing backend (Moondream2, Apache-2.0 — native .point() per class)
# --------------------------------------------------------------------------- #
def load_pointer(model_id: str, revision: str, device: str):
    """A pointing VLM: emits a 2D point per instance from a class name (no boxes)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        torch_dtype=torch.float16).to(device).eval()
    return model


@torch.no_grad()
def vlm_points(model, pil: Image.Image, classes: list[str]):
    """Query the VLM once per class -> list of (x_px, y_px, class, score) in original px.

    Moondream `.point(image, "<class>")` returns {"points": [{"x":0..1,"y":0..1}, ...]},
    one entry per detected instance (normalised coords). No per-point confidence, so score=1.
    """
    W, H = pil.size
    out = []
    for c in classes:
        res = model.point(pil, c)
        pts = res.get("points", res) if isinstance(res, dict) else res
        for p in pts:
            x = (p["x"] if isinstance(p, dict) else p[0]) * W
            y = (p["y"] if isinstance(p, dict) else p[1]) * H
            out.append((float(x), float(y), c, 1.0))
    return out


# --------------------------------------------------------------------------- #
# Gemini pointing backend (cloud oracle — needs $GEMINI_API_KEY). Not shippable
# as an offline model; used to measure the quality ceiling above the local pointers.
# --------------------------------------------------------------------------- #
def load_gemini(model_id: str):
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("--backend gemini needs GEMINI_API_KEY in the environment")
    return genai.Client(api_key=key), model_id


def _parse_gemini(txt: str, W: int, H: int):
    """Tolerant parse of Gemini's pointing reply -> [(x_px, y_px, label, score)].

    Slice to the outermost JSON array; if that won't parse, regex each {...} object. Accepts
    'point'/'point_2d'/'box_2d' [y,x] (0-1000). Robust to stray prose or code fences."""
    i, j = txt.find("["), txt.rfind("]")
    data = None
    if 0 <= i < j:
        try:
            data = json.loads(txt[i:j + 1])
        except Exception:
            data = None
    if data is None:
        data = []
        for m in re.finditer(r"\{[^{}]*\}", txt):
            try:
                data.append(json.loads(m.group()))
            except Exception:
                pass
    out = []
    for d in data:
        if not isinstance(d, dict):
            continue
        p = d.get("point") or d.get("point_2d") or d.get("box_2d")
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        y, x = float(p[0]), float(p[1])
        out.append((x / 1000 * W, y / 1000 * H, str(d.get("label", "")).lower(), 1.0))
    return out


def gemini_points(client_model, pil: Image.Image, classes: list[str], retries: int = 4):
    """One multi-class call per frame -> list of (x_px, y_px, class, score). Gemini returns
    ground-contact points as {"point": [y, x], "label"} normalised 0-1000. Retries only on
    rate-limit / transient network errors (content errors just yield whatever parsed)."""
    client, model_id = client_model
    W, H = pil.size
    prompt = ("Point to each " + ", ".join(classes) + " in this image, at the spot where the "
              "object meets the ground. Respond with ONLY a JSON list of "
              '{"point": [y, x], "label": "<class>"}, where class is exactly one of the listed '
              "names and [y, x] are normalised to 0-1000. No prose, no code fences.")
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model_id, contents=[pil, prompt])
        except Exception as e:
            msg = str(e)
            transient = any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "503",
                                               "UNAVAILABLE", "name resolution", "timed out"))
            if transient and attempt < retries - 1:
                time.sleep(15 * (attempt + 1)); continue
            print(f"  gemini call error: {msg[:110]}")
            return []
        return _parse_gemini(resp.text or "", W, H)
    return []


# --------------------------------------------------------------------------- #
# Moondream3 cloud pointing backend (needs $MOONDREAM_API_KEY). Same oracle role
# as gemini: Moondream3 is 9B-MoE (2B active) — the whole 9B (~18 GB bf16) must be
# resident, so it does NOT fit local 8 GB VRAM; the hosted API exposes the identical
# .point() interface. BSL-1.1 + Additional Use Grant (commercial internal use OK).
# --------------------------------------------------------------------------- #
def load_moondream_cloud(model_id: str):
    """Moondream cloud client (moondream3-preview by default). Needs $MOONDREAM_API_KEY."""
    import moondream as md
    key = os.environ.get("MOONDREAM_API_KEY")
    if not key:
        raise SystemExit("--backend moondream3 needs MOONDREAM_API_KEY in the environment")
    return md.vl(api_key=key, model=model_id) if model_id else md.vl(api_key=key)


def moondream_cloud_points(model, pil: Image.Image, classes: list[str], retries: int = 4):
    """One cloud .point() call per class -> [(x_px, y_px, class, 1.0)] in original px.

    Cloud .point() returns the SAME normalised {"points":[{"x":0..1,"y":0..1}]} shape as the
    local moondream backend, but each call is a network round-trip -> retry on transient errors
    (mirrors gemini_points). API returns no per-point confidence, so score=1."""
    W, H = pil.size
    out = []
    for c in classes:
        res = None
        for attempt in range(retries):
            try:
                res = model.point(pil, c)
            except Exception as e:
                msg = str(e)
                transient = any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "503", "502",
                                                   "timeout", "timed out", "Connection",
                                                   "name resolution"))
                if transient and attempt < retries - 1:
                    time.sleep(10 * (attempt + 1)); continue
                print(f"  moondream cloud error [{c}]: {msg[:110]}")
                res = None
            break
        if not res:
            continue
        pts = res.get("points", res) if isinstance(res, dict) else res
        for p in pts:
            x = (p["x"] if isinstance(p, dict) else p[0]) * W
            y = (p["y"] if isinstance(p, dict) else p[1]) * H
            out.append((float(x), float(y), c, 1.0))
    return out


# --------------------------------------------------------------------------- #
# foot point -> ground world point
# --------------------------------------------------------------------------- #
def world_from_render_px(c, r, d, scale, cam, im):
    """Render-res pixel (c,r) + its ground depth -> world point."""
    fx, fy, cx, cy = (v * scale for v in pinhole_params(cam))
    cam_pt = np.array([(c - cx) / fx * d, (r - cy) / fy * d, d])
    R = qvec2rotmat(im.qvec)
    return cam_pt @ R + (-R.T @ im.tvec)  # world = cam @ R + C


def pixel_world(px, py, ground_depth, scale, cam, im, min_depth=0.5):
    """A single original-image pixel -> world point on the DEM ground (None if off-ground).

    `px,py` are in original image coordinates; `scale` maps them into the render-res
    ground-depth buffer. Shared by the box foot-point path and the VLM point path.
    """
    h, w = ground_depth.shape
    fx_px = int(round(np.clip(px * scale, 0, w - 1)))
    fy_px = int(round(np.clip(py * scale, 0, h - 1)))
    d = ground_depth[fy_px, fx_px]
    if not np.isfinite(d) or d < min_depth:
        return None
    return world_from_render_px(fx_px, fy_px, d, scale, cam, im)


def foot_world(box, ground_depth, scale, cam, im, min_depth=0.5):
    """Bottom-centre of the box -> world point on the DEM ground (or None if off-ground)."""
    x0, y0, x1, y1 = box
    return pixel_world((x0 + x1) * 0.5, y1, ground_depth, scale, cam, im, min_depth)


def drop_to_ground(px, py, ground_depth, scale, min_depth=0.5):
    """A VLM object point lands mid-object (trunk/seat), where DEM depth is NaN. Walk DOWN
    that image column to the first ground pixel = the object's ground contact directly below.
    Returns (col, row, depth) in render-res, or None if no ground below the point."""
    h, w = ground_depth.shape
    c = int(round(np.clip(px * scale, 0, w - 1)))
    r0 = int(round(np.clip(py * scale, 0, h - 1)))
    col = ground_depth[:, c]
    finite = np.where(np.isfinite(col) & (col >= min_depth))[0]
    finite = finite[finite >= r0]
    if len(finite) == 0:
        return None
    r = int(finite[0])
    return c, r, float(col[r])


def pixel_ray(c, r, scale, cam, im):
    """Render-res pixel (c,r) -> world-space ray (origin = camera centre, unit direction).

    The metric placement's reliable half: uses only the (trusted) pose, no ground depth."""
    fx, fy, cx, cy = (v * scale for v in pinhole_params(cam))
    cam_dir = np.array([(c - cx) / fx, (r - cy) / fy, 1.0])
    R = qvec2rotmat(im.qvec)
    C = -R.T @ im.tvec                       # camera centre in world
    d = cam_dir @ R                          # camera dir -> world (world = cam @ R + C)
    return C, d / (np.linalg.norm(d) + 1e-12)


def triangulate_rays(origins: np.ndarray, dirs: np.ndarray):
    """Least-squares 3D point nearest a bundle of world rays (o_i, unit d_i).

    Solves (Σ Pᵢ) p = Σ Pᵢ oᵢ with Pᵢ = I - dᵢdᵢᵀ (projector onto the plane ⟂ the ray).
    Returns (point, mean_perp_residual_m, condition_number). A narrow-baseline cluster (all
    rays near-parallel) yields a high condition number -> triangulation is untrustworthy there."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    Ps = []
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o; Ps.append(P)
    cond = float(np.linalg.cond(A))
    p = np.linalg.solve(A + 1e-6 * np.eye(3), b)
    resid = float(np.mean([np.linalg.norm(P @ (p - o)) for P, o in zip(Ps, origins)]))
    return p, resid, cond


# --------------------------------------------------------------------------- #
# clustering (DBSCAN via KDTree; no sklearn)
# --------------------------------------------------------------------------- #
def dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = len(points)
    labels = np.full(n, -1, np.int64)
    if n == 0:
        return labels
    tree = cKDTree(points)
    visited = np.zeros(n, bool)
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        seeds = tree.query_ball_point(points[i], eps)
        if len(seeds) < min_samples:
            continue  # provisional noise; may still be absorbed into a cluster below
        labels[i] = cid
        k = 0
        while k < len(seeds):
            j = seeds[k]; k += 1
            if labels[j] == -1:
                labels[j] = cid
            if not visited[j]:
                visited[j] = True
                nj = tree.query_ball_point(points[j], eps)
                if len(nj) >= min_samples:
                    seeds.extend(nj)
        cid += 1
    return labels


def cluster_footprints(pts_uv: np.ndarray, labels: list[str], scores: np.ndarray,
                       classes: list[str], eps: float, min_samples: int, log=print):
    """Per-class DBSCAN -> one landmark per cluster (score-weighted centroid)."""
    out = []
    labels = np.asarray(labels)
    for c in sorted(set(labels)):
        m = labels == c
        p = pts_uv[m]; sc = scores[m]
        cl = dbscan(p, eps, min_samples)
        for k in sorted(set(cl) - {-1}):
            sel = cl == k
            w = sc[sel]
            centre = np.average(p[sel], axis=0, weights=w)
            out.append({"class": str(c), "u": float(centre[0]), "v": float(centre[1]),
                        "count": int(sel.sum()), "score": float(w.mean())})
    out.sort(key=lambda d: -d["count"])
    log(f"  clustered {len(pts_uv)} foot points -> {len(out)} objects "
        f"({len(set(labels))} classes)")
    return out


def cluster_triangulate(pts_uv, ray_o, ray_d, labels, scores, gf,
                        eps, min_samples, ground_tol=0.75, max_cond=5e3, log=print):
    """Per-class DBSCAN on the rough (DEM) positions to *group*, then multi-view **triangulate**
    each cluster's 3D contact from its member rays on the trusted poses — placement no longer
    rides on single-view depth. Singletons / near-parallel bundles fall back to the DEM centroid.
    Also reports the triangulated point's height above the DEM ground (should be ~0 if the
    contacts are consistent) as a built-in quality check."""
    labels = np.asarray(labels); scores = np.asarray(scores)
    ray_o = np.asarray(ray_o); ray_d = np.asarray(ray_d)
    out = []
    n_tri = n_fallback = 0
    for c in sorted(set(labels)):
        m = np.where(labels == c)[0]
        cl = dbscan(pts_uv[m], eps, min_samples)
        for k in sorted(set(cl) - {-1}):
            sel = m[cl == k]
            centroid = np.average(pts_uv[sel], axis=0, weights=scores[sel])
            rec = {"class": str(c), "count": int(len(sel)), "score": float(scores[sel].mean())}
            if len(sel) >= 2:
                P, resid, cond = triangulate_rays(ray_o[sel], ray_d[sel])
                if cond <= max_cond:
                    local = P @ gf.R.T                       # world -> gravity-local
                    uv = local[:2]                           # axes 0,1 horizontal
                    hag = float(local[2] - gf.height_at(uv[None, :])[0])   # height above DEM ground
                    rec.update(u=float(uv[0]), v=float(uv[1]), method="triangulated",
                               resid_m=round(resid, 3), cond=round(cond, 1), hag_m=round(hag, 3))
                    out.append(rec); n_tri += 1
                    continue
            # singleton / ill-conditioned: no reliable 3D -> DEM centroid, ground unknown
            rec.update(u=float(centroid[0]), v=float(centroid[1]), method="dem_fallback",
                       on_ground=True)
            out.append(rec); n_fallback += 1
    # The DEM is not a trustworthy ABSOLUTE ground datum (systematic vertical bias), so classify
    # ground-vs-elevated RELATIVE to the object population's own ground level: on-ground objects
    # pile up at the median hag; a vase-on-table sits a table-height above it. Absorbs DEM bias.
    hags = [o["hag_m"] for o in out if o.get("method") == "triangulated"]
    datum = float(np.median(hags)) if hags else 0.0
    for o in out:
        if "hag_m" in o:
            o["hag_rel"] = round(o["hag_m"] - datum, 3)
            o["on_ground"] = bool(o["hag_rel"] <= ground_tol)   # above the local ground = elevated
    log(f"  ground datum (median hag vs DEM) = {datum:+.2f} units; "
        f"elevated = hag_rel > {ground_tol}")
    out.sort(key=lambda d: -d["count"])
    resids = [o["resid_m"] for o in out if o.get("method") == "triangulated"]
    med = float(np.median(resids)) if resids else float("nan")
    log(f"  clustered {len(pts_uv)} pts -> {len(out)} objects "
        f"({n_tri} triangulated, {n_fallback} DEM-fallback); median ray residual {med:.2f} m")
    return out


# --------------------------------------------------------------------------- #
# mask-contour mode: SAM masks -> ground-contact silhouette -> footprint polygons
# --------------------------------------------------------------------------- #
def load_sam(model_id: str, device: str):
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def sam_masks(proc, model, pil: Image.Image, boxes: list, device: str):
    """One boolean mask (original image res) per input box, best of SAM's 3 proposals."""
    if not boxes:
        return []
    inputs = proc(pil, input_boxes=[[list(map(float, b)) for b in boxes]],
                  return_tensors="pt").to(device)
    out = model(**inputs)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0]          # (N, 3, H, W) bool
    best = out.iou_scores.cpu().numpy()[0].argmax(1)      # (N,) best proposal per box
    return [masks[i, best[i]].numpy().astype(bool) for i in range(len(boxes))]


def bottom_silhouette(mask: np.ndarray, stride: int = 2):
    """Lowest mask pixel per column = the object's ground-facing silhouette."""
    h = mask.shape[0]
    rows = np.arange(h)[:, None]
    has = mask.any(0)
    bottom = (mask * rows).argmax(0)          # largest True row index per column
    cols = np.where(has)[0][::stride]
    return cols, bottom[cols]


def backproject_pixels(px, py, depth, cam, im, scale):
    """(px,py) render-res pixels + per-pixel depth -> (N,3) world points."""
    fx, fy, cx, cy = (v * scale for v in pinhole_params(cam))
    cam_pts = np.stack([(px - cx) / fx * depth, (py - cy) / fy * depth, depth], axis=1)
    R = qvec2rotmat(im.qvec)
    return cam_pts @ R + (-R.T @ im.tvec)


def extract_polygons(accum_count, accum_weight, spec, accum_cell, cls,
                     min_hits, min_area, approx_eps):
    """Per-class BEV accumulator -> footprint polygons via connected components.

    Multi-view denoises: a real contact (e.g. a bench leg) projects to the SAME world cell
    from every view and accumulates; a floating silhouette edge (seat underside) projects to
    whatever ground is behind it, scatters, and stays below threshold.
    """
    binary = (accum_count >= min_hits).astype(np.uint8)
    if not binary.any():
        return []
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cell_area = accum_cell * accum_cell
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] * cell_area
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        eps = approx_eps * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        verts = [[float(spec.origin_u + c * accum_cell), float(spec.origin_v + r * accum_cell)]
                 for c, r in poly]
        cu = spec.origin_u + cents[i, 0] * accum_cell
        cv_ = spec.origin_v + cents[i, 1] * accum_cell
        out.append({"class": cls, "area_m2": round(float(area), 2),
                    "hits": int(accum_count[labels == i].sum()),
                    "centroid": [float(cu), float(cv_)], "polygon": verts})
    return out


def render_bev_polygons(gf, polygons, traj_uv, out_path: Path, ppm: float = 6.0):
    spec = gf.spec
    origin_u, origin_v, cs = spec.origin_u, spec.origin_v, spec.cell_size
    W = int(spec.cols * cs * ppm) + 40
    H = int(spec.rows * cs * ppm) + 40

    def to_px(u, v):
        return int((u - origin_u) * ppm) + 20, int((v - origin_v) * ppm) + 20

    img = np.full((H, W, 3), 248, np.uint8)
    cov = cv2.resize(gf.coverage.astype(np.uint8) * 255, (W - 40, H - 40),
                     interpolation=cv2.INTER_NEAREST)
    img[20:H - 20, 20:W - 20][cov > 0] = (233, 233, 233)
    if len(traj_uv) > 1:
        pts = np.array([to_px(u, v) for u, v in traj_uv], np.int32)
        cv2.polylines(img, [pts], False, (170, 170, 170), 1, cv2.LINE_AA)

    classes = sorted({p["class"] for p in polygons})
    cidx = {c: i for i, c in enumerate(classes)}
    overlay = img.copy()
    for p in polygons:
        col = colour_for(p["class"], cidx[p["class"]])
        poly = np.array([to_px(u, v) for u, v in p["polygon"]], np.int32)
        cv2.fillPoly(overlay, [poly], col)
        cv2.polylines(img, [poly], True, tuple(int(c * 0.6) for c in col), 1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    y = 30
    for c in classes:
        col = colour_for(c, cidx[c])
        k = sum(1 for p in polygons if p["class"] == c)
        cv2.rectangle(img, (24, y - 7), (36, y + 5), col, -1)
        cv2.putText(img, f"{c}: {k}", (44, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    cv2.imwrite(str(out_path), img)


# --------------------------------------------------------------------------- #
# BEV render
# --------------------------------------------------------------------------- #
def colour_for(cls: str, i: int):
    return CLASS_COLORS.get(cls, _FALLBACK[i % len(_FALLBACK)])


def render_bev(gf, objects, traj_uv, out_path: Path, ppm: float = 6.0):
    spec = gf.spec
    origin_u, origin_v, cs = spec.origin_u, spec.origin_v, spec.cell_size
    W = int(spec.cols * cs * ppm) + 40
    H = int(spec.rows * cs * ppm) + 40

    def to_px(u, v):
        return int((u - origin_u) * ppm) + 20, int((v - origin_v) * ppm) + 20

    img = np.full((H, W, 3), 248, np.uint8)
    # faint observed-DEM coverage as background context
    cov = cv2.resize(gf.coverage.astype(np.uint8) * 255, (W - 40, H - 40),
                     interpolation=cv2.INTER_NEAREST)
    img[20:H - 20, 20:W - 20][cov > 0] = (233, 233, 233)

    # camera trajectory
    if len(traj_uv) > 1:
        pts = np.array([to_px(u, v) for u, v in traj_uv], np.int32)
        cv2.polylines(img, [pts], False, (170, 170, 170), 1, cv2.LINE_AA)

    classes = sorted({o["class"] for o in objects})
    cidx = {c: i for i, c in enumerate(classes)}
    for o in objects:
        col = colour_for(o["class"], cidx[o["class"]])
        px, py = to_px(o["u"], o["v"])
        r = int(np.clip(3 + o["count"] * 0.5, 4, 18))
        cv2.circle(img, (px, py), r, col, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), r, (40, 40, 40), 1, cv2.LINE_AA)

    # legend
    y = 30
    for c in classes:
        col = colour_for(c, cidx[c])
        n = sum(o["count"] for o in objects if o["class"] == c)
        k = sum(1 for o in objects if o["class"] == c)
        cv2.circle(img, (30, y), 6, col, -1, cv2.LINE_AA)
        cv2.putText(img, f"{c}: {k} objs ({n} det)", (44, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    cv2.imwrite(str(out_path), img)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session")
    ap.add_argument("--model", help="COLMAP text model dir (default: session.best_model())")
    ap.add_argument("--images", help="image dir (default: session.images)")
    ap.add_argument("--out", help="output dir (default: output/<session>-basepoints)")
    ap.add_argument("--dem-source", choices=["colmap", "mono", "gsplat2d"], default="colmap",
                    help="ground-surface cloud for the DEM: colmap (sparse) / mono (dense but "
                         "vertically noisy) / gsplat2d (dense AND surface-aligned surfels). "
                         "Occupancy always stays on the COLMAP obstacle cloud.")
    ap.add_argument("--depth-dir", help="mono depth dir (default: session.depth_dir('mono'))")
    ap.add_argument("--gsplat-ply", help="2DGS point_cloud.ply (default: session gsplat2d out)")
    ap.add_argument("--gsplat-opacity", type=float, default=0.1,
                    help="drop 2DGS surfels below this sigmoid opacity (floaters)")
    ap.add_argument("--mode", choices=["points", "mask-contour"], default="points",
                    help="points = foot point per object (thin objects); "
                         "mask-contour = SAM mask -> ground silhouette -> footprint polygon "
                         "(multi-contact objects: benches, signs)")
    ap.add_argument("--backend", choices=["gdino", "moondream", "moondream3", "gemini"],
                    default="gdino",
                    help="gdino = Grounding DINO boxes (foot point / mask-contour); "
                         "moondream = local Moondream2 VLM pointing (Apache-2.0); "
                         "moondream3 = Moondream3 cloud pointing (needs $MOONDREAM_API_KEY; "
                         "9B-MoE, too big for local 8 GB VRAM); "
                         "gemini = cloud VLM pointing oracle (needs $GEMINI_API_KEY). "
                         "All VLM backends emit one point per instance, then drop to the "
                         "ground contact. Force --mode points.")
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--vlm-model", default="vikhyatk/moondream2")
    ap.add_argument("--vlm-revision", default="2025-06-21")
    ap.add_argument("--gemini-model", default="gemini-flash-lite-latest")
    ap.add_argument("--moondream-model", default="",
                    help="Moondream cloud model id for --backend moondream3; empty (default) = "
                         "client/server default, which is moondream3-preview. Pin a finetune "
                         "via 'moondream3-preview/ft_id@step'.")
    ap.add_argument("--sam-model", default="facebook/sam-vit-base",
                    help="SAM checkpoint for mask-contour (SAM2 is a drop-in)")
    ap.add_argument("--accum-cell", type=float, default=0.25, help="BEV accumulator cell (m)")
    ap.add_argument("--min-area", type=float, default=0.1, help="min footprint polygon area (m^2)")
    ap.add_argument("--approx-eps", type=float, default=0.02, help="polygon simplify (frac of perimeter)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="open-vocab classes, '.'-separated")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=1.2, help="DBSCAN cluster radius (m)")
    ap.add_argument("--min-samples", type=int, default=3, help="min detections to accept an object")
    ap.add_argument("--placement", choices=["dem", "triangulate"], default="dem",
                    help="dem = single-view back-projection to the DEM ground (default); "
                         "triangulate = group by DEM position, then multi-view ray "
                         "triangulation on the poses (drops single-view depth for the final "
                         "position). points mode only.")
    ap.add_argument("--min-depression", type=float, default=7.0,
                    help="reject a detection whose contact ray is shallower than this many "
                         "degrees below horizontal (grazing near-horizon rays carry ~no range "
                         "info; a walking cam sees most distant objects this way).")
    ap.add_argument("--ground-tol", type=float, default=0.75,
                    help="triangulated anchor within this height (scene units) of the DEM "
                         "ground = a ground obstacle; higher = elevated (vase-on-table, hung "
                         "sign) and flagged on_ground=false. Scale with the scene.")
    ap.add_argument("--cell-size", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=512, help="DEM render resolution for depth lookup")
    ap.add_argument("--max-range", type=float, default=30.0)
    ap.add_argument("--sample", type=int, default=80, help="frames spaced evenly across the capture")
    ap.add_argument("--save-detections", type=int, default=8, help="save N per-frame detection overlays")
    args = ap.parse_args()

    s = Session(args.session) if args.session else None
    if s:
        model = Path(args.model) if args.model else s.best_model()
        images = Path(args.images) if args.images else s.images
        out = Path(args.out) if args.out else s.root / "output" / f"{s.name}-basepoints"
        depth_dir = Path(args.depth_dir) if args.depth_dir else s.depth_dir("mono")
        gsplat_ply = (Path(args.gsplat_ply) if args.gsplat_ply
                      else s.gsplat_out(mode="2dgs") / "point_cloud.ply")
    else:
        if not (args.model and args.images and args.out):
            ap.error("without --session, pass --model, --images and --out")
        model, images, out = Path(args.model), Path(args.images), Path(args.out)
        depth_dir = Path(args.depth_dir) if args.depth_dir else None
        gsplat_ply = Path(args.gsplat_ply) if args.gsplat_ply else None
    out.mkdir(parents=True, exist_ok=True)
    det_dir = out / "detections"
    det_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model {model}\nimages {images}\nout {out}\ndevice {device}")
    recon = read_model(str(model))
    adapt_cameras_to_images(recon, images)   # handle downsampled images_N dirs (no-op if matched)

    dem_world = None
    if args.dem_source == "mono":
        if depth_dir is None:
            ap.error("--dem-source mono needs --depth-dir (or --session)")
        print(f"DEM source: mono depth <- {depth_dir}")
        dem_world = backproject_positions(recon, depth_dir, list(recon.images), 8, 0.1)
    elif args.dem_source == "gsplat2d":
        if gsplat_ply is None or not gsplat_ply.exists():
            ap.error(f"--dem-source gsplat2d needs a 2DGS ply (--gsplat-ply); got {gsplat_ply}")
        print(f"DEM source: 2DGS surfels <- {gsplat_ply}")
        dem_world = read_gsplat_ply(gsplat_ply, args.gsplat_opacity)
        # 3DGS/2DGS park a few background surfels at extreme coords -> clip to the scene
        # (camera-trajectory bbox + object range) or they blow up the DEM grid.
        cams = np.array([-qvec2rotmat(im.qvec).T @ im.tvec for im in recon.images.values()])
        lo, hi = cams.min(0) - args.max_range, cams.max(0) + args.max_range
        inside = np.all((dem_world >= lo) & (dem_world <= hi), axis=1)
        print(f"  clipped to scene bbox: {len(dem_world)} -> {int(inside.sum())} surfels")
        dem_world = dem_world[inside]
    mesh, occ_world = build_ground_mesh(
        recon, args.cell_size, (0.4, 2.0), 3, 0.8, dem_world=dem_world)
    gf = mesh.gf

    classes = [c.strip().lower() for c in args.prompt.split(".") if c.strip()]
    print(f"classes: {classes}")
    proc = det_model = pointer = gemini = sam_proc = sam_model = None
    is_vlm = args.backend in ("moondream", "moondream3", "gemini")
    if is_vlm and args.mode != "points":
        print(f"{args.backend} backend -> forcing --mode points")
        args.mode = "points"
    if args.backend == "moondream":
        print(f"loading pointer VLM {args.vlm_model}@{args.vlm_revision}")
        pointer = load_pointer(args.vlm_model, args.vlm_revision, device)
    elif args.backend == "moondream3":
        print(f"moondream3 cloud pointing: {args.moondream_model or '(client default)'}")
        pointer = load_moondream_cloud(args.moondream_model)
    elif args.backend == "gemini":
        print(f"gemini pointing oracle: {args.gemini_model}")
        gemini = load_gemini(args.gemini_model)
    else:
        proc, det_model = load_detector(args.detector, device)
        if args.mode == "mask-contour":
            print(f"loading SAM {args.sam_model}")
            sam_proc, sam_model = load_sam(args.sam_model, device)

    ims = sorted(recon.images.values(), key=lambda im: im.name)
    if args.sample and args.sample < len(ims):
        idx = np.linspace(0, len(ims) - 1, args.sample).round().astype(int)
        ims = [ims[i] for i in idx]

    spec = gf.spec
    acols = int(spec.cols * spec.cell_size / args.accum_cell) + 1
    arows = int(spec.rows * spec.cell_size / args.accum_cell) + 1
    accum_count: dict[str, np.ndarray] = {}
    accum_weight: dict[str, np.ndarray] = {}

    def accum_add(cls, uv, score):
        col = int((uv[0] - spec.origin_u) / args.accum_cell)
        row = int((uv[1] - spec.origin_v) / args.accum_cell)
        if 0 <= col < acols and 0 <= row < arows:
            if cls not in accum_count:
                accum_count[cls] = np.zeros((arows, acols), np.int32)
                accum_weight[cls] = np.zeros((arows, acols), np.float32)
            accum_count[cls][row, col] += 1
            accum_weight[cls][row, col] += score

    all_uv, all_lab, all_score, traj_uv = [], [], [], []
    all_ray_o, all_ray_d = [], []          # per-detection world ray (origin, unit dir) for triangulation
    all_dep = []                            # sin(depression below horizontal) of each contact ray
    n_pts = 0
    for fi, im in enumerate(ims):
        cam = recon.cameras[im.camera_id]
        R = qvec2rotmat(im.qvec)
        traj_uv.append(((-R.T @ im.tvec) @ gf.R.T)[:2])   # camera centre -> gravity-local
        res = make_frame_labels(mesh, occ_world, im, cam, args.size,
                                args.max_range, 0.5, 0)
        if res is None:
            continue
        ground_depth = res["depth"]
        gh, gw = ground_depth.shape
        scale = args.size / max(cam.width, cam.height)
        pil = Image.open(images / im.name).convert("RGB")

        if is_vlm:
            if args.backend == "moondream":
                vpts = vlm_points(pointer, pil, classes)
            elif args.backend == "moondream3":
                vpts = moondream_cloud_points(pointer, pil, classes)
            else:
                vpts = gemini_points(gemini, pil, classes)
            overlay = cv2.imread(str(images / im.name)) if fi < args.save_detections else None
            for (x, y, cls_raw, score) in vpts:
                cls = canonical(cls_raw, classes)
                hit = drop_to_ground(x, y, ground_depth, scale)   # object point -> ground contact
                if hit is None:
                    continue
                c, r, d = hit
                w = world_from_render_px(c, r, d, scale, cam, im)
                ro, rd = pixel_ray(x * scale, y * scale, scale, cam, im)   # ray through the anchor
                all_uv.append((w @ gf.R.T)[:2]); all_lab.append(cls)
                all_score.append(score); all_ray_o.append(ro); all_ray_d.append(rd)
                all_dep.append(float(-np.dot(rd, gf.up))); n_pts += 1
                if overlay is not None:
                    col = colour_for(cls, classes.index(cls) if cls in classes else 0)
                    ox, oy = int(x), int(y)                        # VLM object point
                    gx, gy = int(c / scale), int(r / scale)        # ground contact (orig px)
                    cv2.circle(overlay, (ox, oy), 5, col, 2, cv2.LINE_AA)
                    cv2.line(overlay, (ox, oy), (gx, gy), col, 1, cv2.LINE_AA)
                    cv2.circle(overlay, (gx, gy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(overlay, cls, (ox + 6, max(oy - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
            if overlay is not None:
                cv2.imwrite(str(det_dir / f"{Path(im.name).stem}.jpg"), overlay)
            if (fi + 1) % 20 == 0:
                print(f"  {fi + 1}/{len(ims)} frames, {n_pts} points")
            continue

        dets = detect(proc, det_model, pil, args.prompt, device, args.box_thr, args.text_thr)
        masks = (sam_masks(sam_proc, sam_model, pil, [d[0] for d in dets], device)
                 if args.mode == "mask-contour" else [None] * len(dets))
        overlay = cv2.imread(str(images / im.name)) if fi < args.save_detections else None

        for di, (box, label, score) in enumerate(dets):
            cls = canonical(label, classes)
            col = colour_for(cls, classes.index(cls) if cls in classes else 0)
            if args.mode == "points":
                w = foot_world(box, ground_depth, scale, cam, im)
                if w is None:
                    continue
                x0b, y0b, x1b, y1b = box
                cg = int(round(np.clip((x0b + x1b) * 0.5 * scale, 0, gw - 1)))
                rg = int(round(np.clip(y1b * scale, 0, gh - 1)))
                ro, rd = pixel_ray(cg, rg, scale, cam, im)
                all_uv.append((w @ gf.R.T)[:2]); all_lab.append(cls)
                all_score.append(float(score)); all_ray_o.append(ro); all_ray_d.append(rd)
                all_dep.append(float(-np.dot(rd, gf.up))); n_pts += 1
                if overlay is not None:
                    x0, y0, x1, y1 = box.astype(int)
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 2)
                    cv2.circle(overlay, (int((x0 + x1) / 2), y1), 6, (0, 0, 255), -1)
            else:  # mask-contour
                mask_r = cv2.resize(masks[di].astype(np.uint8), (gw, gh),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                cols, brows = bottom_silhouette(mask_r, stride=2)
                if len(cols) == 0:
                    continue
                d = ground_depth[brows, cols]
                ok = np.isfinite(d) & (d > 0.5)
                if not ok.any():
                    continue
                world = backproject_pixels(cols[ok].astype(float), brows[ok].astype(float),
                                           d[ok], cam, im, scale)
                for uv in (world @ gf.R.T)[:, :2]:
                    accum_add(cls, uv, float(score)); n_pts += 1
                if overlay is not None:
                    cnts, _ = cv2.findContours(masks[di].astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, cnts, -1, col, 2)
            if overlay is not None:
                x0, y0 = box[:2].astype(int)
                cv2.putText(overlay, f"{cls} {score:.2f}", (x0, max(y0 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        if overlay is not None:
            cv2.imwrite(str(det_dir / f"{Path(im.name).stem}.jpg"), overlay)
        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(ims)} frames, {n_pts} points")

    traj_uv = [(t[0], t[1]) for t in traj_uv]
    print(f"collected {n_pts} contact points over {len(ims)} frames (mode={args.mode})")

    if args.mode == "points":
        if not all_uv:
            raise SystemExit("no foot points landed on the ground — check detector/prompt/DEM")
        uv = np.array(all_uv); ro = np.array(all_ray_o); rd = np.array(all_ray_d)
        lab = np.array(all_lab); sc = np.array(all_score); dep = np.array(all_dep)
        # depression-angle gate: drop grazing near-horizon rays (no usable range; a walking cam
        # sees most distant objects this way). Steep/close observations are what localise.
        keep = dep >= np.sin(np.radians(args.min_depression))
        print(f"  depression gate (>= {args.min_depression:.0f} deg): "
              f"{len(uv)} -> {int(keep.sum())} contacts kept")
        uv, ro, rd, lab, sc = uv[keep], ro[keep], rd[keep], lab[keep], sc[keep]
        if len(uv) == 0:
            raise SystemExit("no contacts survived the depression gate — lower --min-depression")
        if args.placement == "triangulate":
            objects = cluster_triangulate(uv, ro, rd, lab, sc, gf,
                                          args.eps, args.min_samples, ground_tol=args.ground_tol)
            ground = [o for o in objects if o.get("on_ground", True)]
            print(f"  {len(ground)}/{len(objects)} on-ground obstacles "
                  f"(rest elevated: hag > {args.ground_tol} units)")
        else:
            objects = cluster_footprints(uv, lab, sc, classes, args.eps, args.min_samples)
        # BEV shows the walkable-footprint layer = on-ground obstacles only (elevated objects
        # stay in objects.json flagged on_ground=false for downstream use).
        render_bev(gf, [o for o in objects if o.get("on_ground", True)], traj_uv,
                   out / "bev_footprints.png")
    else:
        objects = []
        for cls in accum_count:
            objects += extract_polygons(accum_count[cls], accum_weight[cls], spec,
                                        args.accum_cell, cls, args.min_samples,
                                        args.min_area, args.approx_eps)
        objects.sort(key=lambda o: -o["area_m2"])
        render_bev_polygons(gf, objects, traj_uv, out / "bev_footprints.png")

    (out / "objects.json").write_text(json.dumps(objects, indent=2))
    print(f"\n{len(objects)} objects -> {out}/bev_footprints.png")
    by_cls: dict[str, int] = {}
    for o in objects:
        by_cls[o["class"]] = by_cls.get(o["class"], 0) + 1
    for c, k in sorted(by_cls.items(), key=lambda x: -x[1]):
        print(f"  {c:14s} {k}")


if __name__ == "__main__":
    main()
