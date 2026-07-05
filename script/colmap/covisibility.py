"""ARKit-covisibility matching: propose image pairs to match from ARKit geometry.

Appearance retrieval (vocab tree) fails to connect revisits on repetitive,
low-texture capture (e.g. park foliage) -- the view graph fragments. ARKit poses
give geometric covisibility for free: two frames whose cameras are near in space
and look in similar directions *should* be matched, regardless of how similar
they look. This module emits those candidate pairs; `colmap matches_importer
--match_type pairs` then does the actual SIFT matching + geometric verification,
so false candidates are rejected by two-view geometry, not trusted blindly.

Drift-aware gating
------------------
ARKit's *global* position drifts, so the straight-line distance between two
camera centers is uncertain -- and the uncertainty grows with the *trajectory
arc length* between them (VIO drift is ~% of distance travelled), NOT with their
frame-index gap. A stationary burst adds many frames but no drift; a fast walk
adds few frames but lots of drift. So the covisibility radius is widened by a
drift slack proportional to the arc length separating the two frames:

    effective_radius(i, j) = base_radius + min(drift_slack(i, j), drift_cap)

where drift_slack is integrated from a per-segment rate that is *higher across
low-confidence intervals* (odom_state / lar::OdometryConfidence). This widens the
gate exactly where ARKit is least trustworthy and keeps it tight where ARKit is
solid -- so we don't over-propose pairs everywhere.
"""

import math
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

from colmap_pose import extract_rotation_translation_from_extrinsics


def _image_name(frame_id):
    return f"{frame_id:08d}_image.jpeg"


# Per-segment drift-rate multiplier keyed on the interval's worst tracking state.
# odom_state mirrors lar::OdometryConfidence: 0 Normal, 1..3 Limited*, 4
# Relocalizing, 5 Unavailable. Normal VIO drifts slowly (base rate); Limited
# tracking drifts faster; a relocalization can teleport the pose, so its interval
# gets a large slack so we still propose the (now far-apart) covisible pair.
def _drift_rate_multiplier(odom_state):
    if odom_state <= 0:
        return 1.0
    if odom_state <= 3:
        return 3.0
    return 8.0


def _frame_geometry(frame):
    """Camera center C and unit viewing direction (both in ARKit world coords)."""
    R_c2w, C = extract_rotation_translation_from_extrinsics(
        frame["extrinsics"], apply_colmap_conversion=False)
    # ARKit camera looks down its local -z; the world-space optical axis is the
    # third column negated.
    forward = -R_c2w[:, 2]
    n = np.linalg.norm(forward)
    if n > 0:
        forward = forward / n
    return C, forward


def _db_image_names(database_path):
    conn = sqlite3.connect(str(database_path))
    try:
        names = {name for (name,) in conn.execute("SELECT name FROM images")}
    finally:
        conn.close()
    return names


def generate_covisibility_pairs(
    frames,
    database_path,
    pairs_path,
    *,
    base_radius=8.0,
    base_drift_rate=0.02,
    drift_cap=15.0,
    max_angle_deg=45.0,
    min_seq_gap=10,
    max_pairs_per_image=30,
):
    """Write an ARKit-covisibility candidate pair list for `matches_importer`.

    Args:
        frames: ARKit frames (dicts with 'id', 'extrinsics', optional 'odom_state').
        database_path: COLMAP db -- only frames whose image is present are emitted.
        pairs_path: output text file, one "name_a name_b" per line.
        base_radius: covisibility radius in metres before drift slack (the scene's
            co-visibility range).
        base_drift_rate: Normal-tracking drift as a fraction of arc length
            (e.g. 0.02 = 2% of distance travelled).
        drift_cap: maximum drift slack added to the radius (metres); keeps the gate
            from exploding across very long loops.
        max_angle_deg: reject pairs whose optical axes differ by more than this --
            SIFT can't match a bench seen from opposite sides, so don't propose it.
        min_seq_gap: skip pairs closer than this in capture order; the sequential
            matcher already covers the local window, so covisibility focuses on the
            non-redundant loop-closure pairs.
        max_pairs_per_image: keep only the nearest N candidates per image (caps
            cost where the drift-widened radius sweeps in many neighbours).

    Returns:
        The number of unique pairs written.
    """
    present = _db_image_names(database_path)
    frames = [f for f in frames if _image_name(f["id"]) in present]
    frames = sorted(frames, key=lambda f: f["id"])
    n = len(frames)
    if n < 2:
        Path(pairs_path).write_text("")
        print("Covisibility: fewer than 2 posed frames in database; no pairs.")
        return 0

    names = [_image_name(f["id"]) for f in frames]
    P = np.empty((n, 3))
    Fwd = np.empty((n, 3))
    odom = np.array([int(f.get("odom_state", 0)) for f in frames])
    for i, f in enumerate(frames):
        P[i], Fwd[i] = _frame_geometry(f)

    # Cumulative arc length and cumulative drift budget along the capture path.
    # seg[k] is the segment from frame k-1 to k; its drift contribution uses that
    # interval's tracking state (odom_state describes the interval INTO frame k).
    seg = np.zeros(n)
    seg[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1)
    rate = base_drift_rate * np.array([_drift_rate_multiplier(s) for s in odom])
    drift_budget = np.concatenate([[0.0], np.cumsum(seg[1:] * rate[1:])])

    cos_max = math.cos(math.radians(max_angle_deg))
    idx = np.arange(n)
    pairs = set()

    for i in range(n):
        d = np.linalg.norm(P - P[i], axis=1)
        slack = np.minimum(np.abs(drift_budget - drift_budget[i]), drift_cap)
        eff_radius = base_radius + slack
        cos_ang = Fwd @ Fwd[i]
        seq_gap = np.abs(idx - i)

        mask = (d <= eff_radius) & (cos_ang >= cos_max) & (seq_gap >= min_seq_gap)
        mask[i] = False
        cand = np.where(mask)[0]
        if cand.size == 0:
            continue
        # Nearest-first, capped.
        cand = cand[np.argsort(d[cand])[:max_pairs_per_image]]
        for j in cand:
            a, b = (i, int(j)) if i < j else (int(j), i)
            pairs.add((a, b))

    lines = sorted(f"{names[a]} {names[b]}" for a, b in pairs)
    Path(pairs_path).write_text("\n".join(lines) + ("\n" if lines else ""))

    # Report per-image fan-out so the widened radius is visible/tunable.
    per_image = np.zeros(n, dtype=int)
    for a, b in pairs:
        per_image[a] += 1
        per_image[b] += 1
    posed_with_pairs = int(np.count_nonzero(per_image))
    print(f"Covisibility: {len(pairs)} candidate pairs over {n} posed frames "
          f"({posed_with_pairs} frames have >=1 pair; "
          f"mean {per_image.mean():.1f}, max {per_image.max()} per frame).")
    return len(pairs)


def run_colmap_matches_importer(database_path, pairs_path):
    """Match + geometrically verify a custom pair list into the COLMAP database.

    `--match_type pairs` runs SIFT matching and two-view geometry on exactly the
    listed pairs, accumulating results alongside any existing matches (so this
    augments sequential/vocab matching rather than replacing it).
    """
    cmd = [
        "colmap", "matches_importer",
        "--database_path", str(database_path),
        "--match_list_path", str(pairs_path),
        "--match_type", "pairs",
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.guided_matching", "1",
        # Same tight verification as the sequential loop-closure edges, so the
        # covisibility pairs that survive are reliable.
        "--SiftMatching.max_ratio", "0.8",
        "--TwoViewGeometry.min_num_inliers", "15",
    ]
    print("Running COLMAP matches_importer on ARKit-covisibility pairs...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Covisibility matching completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Covisibility matching failed: {e}")
        return False


def report_view_graph_connectivity(database_path, label=""):
    """Print connected-component stats of the verified view graph.

    This is the measurement that matters (the "192/673" number): after matching,
    how many images are actually connected? Builds a graph from
    two_view_geometries with verified inliers and runs union-find. Run with and
    without --use_covisibility to see the connectivity gain.
    """
    conn = sqlite3.connect(str(database_path))
    try:
        image_ids = [image_id for (image_id,) in conn.execute("SELECT image_id FROM images")]
        # two_view_geometries.pair_id packs the two image ids; rows>0 means the
        # pair has verified inlier matches (an edge in the view graph).
        edges = list(conn.execute(
            "SELECT pair_id FROM two_view_geometries WHERE rows > 0"))
    finally:
        conn.close()

    n = len(image_ids)
    if n == 0:
        print("View graph: no images in database.")
        return

    parent = {img: img for img in image_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    MAX_IMAGE_ID = 2147483647
    n_edges = 0
    for (pair_id,) in edges:
        id1 = pair_id % MAX_IMAGE_ID
        id2 = pair_id // MAX_IMAGE_ID
        if id1 in parent and id2 in parent:
            union(id1, id2)
            n_edges += 1

    comps = {}
    for img in image_ids:
        r = find(img)
        comps[r] = comps.get(r, 0) + 1
    sizes = sorted(comps.values(), reverse=True)
    largest = sizes[0] if sizes else 0

    tag = f" [{label}]" if label else ""
    print(f"View graph connectivity{tag}: {len(comps)} component(s) over {n} images "
          f"({n_edges} verified edges); largest component {largest}/{n} "
          f"({100.0 * largest / n:.1f}%).")
    if len(sizes) > 1:
        print(f"  component sizes: {sizes[:10]}" + (" ..." if len(sizes) > 10 else ""))
