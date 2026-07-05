# Reconstruction Pipeline — Cleanup & Hybrid Plan

Status: **draft / planning**. Owner: (kobejean). Last updated: 2026-07-05.

This document records what we learned while getting the COLMAP → map pipeline to
work on park-scale ARKit capture, the keep/remove decisions to simplify the code
before merge, and the design for a future **hybrid** reconstruction path.

Reference session used throughout: `input/meguru-park-itchy-sequential`
(765 images, 673 ARKit frames; the last 92 image files have **no** ARKit pose).

---

## 1. Findings (empirical — do not re-litigate without new data)

1. **Vision-only SfM cannot cover this data.** Matches are rich (median 331
   inliers/pair, 492 adjacent pairs), but two-view *relative poses* are
   inconsistent (park foliage / low parallax). GLOMAP's rotation-averaged
   connected component is only **192 / 673** images; incremental COLMAP dies at a
   degenerate initial pair (2-image, 5-point model).

2. **Init-pair tuning can't exceed view-graph connectivity.** `init_min_tri_angle`,
   `init_max_forward_motion`, etc. address the symptom (degenerate init), not the
   disease. GLOMAP (no init step at all) still tops out at 192, so ~192/673 is the
   hard ceiling for *any* vision-only method here.

3. **`pose_prior_mapper` fails even with correct, tight priors.** Re-tested with
   priors rewritten in the corrected convention and `prior_position_std ≈ 0.3`,
   robust loss: still a 2-image init. COLMAP's `pose_prior_mapper` uses priors only
   *after* a model is growing (as a BA regularizer); initialization is pure vision,
   so it dies before priors matter. → **dead end for this data.**

4. **Odometry cannot be injected into COLMAP/GLOMAP.** GLOMAP re-estimates relative
   poses from feature matches (even with `--skip_view_graph_calibration`) and
   filters any edge with `< 30` inliers. Injected pose-only odometry edges (1
   placeholder match) are discarded — connectivity moved only 192 → 232. **Odometry
   belongs in the g2o BA** (`ColmapRefiner::optimize()` already adds it as SE3
   relative-pose edges), not in the SfM view graph.

5. **Multi-model COLMAP** (`multiple_models 1`) yields ~11 fragments covering
   **371 / 673** unique frames with **longer tracks (3.0–4.9, vs 2.5 for seeding)**.
   The remaining 302 frames are visual dust. → basis for the hybrid (§4).

6. **ARKit↔COLMAP convention was wrong.** The old code applied a *world-axis* Y/Z
   flip; the correct transform is a *camera-axis* flip on the inverted matrix:
   `R_w2c = diag(1,-1,-1) · R_c2wᵀ`, `t = -R_w2c · C`. The fix spans Python
   (`colmap_pose.py`, `arkit_integration.py`, `database_operations.py`) **and** C++
   (`colmap_database.cpp::colmapPoseToMatrix` + landmark/camera flips). Validated by
   round-trip (float precision) and by the refiner's g2o BA converging to 0.39–0.42px
   (a wrong convention diverges).

## 2. What works (the shipping pipeline)

```
colmap.py --use_arkit_poses            (matches already in DB, e.g. via --use_sequential)
  ├─ create_arkit_seed_model           seed COLMAP model from ARKit poses (fixed)
  ├─ colmap point_triangulator         landmarks vs fixed poses; 12px merge threshold
  └─ (skip model alignment — already in ARKit coords)
        ↓  colmap/poses_txt + sparse/0
lar_refine_colmap <session>            g2o BA: landmark reprojection + ARKit odometry
        ↓
  → 673 frames posed, ~31k usable landmarks @ 0.42px  (12px merge; 15.7k @ 0.39px at 6px)
```

Why this is the correct decomposition: seeding solves connectivity (poses from
ARKit, no view graph needed); COLMAP provides structure from matches only; g2o
fuses odometry where pose-only constraints actually belong.

## 3. Keep / Remove / Decide

### Keep
- **`--use_arkit_poses`** + `create_arkit_seed_model` + `run_arkit_pose_triangulation`
  (the working reconstruction path).
- **ARKit↔COLMAP convention fix** (Python + C++) — correctness-critical; the C++
  refiner and Python must stay on the same convention.
- **`--use_sequential`** + bundled `vocab_tree.bin` + `resolve_vocab_tree_path`
  (produces the strong match graph; loop-closure tuning validated).
- **12px triangulation merge** in the ARKit-pose path (≈2× usable landmarks vs 6px,
  same reprojection error).
- **g2o odometry** in `ColmapRefiner::optimize()` (correct home for odometry).

### Remove (simplify before merge)
- **`--use_pose_prior`** + `run_colmap_pose_prior_mapping` — validated dead end
  (finding #3). Delete the flag, the function, and drop commit `8787d67` on rebase.
- **`insert_arkit_odometry` / `insert_two_view_geometries_from_arkit`** — validated
  ineffective (finding #4); odometry is handled in g2o. Remove the pipeline Step 4
  call and the function.
- **pose_priors writing in `create_colmap_database`** — only consumed by
  `pose_prior_mapper`, which we're removing. Drop the `pose_priors` INSERT to keep
  DB setup simple (leave a note in case a future prior-aware mapper wants it).

### Decide (not blocking)
- **`--use_vocab_tree`** (standalone vocab-tree matching) — largely subsumed by
  `--use_sequential`. Keep as a thin option or remove. Low stakes.
- **Default `colmap mapper` / `--use_glomap`** — fail on *this* data but are valid
  general COLMAP paths (and useful for comparison / other datasets). **Keep**, but
  document that they are not expected to fully reconstruct park-scale capture.

## 4. Future work — Hybrid (`--use_hybrid`)

Goal: use *visual* poses where SfM is reliable (longer tracks → more usable
landmarks) and *ARKit* poses elsewhere, in one metric frame.

```
1. colmap mapper --multiple_models 1        → N visual sub-reconstructions (371 frames)
2. model_aligner each sub-model → ARKit      (sim3 from ARKit ref positions)
3. build ONE hybrid seed model, per frame:
     pose = aligned visual pose   if frame in a (large) sub-model
     pose = ARKit pose            otherwise
4. point_triangulator over all 673 hybrid poses
5. lar_refine_colmap (unchanged)
```

Rationale: the win is at *triangulation* time — visually-consistent poses give
≥3-view tracks that survive the `sightings >= 3` cull; the refiner cannot resurrect
a landmark that was never triangulated multi-view. Strictly a superset of seeding
(the 302 uncovered frames stay as good as today).

Risks / mitigations:
- **Small-fragment sim3 alignment is noisy** → only trust sub-models above a size
  threshold (e.g. ≥30 frames: models covering ~262 frames), seed the rest.
- **Seam consistency** between visual and seeded regions → the g2o odometry BA
  smooths it; acceptable at ARKit precision.
- Only ~7 frames appear in >1 sub-model → de-dup is trivial (pick one).

Decision gate before wiring in: prototype in scratch, compare usable-landmark count
and track length against the seeding baseline (31k @ 0.42px). Wire in only on a
clear win.

## 5. Merge / cleanup action items

Branch today: `colmap-sequential-matcher` (PR #70) mixes three concerns.

- [ ] Commit the 12px triangulation merge change (currently uncommitted in
      `colmap.py`).
- [ ] Apply §3 removals (`--use_pose_prior`, odometry injection, pose_priors write).
- [ ] Split into reviewable PRs:
      - **PR A** — sequential matcher + bundled vocab tree (`401b321`–`609b092`).
      - **PR B** — convention fix + `--use_arkit_poses` + 12px (`38b7809` + new),
        onto `origin/main`; includes the C++ `colmap_database.cpp` change.
      - Drop `8787d67` (pose_prior) during the rebase.
- [ ] Repo hygiene: `.gitignore` `.claude/` and `debug.txt` (g2o dump); `git rm
      --cached` the committed `__pycache__/*.pyc` (already covered by `*.pyc`).
- [ ] Update `docs/RECONSTRUCTION.md` to point at `--use_arkit_poses` +
      `lar_refine_colmap` as the park-scale path.

## 6. Open questions
- Iterative refinement: does `seed → refine → re-triangulate against refined poses`
  lengthen tracks enough to rival the hybrid, with far less complexity? (untested)
- `isUseable()` is `sightings >= 3`; is 3 the right localization bar, or should it
  be data-driven?
