#!/usr/bin/env python3
"""Train a (optionally semantic) 3D Gaussian Splatting model from a LAR/COLMAP model.

Engine: gsplat with the **MCMC strategy**, whose ``--cap-max`` hard-limits the Gaussian
count. That cap is the whole point for park scale: it bounds VRAM regardless of scene
size, trading detail for a coarse-but-complete model that fits an 8 GB laptop GPU.

  # RGB only, coarse park model, half-res images, 300k Gaussian cap
  uv run --extra gsplat python train.py \
      --model  ../../input/<session>/colmap/poses_txt \
      --out    ../../output/<session>-gsplat \
      --data-factor 2 --cap-max 300000 --max-steps 30000

  # Semantic 3DGS: distil a segmenter into per-Gaussian class logits
  uv run --extra gsplat --extra segmentation python train.py ... \
      --semantic --segmenter mask2former-large

Semantic training is two-phase (the reliable, canonical recipe):
  Phase 1 (--max-steps) trains RGB + geometry with MCMC densification -- identical to a
    plain RGB run, no semantic field involved.
  Phase 2 (--sem-steps) attaches a fresh semantic field to the CONVERGED Gaussians and
    trains only it, with geometry detached (MCMC off). Cross-entropy distils the 2D masks
    onto fixed supports, so semantics can never move or degrade the reconstruction.

See README.md. Requires a CUDA toolchain (nvcc) matching the installed torch, because
gsplat compiles kernels on first import.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

import model as gmodel
from colmap_dataset import ColmapDataset


# --------------------------------------------------------------------------- metrics

def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM for (1, C, H, W) images in [0, 1]."""
    c = x.shape[1]
    w = _gaussian_window(size, sigma, x.device).expand(c, 1, size, size)
    pad = size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return s.mean()


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    return 100.0 if mse == 0 else -10.0 * math.log10(mse)


# ------------------------------------------------------------------------------- train

def build_optimizers(params, scene_scale: float, sem: bool):
    """One Adam per attribute (gsplat strategies edit each optimizer's state in place)."""
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    if sem:
        lrs["sem"] = 1e-2
    return {
        name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}],
                               eps=1e-15, betas=(0.9, 0.999))
        for name, lr in lrs.items()
    }, lrs


def train(args):
    assert torch.cuda.is_available(), "gsplat training needs a CUDA GPU"
    device = "cuda"
    torch.manual_seed(0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.images) if args.images else Path(args.model).parent

    print(f"gsplat 3DGS training -> {out}")
    ds = ColmapDataset(args.model, image_dir, data_factor=args.data_factor, limit=args.limit)

    # Optionally subsample the init cloud. MCMC's --cap-max only bounds *growth*, so if
    # the sparse cloud already exceeds the cap the model can never get coarser than it.
    # Subsampling to <= cap_max makes the cap a real VRAM budget for park scale.
    points_xyz, points_rgb = ds.points_xyz, ds.points_rgb
    init_max = args.init_points if args.init_points else args.cap_max
    if len(points_xyz) > init_max:
        sel = np.random.default_rng(0).choice(len(points_xyz), init_max, replace=False)
        points_xyz, points_rgb = points_xyz[sel], points_rgb[sel]
        print(f"subsampled init cloud {len(ds.points_xyz)} -> {init_max} points (<= cap_max)")

    # Semantic supervision (Phase 2): cache masks up front so a misconfigured segmenter
    # fails fast, not after a multi-hour RGB phase.
    masks = None
    num_classes = None
    color_lut = None
    ignore_index = 0
    if args.semantic:
        import semantic as sem_mod
        num_classes = sem_mod.NUM_CLASSES
        color_lut = sem_mod.SEMANTIC_COLOR_LUT
        ignore_index = sem_mod.IGNORE_INDEX
        # Namespace the cache by segmenter so switching backends (oneformer ->
        # mask2former-large) against the same --out never silently reuses stale masks.
        store = sem_mod.build_mask_cache(
            [v.name for v in ds.views], image_dir, out / "masks" / args.segmenter,
            segmenter_kind=args.segmenter, overwrite=args.overwrite_masks,
        )
        masks = [
            torch.from_numpy(sem_mod.load_mask(store, v.name, v.width, v.height).astype(np.int64))
            for v in ds.views
        ]
        print(f"semantic enabled: {num_classes} classes, segmenter={args.segmenter}")

    # Phase 1 Gaussians carry no semantic field -- this phase is identical to a plain,
    # well-tested RGB run. The semantic field is attached afterwards (Phase 2).
    params = gmodel.init_gaussians(
        points_xyz, points_rgb, sh_degree=args.sh_degree, num_classes=None, device=device,
    )
    print(f"initialised {params['means'].shape[0]} Gaussians (sh_degree={args.sh_degree})")

    optimizers, lrs = build_optimizers(params, ds.scene_scale, sem=False)
    # Exponential decay of the means lr (positions settle as training progresses).
    means_gamma = (0.01) ** (1.0 / args.max_steps)

    strategy = MCMCStrategy(
        cap_max=args.cap_max,
        refine_start_iter=args.refine_start,
        refine_stop_iter=int(args.max_steps * 0.83),
        refine_every=args.refine_every,
        min_opacity=0.005,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state()

    # Pre-stage per-view camera tensors.
    viewmats = [torch.from_numpy(v.viewmat).float().to(device) for v in ds.views]
    Ks = [torch.from_numpy(v.K).float().to(device) for v in ds.views]

    lam = args.ssim_weight
    n_views = len(ds.views)
    t0 = time.time()

    # ===================== Phase 1: RGB + geometry (MCMC densification) =====================
    print(f"\nPhase 1 — RGB/geometry, {args.max_steps} steps")
    order = np.random.permutation(n_views)
    for step in range(args.max_steps):
        idx = int(order[step % n_views])
        if step % n_views == 0 and step > 0:
            order = np.random.permutation(n_views)

        v = ds.views[idx]
        gt = torch.from_numpy(ds.images[idx]).to(device)  # (H, W, 3)
        H, W = v.height, v.width

        colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N, K, 3)
        renders, alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmats[idx][None],
            Ks=Ks[idx][None],
            width=W, height=H,
            sh_degree=args.sh_degree,
            packed=True,
            render_mode="RGB",
        )
        rgb = renders[0].clamp(0.0, 1.0)  # (H, W, 3)

        l1 = F.l1_loss(rgb, gt)
        ssim_val = ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1.0 - lam) * l1 + lam * (1.0 - ssim_val)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        cur_means_lr = lrs["means"] * (means_gamma ** step)
        for g in optimizers["means"].param_groups:
            g["lr"] = cur_means_lr
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, lr=cur_means_lr)

        if step % args.log_every == 0 or step == args.max_steps - 1:
            n = params["means"].shape[0]
            rate = (step + 1) / (time.time() - t0)
            print(f"[P1 {step:6d}/{args.max_steps}] loss={loss.item():.4f} "
                  f"psnr={psnr(rgb, gt):.2f} gaussians={n} {rate:.1f} it/s")
        if args.preview_every and (step % args.preview_every == 0 or step == args.max_steps - 1):
            _save_preview(rgb, out / f"preview_{step:06d}.png")
        # Periodic geometry checkpoint so a long RGB phase survives a late crash/OOM.
        if args.save_every and step > 0 and step % args.save_every == 0:
            gmodel.export_gaussian_ply(params, out / "point_cloud.ply")

    gmodel.export_gaussian_ply(params, out / "point_cloud.ply")
    print(f"Phase 1 done in {time.time() - t0:.1f}s")

    # ============= Phase 2: semantic field on FROZEN geometry (canonical recipe) =============
    # Attach a fresh semantic field to the final Gaussians and train only it, with the
    # geometry inputs detached: cross-entropy can label the supports but can never move
    # or degrade them, and MCMC is off. This is the reliable LangSplat/Feature-3DGS recipe.
    if args.semantic:
        n = params["means"].shape[0]
        params["sem"] = torch.nn.Parameter(torch.zeros(n, num_classes, device=device))
        sem_opt = torch.optim.Adam([params["sem"]], lr=args.sem_lr, eps=1e-15)

        geo = dict(  # frozen snapshot of the converged geometry
            means=params["means"].detach(),
            quats=params["quats"].detach(),
            scales=torch.exp(params["scales"]).detach(),
            opacities=torch.sigmoid(params["opacities"]).detach(),
        )

        print(f"\nPhase 2 — semantic ({args.segmenter}), {args.sem_steps} steps, geometry frozen")
        t1 = time.time()
        order = np.random.permutation(n_views)
        for step in range(args.sem_steps):
            idx = int(order[step % n_views])
            if step % n_views == 0 and step > 0:
                order = np.random.permutation(n_views)
            v = ds.views[idx]

            sem_render, _, _ = rasterization(
                means=geo["means"], quats=geo["quats"], scales=geo["scales"],
                opacities=geo["opacities"], colors=params["sem"],
                viewmats=viewmats[idx][None], Ks=Ks[idx][None],
                width=v.width, height=v.height, sh_degree=None, packed=True, render_mode="RGB",
            )
            logits = sem_render.permute(0, 3, 1, 2)          # (1, C, H, W)
            target = masks[idx].to(device)[None]             # (1, H, W)
            ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
            ce.backward()
            sem_opt.step()
            sem_opt.zero_grad(set_to_none=True)

            if step % args.log_every == 0 or step == args.sem_steps - 1:
                rate = (step + 1) / (time.time() - t1)
                print(f"[P2 {step:6d}/{args.sem_steps}] sem_ce={ce.item():.4f} {rate:.1f} it/s")
            if args.save_every and step > 0 and step % args.save_every == 0:
                gmodel.export_semantic(params, out / "point_cloud", color_lut)

    save_outputs(params, out, args.semantic, color_lut, num_classes, label="final")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"\n✅ done in {time.time() - t0:.1f}s -> {out}")


def save_outputs(params, out: Path, semantic: bool, color_lut, num_classes, label: str):
    """Write point_cloud.ply (+ semantic labels/ply if enabled). Overwrites in place."""
    n = gmodel.export_gaussian_ply(params, out / "point_cloud.ply")
    msg = f"[{label}] exported {n} Gaussians -> {out}/point_cloud.ply"
    if semantic:
        labels = gmodel.export_semantic(params, out / "point_cloud", color_lut)
        counts = np.bincount(labels, minlength=num_classes)
        msg += (f"\n  semantic labels -> point_cloud_labels.npy (+ _semantic.ply); "
                f"per-class counts {counts.tolist()}")
    print(msg)


@torch.no_grad()
def _save_preview(rgb: torch.Tensor, path: Path) -> None:
    img = (rgb.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=None,
                   help="LAR session name: fills --model/--images/--out from the canonical "
                        "layout (script/lar_session.py). Explicit flags override.")
    p.add_argument("--model", default=None,
                   help="COLMAP text model dir (…/colmap/sparse/0 or …/poses_txt)")
    p.add_argument("--images", default=None,
                   help="image dir (default: session images, else parent of --model)")
    p.add_argument("--out", default=None, help="output dir for plys/previews/config")
    p.add_argument("--data-factor", type=int, default=2,
                   help="downscale images by this integer factor (default: 2; use >=2 at park scale)")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N images (smoke tests)")

    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--cap-max", type=int, default=1_000_000,
                   help="hard cap on Gaussian count (MCMC). Lower = coarser + less VRAM.")
    p.add_argument("--init-points", type=int, default=None,
                   help="randomly subsample the sparse init cloud to this many points "
                        "(default: min(len, cap_max)). Needed to get a model coarser than "
                        "the COLMAP cloud, since --cap-max only bounds growth.")
    p.add_argument("--sh-degree", type=int, default=1,
                   help="SH degree for view-dependent colour (default 1 = coarse/cheap)")
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--refine-start", type=int, default=500)
    p.add_argument("--refine-every", type=int, default=100)

    p.add_argument("--semantic", action="store_true",
                   help="train a per-Gaussian semantic head distilled from a 2D segmenter")
    p.add_argument("--segmenter", default="mask2former-large",
                   help="semantic_bev segmenter kind (default: mask2former-large; also "
                        "oneformer[-large]/mask2former/clipseg)")
    p.add_argument("--sem-steps", type=int, default=7000,
                   help="Phase 2 steps: train the semantic field on the frozen geometry")
    p.add_argument("--sem-lr", type=float, default=1e-2, help="learning rate for semantic logits")
    p.add_argument("--overwrite-masks", action="store_true",
                   help="re-run the segmenter even if masks are cached")

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--preview-every", type=int, default=0,
                   help="save a render preview every N steps (0 = only at the end)")
    p.add_argument("--save-every", type=int, default=5000,
                   help="checkpoint the .ply (+ labels) every N steps so a long train "
                        "survives a late crash (default 5000; 0 = only at the end)")

    args = p.parse_args()
    _resolve_session(args, p)
    if args.preview_every == 0:
        args.preview_every = args.max_steps  # still dump one at the end
    train(args)


def _resolve_session(args, parser):
    """Fill --model/--images/--out from --session (canonical layout); explicit flags win."""
    if args.session:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from lar_session import Session
        s = Session(args.session)
        args.model = args.model or str(s.best_model())
        args.images = args.images or str(s.images)
        args.out = args.out or str(s.gsplat_out(semantic=args.semantic))
        print(f"session '{args.session}':\n  model  = {args.model}\n"
              f"  images = {args.images}\n  out    = {args.out}")
    missing = [m for m in ("model", "out") if not getattr(args, m)]
    if missing:
        parser.error("need --session or explicit " + " ".join(f"--{m}" for m in missing))


if __name__ == "__main__":
    main()
