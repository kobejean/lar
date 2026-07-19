"""Probe the LingBot-Vision ViT backbone on park frames.

Two questions, two tasks:

  pca      — are the frozen patch features spatially meaningful? PCA the tokens to
             3 channels (shared basis across frames) and render RGB | feature-PCA.
             The fast "are these worth building on" eyeball test.

  linprobe — do the frozen features *linearly* separate OUR park taxonomy? Train a
             single linear layer (patch feature -> Klass) on Mask2Former pseudo-labels
             (no manual labels), evaluate held-out mIoU, and render pred vs pseudo.
             If a linear head already works, a real custom head is cheap.

Backbone: robbyant/lingbot-vision (Apache-2.0 code + weights). Frozen, eval-only.

Optionally feature-upsample the frozen tokens with AnyUp (wimmerth/anyup,
CC-BY-4.0) before probing — a feature-agnostic, post-hoc upsampler that sharpens
the coarse patch-16 grid to object boundaries without touching the backbone.
Add ``--upsample anyup [--variant large] [--up-size 128]``.

Run (from repo root):
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/probe.py pca      --session maguro-park-after-itchy --num 6
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/probe.py linprobe --session maguro-park-after-itchy --num 24
  # AnyUp-upsampled large backbone:
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/probe.py linprobe --variant large --upsample anyup --up-size 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# --- make sibling script packages importable (taxonomy, segmentation, lar_session) ---
_HERE = Path(__file__).resolve().parent
_SCRIPT_ROOT = _HERE.parent
sys.path.insert(0, str(_SCRIPT_ROOT))
sys.path.insert(0, str(_SCRIPT_ROOT / "semantic_bev"))

from lar_session import Session  # noqa: E402
from lingbot_vision import extract_patch_tokens, load_image, load_pretrained_backbone  # noqa: E402


# ---------------------------------------------------------------------------
# shared: load backbone + sample frames + extract a feature grid per frame
# ---------------------------------------------------------------------------
def sample_frames(session: Session, num: int) -> list[Path]:
    paths = sorted(session.images.glob("*_image.jpeg"))
    if not paths:
        raise SystemExit(f"no *_image.jpeg under {session.images}")
    idx = np.linspace(0, len(paths) - 1, num).round().astype(int)
    return [paths[i] for i in idx]


# DINOv2 (facebookresearch/dinov2, Apache-2.0). Same ImageNet normalization and
# the same "x_norm_patchtokens" key as LingBot, so it drops into feature_grid
# behind the ``kind`` dispatch below. patch_size=14 (vs LingBot's 16).
_DINOV2_HUB = {
    "small": "dinov2_vits14", "base": "dinov2_vitb14",
    "large": "dinov2_vitl14", "giant": "dinov2_vitg14",
}


def load_backbone(variant: str, kind: str = "lingbot"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if kind == "lingbot":
        backbone, embed_dim = load_pretrained_backbone(variant=variant, device=device, dtype=dtype)
    elif kind == "dinov2":
        backbone = torch.hub.load("facebookresearch/dinov2", _DINOV2_HUB[variant], verbose=False)
        backbone = backbone.to(device).to(dtype).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        embed_dim = backbone.embed_dim
    else:
        raise ValueError(f"unknown backbone kind: {kind!r}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return backbone, embed_dim, device, dtype


@torch.no_grad()
def extract_tokens(backbone, img_norm, device: str, dtype, kind: str):
    """Backbone-agnostic patch tokens: (tokens [B,N,C], (h,w)).

    LingBot exposes them via ``extract_patch_tokens``; DINOv2 via
    ``forward_features(...)['x_norm_patchtokens']``. Both consume the same
    ImageNet-normalized batch and return the same grid convention.
    """
    if kind == "lingbot":
        return extract_patch_tokens(backbone, img_norm, device, dtype)
    x = img_norm.to(device).to(dtype)
    ps = backbone.patch_size
    _, _, H, W = x.shape
    h, w = H // ps, W // ps
    use_ac = device.startswith("cuda") and dtype != torch.float32
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_ac):
        out = backbone.forward_features(x)
    return out["x_norm_patchtokens"], (h, w)


# --- AnyUp feature upsampler (wimmerth/anyup, CC-BY-4.0) -------------------
# Feature-agnostic: consumes any encoder's low-res token grid + the guidance
# image and produces edge-aligned high-res features WITHOUT touching the frozen
# backbone. Applied post-hoc, so it composes with the linprobe head unchanged.
_ANYUP_CACHE = {}


def load_anyup(ckpt: str, src: str, device: str):
    """Lazily build the AnyUp upsampler (frozen, eval). Cached per checkpoint.

    ckpt='multi' -> anyup_multi_backbone, trained on several backbones; best
    generalization to an unseen encoder like LingBot-Vision. ckpt='paper' was
    trained on DINOv2-S only. Weights auto-download to the torch hub cache.
    """
    key = (ckpt, src, device)
    if key in _ANYUP_CACHE:
        return _ANYUP_CACHE[key]
    if src and src not in sys.path:
        sys.path.insert(0, src)
    from anyup.model import AnyUp  # noqa: E402
    urls = {
        "multi": "https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth",
        "paper": "https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth",
    }
    model = AnyUp().to(device).eval()
    sd = torch.hub.load_state_dict_from_url(urls[ckpt], map_location=device, progress=True)
    model.load_state_dict(sd)
    _ANYUP_CACHE[key] = model
    return model


def report_vram(tag: str, device: str):
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{tag}] peak VRAM = {peak:.2f} GB")


@torch.no_grad()
def feature_grid(backbone, path: Path, size: int, mode: str, device: str, dtype,
                 upsampler=None, up_size: int = 0, kind: str = "lingbot"):
    """Return (feat [G,G,C] float32, rgb [H,W,3] uint8, (G,G)).

    Native grid is (h,w) = size/patch_size. With an AnyUp ``upsampler`` the
    frozen tokens are feature-upsampled to (up_size, up_size), guided by the
    same ImageNet-normalized image the backbone saw.
    """
    img_norm, rgb, _ = load_image(str(path), size=size, patch_size=backbone.patch_size, mode=mode)
    tokens, (h, w) = extract_tokens(backbone, img_norm, device, dtype, kind)
    if upsampler is None:
        feat = tokens[0].float().cpu().numpy().reshape(h, w, -1)
        return feat, rgb, (h, w)
    lr = tokens[0].float().reshape(1, h, w, -1).permute(0, 3, 1, 2).contiguous()  # [1,C,h,w]
    guide = img_norm.to(device).float()
    hr = upsampler(guide, lr, output_size=(up_size, up_size), q_chunk_size=256)  # [1,C,up,up]
    feat = hr[0].permute(1, 2, 0).float().cpu().numpy()  # [up,up,C]
    return feat, rgb, (up_size, up_size)


def _upsample(grid: np.ndarray, size: int, interp=cv2.INTER_NEAREST) -> np.ndarray:
    return cv2.resize(grid, (size, size), interpolation=interp)


def _tag(args) -> str:
    """Output-filename suffix so baseline and AnyUp runs don't overwrite."""
    return "" if args.upsample == "none" else f"_{args.upsample}{args.up_size}"


def _bb(args) -> str:
    """Backbone prefix for output names; empty for the default (lingbot)."""
    return "" if args.backbone == "lingbot" else f"{args.backbone}_"


# ---------------------------------------------------------------------------
# task: pca
# ---------------------------------------------------------------------------
def robust_norm(x: np.ndarray, lo=2.0, hi=98.0) -> np.ndarray:
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-8), 0, 1)


def run_pca(args):
    backbone, embed_dim, device, dtype = load_backbone(args.variant, args.backbone)
    up = load_anyup(args.anyup_ckpt, args.anyup_src, device) if args.upsample == "anyup" else None
    up_size = args.up_size if up is not None else 0
    frames = sample_frames(Session(args.session), args.num)
    print(f"[pca] backbone={args.backbone} variant={args.variant} embed_dim={embed_dim} "
          f"frames={len(frames)} size={args.size} upsample={args.upsample}"
          + (f" up_size={up_size}" if up else ""))

    feats, rgbs, grids = [], [], []
    for p in frames:
        f, rgb, hw = feature_grid(backbone, p, args.size, args.mode, device, dtype, up, up_size, args.backbone)
        feats.append(f); rgbs.append(rgb); grids.append(hw)

    # shared PCA basis across every patch of every frame -> consistent colours
    stack = np.concatenate([f.reshape(-1, embed_dim) for f in feats], axis=0)
    mean = stack.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(stack - mean, full_matrices=False)
    basis = vt[:3]  # (3, C)

    proj_all = (stack - mean) @ basis.T
    lo = np.percentile(proj_all, 2, axis=0)
    hi = np.percentile(proj_all, 98, axis=0)

    tiles = []
    for f, rgb, (h, w) in zip(feats, rgbs, grids):
        proj = (f.reshape(-1, embed_dim) - mean) @ basis.T
        proj = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1).reshape(h, w, 3)
        pca_img = _upsample((proj * 255).astype(np.uint8), rgb.shape[0], cv2.INTER_NEAREST)
        row = np.concatenate([rgb, pca_img], axis=1)  # RGB | PCA
        tiles.append(row)

    sheet = np.concatenate(tiles, axis=0)
    out = Path(args.out or (Session(args.session).root / "output" / f"{args.session}-backbone"))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"pca_{_bb(args)}{args.variant}{_tag(args)}.png"
    cv2.imwrite(str(dst), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[pca] wrote {dst}  (left=RGB, right=feature-PCA; shared basis)")


# ---------------------------------------------------------------------------
# task: linprobe  (frozen features -> linear layer -> our Klass, vs Mask2Former)
# ---------------------------------------------------------------------------
def patch_labels(mask_hw: np.ndarray, h: int, w: int) -> np.ndarray:
    """Majority Klass per patch block -> (h*w,) int."""
    H, W = mask_hw.shape
    ph, pw = H // h, W // w
    out = np.empty(h * w, dtype=np.int64)
    k = 0
    for i in range(h):
        for j in range(w):
            block = mask_hw[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw].ravel()
            out[k] = np.bincount(block).argmax()
            k += 1
    return out


def run_linprobe(args):
    from taxonomy import Klass, color_lut
    from segmentation import make_segmenter

    backbone, embed_dim, device, dtype = load_backbone(args.variant, args.backbone)
    up = load_anyup(args.anyup_ckpt, args.anyup_src, device) if args.upsample == "anyup" else None
    up_size = args.up_size if up is not None else 0
    seg = make_segmenter("mask2former-large")
    frames = sample_frames(Session(args.session), args.num)
    n_val = max(1, int(round(len(frames) * args.val_frac)))
    val_set = set(frames[-n_val:])
    print(f"[linprobe] backbone={args.backbone} variant={args.variant} embed_dim={embed_dim} "
          f"frames={len(frames)} (val={n_val}) size={args.size} upsample={args.upsample}"
          + (f" up_size={up_size}" if up else ""))

    Xtr, ytr, val = [], [], []  # val: list of (feat[G,G,C], pseudo(G*G), rgb, (G,G))
    for p in frames:
        f, rgb, (h, w) = feature_grid(backbone, p, args.size, args.mode, device, dtype, up, up_size, args.backbone)
        # Segment the exact image the backbone saw (snapped size) so M2F pseudo-labels
        # and features align — patch-14 backbones snap 512->504, patch-16 stay at 512.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        y = patch_labels(seg.segment(bgr), h, w)
        if p in val_set:
            val.append((f, y, rgb, (h, w)))
        else:
            Xtr.append(f.reshape(-1, embed_dim)); ytr.append(y)

    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    keep = ytr != int(Klass.UNKNOWN)              # don't train on "nothing"
    Xtr, ytr = Xtr[keep], ytr[keep]
    n_klass = int(max(Klass)) + 1

    # standardise on train stats
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    yn = torch.tensor(ytr, device=device)

    head = torch.nn.Linear(embed_dim, n_klass).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    head.train()
    for step in range(args.steps):
        opt.zero_grad()
        loss = lossf(head(Xn), yn)
        loss.backward(); opt.step()
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(f"  step {step+1}/{args.steps}  loss={loss.item():.3f}")
    head.eval()

    # evaluate held-out mIoU + render
    lut = np.array(color_lut(), dtype=np.uint8)
    inter = np.zeros(n_klass); union = np.zeros(n_klass); correct = total = 0
    tiles = []
    for f, y, rgb, (h, w) in val:
        Xv = torch.tensor((f.reshape(-1, embed_dim) - mu) / sd, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = head(Xv).argmax(1).cpu().numpy()
        m = y != int(Klass.UNKNOWN)
        correct += int((pred[m] == y[m]).sum()); total += int(m.sum())
        for c in range(n_klass):
            pc, yc = pred == c, y == c
            inter[c] += int((pc & yc).sum()); union[c] += int((pc | yc).sum())
        pred_img = _upsample(lut[pred.reshape(h, w)], rgb.shape[0], cv2.INTER_NEAREST)
        pseudo_img = _upsample(lut[y.reshape(h, w)], rgb.shape[0], cv2.INTER_NEAREST)
        tiles.append(np.concatenate([rgb, pseudo_img, pred_img], axis=1))  # RGB | M2F | linear

    iou = inter / np.maximum(union, 1)
    present = union > 0
    print(f"\n[linprobe] val patch-acc = {correct/max(total,1):.3f}   "
          f"mIoU(present) = {iou[present].mean():.3f}")
    print("  per-class IoU (present classes):")
    for c in np.where(present)[0]:
        print(f"    {Klass(c).name:10s} {iou[c]:.3f}")

    out = Path(args.out or (Session(args.session).root / "output" / f"{args.session}-backbone"))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"linprobe_{_bb(args)}{args.variant}{_tag(args)}.png"
    cv2.imwrite(str(dst), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
    print(f"[linprobe] wrote {dst}  (left=RGB, mid=Mask2Former pseudo-label, right=linear head)")
    report_vram("linprobe", device)

    if args.save_head:
        sp = Path(args.save_head)
        sp.parent.mkdir(parents=True, exist_ok=True)
        # Everything LingBotSegmenter needs to reproduce this head at inference.
        torch.save({
            "head": {k: v.cpu() for k, v in head.state_dict().items()},
            "mu": mu, "sd": sd,
            "backbone": args.backbone,
            "variant": args.variant, "embed_dim": embed_dim, "n_klass": n_klass,
            "size": args.size, "mode": args.mode,
            "upsample": args.upsample, "up_size": up_size,
            "anyup_ckpt": args.anyup_ckpt, "anyup_src": args.anyup_src,
        }, sp)
        print(f"[linprobe] saved head -> {sp}")


# ---------------------------------------------------------------------------
# task: segment  (drive make_segmenter("lingbot") end-to-end from a saved head)
# ---------------------------------------------------------------------------
def run_segment(args):
    from taxonomy import color_lut
    from segmentation import make_segmenter

    seg = make_segmenter("lingbot", ckpt=args.head)
    ref = make_segmenter("mask2former-large") if args.ref else None
    frames = sample_frames(Session(args.session), args.num)
    lut = np.array(color_lut(), dtype=np.uint8)
    print(f"[segment] head={args.head} frames={len(frames)} size={args.size} ref={bool(ref)}")

    tiles = []
    for p in frames:
        bgr = cv2.resize(cv2.imread(str(p)), (args.size, args.size))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        row = [rgb]
        if ref is not None:
            row.append(lut[ref.segment(bgr)])
        row.append(lut[seg.segment(bgr)])
        tiles.append(np.concatenate(row, axis=1))  # RGB | [M2F] | lingbot

    out = Path(args.out or (Session(args.session).root / "output" / f"{args.session}-backbone"))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "segment_lingbot.png"
    cv2.imwrite(str(dst), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
    cols = "RGB | Mask2Former | lingbot" if ref is not None else "RGB | lingbot"
    print(f"[segment] wrote {dst}  ({cols})")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)

    def common(p):
        p.add_argument("--session", default="maguro-park-after-itchy")
        p.add_argument("--backbone", default="lingbot", choices=["lingbot", "dinov2"],
                       help="frozen feature source: lingbot (patch-16) or dinov2 (patch-14)")
        p.add_argument("--variant", default="small", help="small/base/large/giant")
        p.add_argument("--size", type=int, default=512)
        p.add_argument("--mode", default="square", choices=["square", "shortest"])
        p.add_argument("--out", default=None)
        p.add_argument("--upsample", default="none", choices=["none", "anyup"],
                       help="feature upsampler applied to frozen tokens before probing")
        p.add_argument("--up-size", type=int, default=128,
                       help="AnyUp target grid (square); larger = crisper but more VRAM/RAM")
        p.add_argument("--anyup-ckpt", default="multi", choices=["multi", "paper"],
                       help="multi = anyup_multi_backbone (best for unseen backbones)")
        p.add_argument("--anyup-src", default="/home/play/Code/anyup",
                       help="path to cloned wimmerth/anyup repo (added to sys.path)")

    p_pca = sub.add_parser("pca"); common(p_pca)
    p_pca.add_argument("--num", type=int, default=6)

    p_lin = sub.add_parser("linprobe"); common(p_lin)
    p_lin.add_argument("--num", type=int, default=24)
    p_lin.add_argument("--val-frac", type=float, default=0.3)
    p_lin.add_argument("--steps", type=int, default=300)
    p_lin.add_argument("--save-head", default=None,
                       help="save trained head + config to this path for make_segmenter('lingbot')")

    p_seg = sub.add_parser("segment")
    p_seg.add_argument("--session", default="maguro-park-after-itchy")
    p_seg.add_argument("--head", required=True, help="checkpoint from linprobe --save-head")
    p_seg.add_argument("--size", type=int, default=512)
    p_seg.add_argument("--num", type=int, default=6)
    p_seg.add_argument("--ref", action="store_true", help="also render a Mask2Former column")
    p_seg.add_argument("--out", default=None)

    args = ap.parse_args()
    {"pca": run_pca, "linprobe": run_linprobe, "segment": run_segment}[args.task](args)


if __name__ == "__main__":
    main()
