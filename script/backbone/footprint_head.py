"""Train a footprint / free-space / ground-contact head on the frozen LingBot backbone.

This is the *distillation* step: the multi-frame geometric pipeline
(``footprint_labels.py``, DEM-render supervision) teaches a **single-image** head to
predict, from one RGB frame, the ground state a BEV projector needs — where ground is
walkable, where it is occupied by an object's footprint, and where it continues *behind*
an occluder (Niantic "Footprints and Free Space", CVPR2020). The backbone stays frozen;
only a small MLP on its patch tokens is trained, so this needs no manual labels and very
little data.

Two heads share one trunk on the frozen (optionally AnyUp-upsampled) patch grid:

  class   -- 5-way ground state, one of the ``footprint_labels`` classes:
             0 NON_GROUND  1 VISIBLE  2 HIDDEN  3 FOOTPRINT  4 OBJECT
  contact -- binary "ground-contact contour": a FOOTPRINT patch touching walkable ground
             (VISIBLE|HIDDEN). This is the class-agnostic precursor to per-object contact
             *keypoints* — once SAM2 gives instances, each object's contact arc collapses
             to one keypoint. Buildable today with zero extra deps.

Supervision comes from ``footprint_labels.py`` outputs (run that first):
  <stem>.png        uint8 5-class label at camera aspect
  <stem>_cover.npy  float32 {0,1} DEM-observed vs extrapolated -> per-pixel loss weight

Alignment: LingBot ``load_image(mode="square")`` is a plain anamorphic resize of the full
frame to SxS, and the DEM labels are rendered full-frame at camera aspect, so a label maps
to the patch grid by an anamorphic resize to SxS + block-majority-pool to GxG (cover is
mean-pooled). No cropping, so the two stay pixel-aligned.

Run (from repo root, same env as probe.py):
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/footprint_head.py --session maguro-park-after-itchy \
      --upsample anyup --up-size 128 --save-head \
      output/maguro-park-after-itchy-backbone/footprint_head.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from probe import feature_grid, load_anyup, load_backbone, report_vram  # noqa: E402
from lar_session import Session  # noqa: E402

# label ids -- MUST match footprint_labels.py
NON_GROUND, VISIBLE, HIDDEN, FOOTPRINT, OBJECT = 0, 1, 2, 3, 4
CLASS_NAMES = ["non_ground", "visible", "hidden", "footprint", "object"]
N_CLASS = 5
GROUND = (VISIBLE, HIDDEN)  # walkable ground states (contact touches these)

# RGB palette for renders (footprint_labels uses the BGR mirror of this)
PALETTE = np.array([
    [0, 0, 0],        # non_ground  black
    [80, 200, 80],    # visible     green
    [220, 120, 40],   # hidden      orange
    [220, 40, 40],    # footprint   red
    [60, 200, 200],   # object      cyan
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# label <-> patch-grid alignment
# ---------------------------------------------------------------------------
def pool_labels(label_sq: np.ndarray, cover_sq: np.ndarray, g: int):
    """(S,S) label + cover -> (g,g) majority class, (g,g) mean cover.

    S must be a multiple of g (512 & patch-16 -> 32; AnyUp 128 -> 4). Falls back to a
    nearest/area resize if not divisible.
    """
    s = label_sq.shape[0]
    if s % g != 0:
        cls = cv2.resize(label_sq, (g, g), interpolation=cv2.INTER_NEAREST)
        cov = cv2.resize(cover_sq, (g, g), interpolation=cv2.INTER_AREA)
        return cls.astype(np.int64), cov.astype(np.float32)
    b = s // g
    blocks = label_sq.reshape(g, b, g, b).transpose(0, 2, 1, 3).reshape(g * g, b * b)
    # majority class per block
    onehot = (blocks[:, None, :] == np.arange(N_CLASS)[None, :, None]).sum(2)  # (g*g, C)
    cls = onehot.argmax(1).reshape(g, g).astype(np.int64)
    cov = cover_sq.reshape(g, b, g, b).mean((1, 3)).astype(np.float32)
    return cls, cov


def contact_target(cls: np.ndarray) -> np.ndarray:
    """Binary ground-contact contour on the (g,g) class grid.

    A FOOTPRINT cell 4-adjacent to walkable ground (VISIBLE|HIDDEN) — i.e. the object's
    ground-contact edge, not its interior. Class-agnostic; the per-instance keypoint is a
    later SAM2 step.
    """
    foot = cls == FOOTPRINT
    ground = np.isin(cls, GROUND)
    gd = cv2.dilate(ground.astype(np.uint8), np.ones((3, 3), np.uint8))
    return (foot & (gd > 0)).astype(np.float32)


def load_frame_target(label_dir: Path, stem: str, g: int, s: int):
    """(cls (g,g), cover (g,g), contact (g,g)) for a frame, aligned to the patch grid."""
    lab = cv2.imread(str(label_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if lab is None:
        return None
    cov_p = label_dir / f"{stem}_cover.npy"
    cover = np.load(cov_p).astype(np.float32) if cov_p.exists() else np.ones_like(lab, np.float32)
    lab_sq = cv2.resize(lab, (s, s), interpolation=cv2.INTER_NEAREST)
    cov_sq = cv2.resize(cover, (s, s), interpolation=cv2.INTER_LINEAR)
    cls, cov = pool_labels(lab_sq, cov_sq, g)
    return cls, cov, contact_target(cls)


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------
class FootprintHead(torch.nn.Module):
    """Shared MLP trunk on a patch token -> (5-way class logits, contact logit)."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        )
        self.cls = torch.nn.Linear(hidden, N_CLASS)
        self.contact = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.cls(h), self.contact(h).squeeze(-1)


# ---------------------------------------------------------------------------
def frame_stems(label_dir: Path) -> list[str]:
    """Stems that have a label PNG (exclude preview/colourised aux images)."""
    out = []
    for p in sorted(label_dir.glob("*.png")):
        if p.stem.endswith("_preview"):
            continue
        out.append(p.stem)
    return out


def run(args):
    s = Session(args.session)
    label_dir = Path(args.label_dir or (s.root / "output" / f"{args.session}-footprint"))
    if not label_dir.exists():
        raise SystemExit(f"no label dir {label_dir} -- run footprint_labels.py first")
    out = Path(args.out or (s.root / "output" / f"{args.session}-backbone"))
    out.mkdir(parents=True, exist_ok=True)

    backbone, embed_dim, device, dtype = load_backbone(args.variant, args.backbone)
    up = load_anyup(args.anyup_ckpt, args.anyup_src, device) if args.upsample == "anyup" else None
    up_size = args.up_size if up is not None else 0

    stems = frame_stems(label_dir)
    if args.num and args.num < len(stems):
        idx = np.linspace(0, len(stems) - 1, args.num).round().astype(int)
        stems = [stems[i] for i in sorted(set(idx))]
    n_val = max(1, int(round(len(stems) * args.val_frac)))
    val_stems = set(stems[-n_val:])
    print(f"[footprint_head] backbone={args.backbone}/{args.variant} embed_dim={embed_dim} "
          f"frames={len(stems)} (val={n_val}) size={args.size} upsample={args.upsample}"
          + (f" up_size={up_size}" if up else ""))
    print(f"  labels <- {label_dir}")

    Xtr, ytr, wtr, ctr = [], [], [], []
    val = []  # (feat[G,G,C], cls(g,g), cover(g,g), contact(g,g), rgb, g)
    n_ok = 0
    for stem in stems:
        img_path = s.images / f"{stem}.jpeg"
        if not img_path.exists():
            continue
        feat, rgb, (g, _) = feature_grid(backbone, img_path, args.size, args.mode,
                                         device, dtype, up, up_size, args.backbone)
        tgt = load_frame_target(label_dir, stem, g, rgb.shape[0])
        if tgt is None:
            continue
        cls, cov, ctc = tgt
        n_ok += 1
        if stem in val_stems:
            val.append((feat, cls, cov, ctc, rgb, g))
        else:
            Xtr.append(feat.reshape(-1, embed_dim))
            ytr.append(cls.reshape(-1))
            wtr.append(np.clip(cov.reshape(-1), args.cover_floor, 1.0))
            ctr.append(ctc.reshape(-1))
    if not Xtr:
        raise SystemExit("no training frames with both features and labels")
    print(f"  matched {n_ok} frames to labels")

    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    wtr = np.concatenate(wtr); ctr = np.concatenate(ctr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    # keep the (potentially millions of) tokens on CPU; minibatch to GPU per step so the
    # AnyUp-upsampled grid (128^2 tokens/frame) doesn't blow the card's VRAM.
    Xn = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yn = torch.tensor(ytr)
    wn = torch.tensor(wtr, dtype=torch.float32)
    cn = torch.tensor(ctr, dtype=torch.float32)

    # class-balanced CE weights (footprint/hidden are rare) x per-pixel cover weight
    freq = np.bincount(ytr, minlength=N_CLASS).astype(np.float64)
    cls_w = torch.tensor((freq.sum() / np.maximum(freq, 1)) ** 0.5, dtype=torch.float32, device=device)
    cls_w /= cls_w.mean()
    # contact positives are sparse -> pos_weight balances BCE
    pos = float(ctr.sum()); neg = float(len(ctr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    print(f"  class freq {dict(zip(CLASS_NAMES, freq.astype(int)))}  contact pos={int(pos)}")

    head = FootprintHead(embed_dim, args.hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss(weight=cls_w, reduction="none")
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    head.train()
    n = Xn.shape[0]
    bs = min(args.batch, n)
    gen = torch.Generator().manual_seed(0)
    for step in range(args.steps):
        sel = torch.randint(0, n, (bs,), generator=gen)
        xb = Xn[sel].to(device); yb = yn[sel].to(device)
        wb = wn[sel].to(device); cb = cn[sel].to(device)
        opt.zero_grad()
        logit_c, logit_ct = head(xb)
        loss_c = (ce(logit_c, yb) * wb).sum() / wb.sum().clamp_min(1)
        loss_ct = bce(logit_ct, cb)
        loss = loss_c + args.contact_w * loss_ct
        loss.backward(); opt.step()
        if (step + 1) % max(1, args.steps // 6) == 0:
            print(f"  step {step+1}/{args.steps}  cls={loss_c.item():.3f}  contact={loss_ct.item():.3f}")
    head.eval()

    _evaluate(head, val, embed_dim, mu, sd, device, out, args)
    report_vram("footprint_head", device)

    if args.save_head:
        sp = Path(args.save_head)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "head": {k: v.cpu() for k, v in head.state_dict().items()},
            "hidden": args.hidden, "mu": mu, "sd": sd,
            "backbone": args.backbone, "variant": args.variant, "embed_dim": embed_dim,
            "size": args.size, "mode": args.mode,
            "upsample": args.upsample, "up_size": up_size,
            "anyup_ckpt": args.anyup_ckpt, "anyup_src": args.anyup_src,
            "classes": CLASS_NAMES,
        }, sp)
        print(f"[footprint_head] saved head -> {sp}")


def _evaluate(head, val, embed_dim, mu, sd, device, out, args):
    inter = np.zeros(N_CLASS); union = np.zeros(N_CLASS)
    correct = total = 0.0
    ct_tp = ct_fp = ct_fn = 0
    tiles = []
    for feat, cls, cov, ctc, rgb, g in val:
        Xv = torch.tensor((feat.reshape(-1, embed_dim) - mu) / sd, dtype=torch.float32, device=device)
        with torch.no_grad():
            lc, lct = head(Xv)
            pred = lc.argmax(1).cpu().numpy()
            pct = (torch.sigmoid(lct) > 0.5).cpu().numpy()
        y = cls.reshape(-1); w = cov.reshape(-1) > 0.0
        correct += float((pred[w] == y[w]).sum()); total += float(w.sum())
        for c in range(N_CLASS):
            pc, yc = (pred == c) & w, (y == c) & w
            inter[c] += float((pc & yc).sum()); union[c] += float((pc | yc).sum())
        gt = ctc.reshape(-1) > 0.5
        ct_tp += int((pct & gt).sum()); ct_fp += int((pct & ~gt).sum()); ct_fn += int((~pct & gt).sum())
        tiles.append(_tile(rgb, cls, pred.reshape(g, g), ctc, pct.reshape(g, g)))

    iou = inter / np.maximum(union, 1)
    present = union > 0
    prec = ct_tp / max(ct_tp + ct_fp, 1); rec = ct_tp / max(ct_tp + ct_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    print(f"\n[footprint_head] val acc={correct/max(total,1):.3f}  mIoU(present)={iou[present].mean():.3f}")
    for c in np.where(present)[0]:
        print(f"    {CLASS_NAMES[c]:11s} IoU {iou[c]:.3f}")
    print(f"  contact  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    tag = "" if args.upsample == "none" else f"_{args.upsample}{args.up_size}"
    dst = out / f"footprint_head_{args.variant}{tag}.png"
    cv2.imwrite(str(dst), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
    print(f"[footprint_head] wrote {dst}  (RGB | target | pred | contact)")


def _tile(rgb, cls_g, pred_g, ctc_g, pct_g):
    """RGB | target-class | pred-class | pred-contact-over-RGB, all upsampled to rgb size."""
    S = rgb.shape[0]
    up = lambda a: cv2.resize(a, (S, S), interpolation=cv2.INTER_NEAREST)
    tgt = up(PALETTE[cls_g])
    prd = up(PALETTE[pred_g])
    ov = rgb.copy()
    ct = up((pct_g * 255).astype(np.uint8))
    ov[ct > 0] = (0.35 * ov[ct > 0] + np.array([255, 255, 0]) * 0.65).astype(np.uint8)
    # ground-truth contact as a thin outline for reference (magenta)
    gt = up((ctc_g * 255).astype(np.uint8))
    ov[(gt > 0) & (ct == 0)] = (0.5 * ov[(gt > 0) & (ct == 0)] + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
    return np.concatenate([rgb, tgt, prd, ov], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="maguro-park-after-itchy")
    ap.add_argument("--label-dir", default=None, help="footprint_labels.py output (default output/<session>-footprint)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--backbone", default="lingbot", choices=["lingbot", "dinov2"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--mode", default="square", choices=["square", "shortest"])
    ap.add_argument("--num", type=int, default=0, help="subsample labelled frames (0=all)")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=262144, help="tokens/step (minibatch to GPU)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--contact-w", type=float, default=0.5, help="contact-head loss weight")
    ap.add_argument("--cover-floor", type=float, default=0.2, help="min per-patch loss weight")
    ap.add_argument("--upsample", default="none", choices=["none", "anyup"])
    ap.add_argument("--up-size", type=int, default=128)
    ap.add_argument("--anyup-ckpt", default="multi", choices=["multi", "paper"])
    ap.add_argument("--anyup-src", default="/home/play/Code/anyup")
    ap.add_argument("--save-head", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
