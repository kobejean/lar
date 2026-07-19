"""Evaluate the frozen LingBot-Vision backbone against ADE20K **ground truth**.

`probe.py --task linprobe` trains *and* scores its head on `Mask2Former` output, so its
headline 0.87 is **agreement with the teacher**, not accuracy. A head can only look good
there by copying Mask2Former's mistakes, and the number is silently capped by them. This
script swaps in real ADE20K labels so three things become separable:

  * **LingBot + linear head vs GT**  — what the frozen features actually support
  * **Mask2Former vs GT**            — the teacher's own error, i.e. the ceiling
                                       `probe.py` was unknowingly measuring against
  * the gap between them             — whether distilling the teacher is worth it, or
                                       whether a linear head on LingBot already beats it

ADE20K is the right corpus for this: its label set covers our taxonomy far better than
COCO (tree, plant, bush, bench, pole, signboard, rock are all present, outdoors), and the
Mask2Former checkpoint already cached here is ADE-trained — so teacher and GT share a
label space and the comparison is apples-to-apples.

Two tasks:

  semantic — patch-level linear probe in our `Klass` taxonomy. Reports per-class IoU,
             mIoU and pixel accuracy for LingBot and for Mask2Former on the same images.

  instance — the question the BEV pipeline actually chokes on. `base_points.py`'s README
             notes "the same tree, projected from 970 frames, becomes one un-separable
             BEV blob"; instance separation is what fixes it. This measures whether the
             frozen features carry instance identity *at all*: sample patch pairs drawn
             from the same class, and score how well cosine similarity ranks
             same-instance pairs above different-instance ones (ROC AUC). 0.5 = features
             are purely semantic and no head can split touching instances from them; high
             = the information is present and a head can recover it.

Data (no registration needed, ~1 GB + 90 MB):
  http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
  http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar

Run (from repo root):
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/ade20k_eval.py semantic --root /home/play/Code/datasets/ade20k --num 200
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/ade20k_eval.py instance --root /home/play/Code/datasets/ade20k --num 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SCRIPT_ROOT = _HERE.parent
sys.path.insert(0, str(_SCRIPT_ROOT))
sys.path.insert(0, str(_SCRIPT_ROOT / "semantic_bev"))

from lingbot_vision import extract_patch_tokens, load_image, load_pretrained_backbone  # noqa: E402
from taxonomy import Klass, keyword_klass  # noqa: E402


# ---------------------------------------------------------------------------
# ADE20K -> our taxonomy
# ---------------------------------------------------------------------------
def ade_root(root: Path) -> Path:
    """Accept either the extract dir or its ADEChallengeData2016 parent."""
    return root if (root / "images").is_dir() else root / "ADEChallengeData2016"


def load_class_names(base: Path) -> list[str]:
    """objectInfo150 -> index-aligned names; index 0 is ADE's 'ignore'."""
    info = next((p for p in (base / "objectInfo150.txt", base / "objectInfo150.csv")
                 if p.exists()), None)
    if info is None:
        raise SystemExit(f"objectInfo150 not found under {base}")
    names = ["ignore"] * 151
    for line in info.read_text().splitlines()[1:]:
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if 1 <= idx <= 150:
            names[idx] = parts[-1].strip()
    return names


# Explicit ADE-150 -> Klass. NOT derived from `taxonomy.keyword_klass`: that mapper does
# unanchored substring matching and mislabels real classes --
#   "seat" -> WATER ("sea")            "carpet"      -> VEHICLE ("car")
#   "streetlight" -> TREE ("s-TREE-t") "kitchen island" -> TERRAIN ("land")
#   "skyscraper" -> SKY                "pool table"  -> WATER
# Those errors are live in the park pipeline (footprint2d's SegKlass builds its LUT the
# same way): maguro-park's structure raster carries 9 WATER cells in a park with no water.
# Ground truth has to be exact, so this table is hand-written. Classes absent here stay
# UNKNOWN and are excluded from the metric rather than guessed at.
_ADE_KLASS: dict[int, Klass] = {
    1: Klass.WALL, 2: Klass.BUILDING, 3: Klass.SKY, 5: Klass.TREE, 7: Klass.PATH,
    10: Klass.GRASS, 12: Klass.PATH, 13: Klass.PERSON, 14: Klass.TERRAIN,
    17: Klass.TERRAIN, 18: Klass.TREE, 21: Klass.VEHICLE, 22: Klass.WATER,
    26: Klass.BUILDING, 27: Klass.WATER, 30: Klass.GRASS, 32: Klass.FURNITURE,
    33: Klass.WALL, 35: Klass.TERRAIN, 39: Klass.WALL, 41: Klass.FURNITURE,
    43: Klass.FURNITURE, 44: Klass.FURNITURE, 47: Klass.TERRAIN, 49: Klass.BUILDING,
    52: Klass.BUILDING, 53: Klass.PATH, 54: Klass.STAIRS, 55: Klass.PATH,
    60: Klass.STAIRS, 61: Klass.WATER, 67: Klass.GRASS, 69: Klass.TERRAIN,
    70: Klass.FURNITURE, 73: Klass.TREE, 77: Klass.VEHICLE, 80: Klass.BUILDING,
    81: Klass.VEHICLE, 83: Klass.FURNITURE, 84: Klass.VEHICLE, 85: Klass.BUILDING,
    88: Klass.FURNITURE, 89: Klass.BUILDING, 91: Klass.VEHICLE, 92: Klass.TERRAIN,
    94: Klass.FURNITURE, 95: Klass.TERRAIN, 96: Klass.WALL, 97: Klass.STAIRS,
    103: Klass.VEHICLE, 104: Klass.VEHICLE, 105: Klass.WATER, 110: Klass.WATER,
    114: Klass.WATER, 117: Klass.VEHICLE, 122: Klass.STAIRS, 126: Klass.FURNITURE,
    128: Klass.VEHICLE, 129: Klass.WATER, 133: Klass.FURNITURE, 136: Klass.FURNITURE,
    137: Klass.FURNITURE, 139: Klass.FURNITURE, 145: Klass.FURNITURE,
}


def ade_to_klass(names: list[str], mapping: str = "explicit") -> np.ndarray:
    """LUT: ADE id (0..150) -> our Klass.

    mapping='keyword' reproduces the buggy substring mapper on purpose, so the damage it
    does can be measured against the same GT rather than argued about.
    """
    lut = np.zeros(151, dtype=np.uint8)
    if mapping == "keyword":
        for i, n in enumerate(names):
            # keyword_klass matches on word boundaries across the whole synonym list now, so
            # the old per-synonym max() tie-break (which just picked the highest Klass id) is
            # gone -- it had no principled basis.
            lut[i] = int(Klass.UNKNOWN) if i == 0 else int(keyword_klass(n))
        return lut
    for i, k in _ADE_KLASS.items():
        lut[i] = int(k)
    return lut


def list_split(base: Path, split: str, num: int, seed: int = 0):
    img_dir = base / "images" / split
    ann_dir = base / "annotations" / split
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {img_dir}")
    rng = np.random.default_rng(seed)
    if num and num < len(imgs):
        imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), num, replace=False))]
    return [(p, ann_dir / f"{p.stem}.png") for p in imgs if (ann_dir / f"{p.stem}.png").exists()]


# ---------------------------------------------------------------------------
# frozen features — same recipe as probe.py so numbers stay comparable
# ---------------------------------------------------------------------------
def load_backbone(variant: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    backbone, embed_dim = load_pretrained_backbone(variant=variant, device=device, dtype=dtype)
    return backbone, embed_dim, device, dtype


@torch.no_grad()
def feature_grid(backbone, path: Path, size: int, mode: str, device, dtype):
    img_norm, rgb, _ = load_image(str(path), size=size, patch_size=backbone.patch_size, mode=mode)
    tokens, (h, w) = extract_patch_tokens(backbone, img_norm, device, dtype)
    return tokens[0].float().cpu().numpy().reshape(h, w, -1), rgb, (h, w)


def patch_labels(mask_hw: np.ndarray, h: int, w: int, nvals: int) -> np.ndarray:
    """Majority label per patch block (probe.py's rule)."""
    H, W = mask_hw.shape
    ph, pw = H // h, W // w
    out = np.empty(h * w, dtype=np.int64)
    k = 0
    for i in range(h):
        for j in range(w):
            block = mask_hw[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw].ravel()
            out[k] = np.bincount(block, minlength=nvals).argmax()
            k += 1
    return out


def resize_label(mask: np.ndarray, rgb_shape) -> np.ndarray:
    """GT is at native resolution; the backbone saw a resized/snapped crop. Nearest-resize
    the label to exactly what the backbone consumed so patches and labels align."""
    H, W = rgb_shape[:2]
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def iou_report(inter, union, correct, total, names, tag):
    valid = union > 0
    ious = np.divide(inter, np.maximum(union, 1e-9))
    miou = float(ious[valid].mean()) if valid.any() else 0.0
    acc = float(correct / max(total, 1))
    print(f"\n[{tag}]  mIoU {miou*100:.1f}   pixel-acc {acc*100:.1f}   "
          f"({int(valid.sum())} classes present)")
    for i in np.argsort(-union):
        if union[i] <= 0:
            continue
        print(f"    {names[i]:<10s} IoU {ious[i]*100:5.1f}   support {int(union[i]):>9d}")
    return {"miou": miou, "acc": acc,
            "per_class": {names[i]: float(ious[i]) for i in range(len(names)) if union[i] > 0}}


def accumulate(pred, gt, n, inter, union, ignore):
    m = gt != ignore
    pred, gt = pred[m], gt[m]
    for k in range(n):
        p, g = pred == k, gt == k
        inter[k] += np.logical_and(p, g).sum()
        union[k] += np.logical_or(p, g).sum()
    return int((pred == gt).sum()), int(m.sum())


# ---------------------------------------------------------------------------
# task: semantic
# ---------------------------------------------------------------------------
def run_semantic(args):
    base = ade_root(Path(args.root).expanduser())
    names_ade = load_class_names(base)
    lut = ade_to_klass(names_ade, args.mapping)
    n_klass = int(max(Klass)) + 1
    klass_names = [k.name for k in sorted(Klass, key=int)]
    ignore = int(Klass.UNKNOWN)

    backbone, embed_dim, device, dtype = load_backbone(args.variant)
    train = list_split(base, "training", args.num, seed=0)
    val = list_split(base, "validation", args.val_num, seed=1)
    print(f"[semantic] {len(train)} train / {len(val)} val images  size={args.size} "
          f"embed_dim={embed_dim} device={device}")

    Xtr, ytr = [], []
    for i, (ip, ap) in enumerate(train):
        f, rgb, (h, w) = feature_grid(backbone, ip, args.size, args.mode, device, dtype)
        gt = lut[resize_label(cv2.imread(str(ap), cv2.IMREAD_GRAYSCALE), rgb.shape)]
        y = patch_labels(gt, h, w, n_klass)
        Xtr.append(f.reshape(-1, embed_dim)); ytr.append(y)
        if (i + 1) % 25 == 0:
            print(f"  train feat {i+1}/{len(train)}")
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    keep = ytr != ignore
    Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"  {len(Xtr)} labelled patches "
          f"({', '.join(f'{klass_names[c]}:{n}' for c, n in zip(*np.unique(ytr, return_counts=True)))})")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    yn = torch.tensor(ytr, device=device)
    head = torch.nn.Linear(embed_dim, n_klass).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    for step in range(args.steps):
        opt.zero_grad()
        loss = lossf(head(Xn), yn)
        loss.backward(); opt.step()
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(f"  step {step+1}/{args.steps} loss={loss.item():.3f}")
    head.eval()

    seg = None
    if not args.no_teacher:
        from segmentation import make_segmenter
        seg = make_segmenter(args.teacher)
        print(f"  teacher: {args.teacher}")

    li, lu = np.zeros(n_klass), np.zeros(n_klass)
    ti, tu = np.zeros(n_klass), np.zeros(n_klass)
    lc = lt = tc = tt = 0
    for i, (ip, ap) in enumerate(val):
        f, rgb, (h, w) = feature_grid(backbone, ip, args.size, args.mode, device, dtype)
        gt_full = lut[resize_label(cv2.imread(str(ap), cv2.IMREAD_GRAYSCALE), rgb.shape)]
        with torch.no_grad():
            Xv = torch.tensor((f.reshape(-1, embed_dim) - mu) / sd, dtype=torch.float32, device=device)
            pred_patch = head(Xv).argmax(1).cpu().numpy().reshape(h, w)
        # score at full image resolution: upsample patch predictions, as any real
        # consumer would. Patch-grid scoring flatters the head by hiding boundary error.
        pred = cv2.resize(pred_patch.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        c, t = accumulate(pred.ravel(), gt_full.ravel(), n_klass, li, lu, ignore)
        lc += c; lt += t
        if seg is not None:
            tpred = seg.segment(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            c, t = accumulate(tpred.ravel(), gt_full.ravel(), n_klass, ti, tu, ignore)
            tc += c; tt += t
        if (i + 1) % 25 == 0:
            print(f"  val {i+1}/{len(val)}")

    res = {"lingbot": iou_report(li, lu, lc, lt, klass_names, f"LingBot-{args.variant} + linear  vs GT")}
    if seg is not None:
        res["teacher"] = iou_report(ti, tu, tc, tt, klass_names, f"{args.teacher}  vs GT")
        d = (res["lingbot"]["miou"] - res["teacher"]["miou"]) * 100
        print(f"\n  => LingBot linear {'beats' if d > 0 else 'trails'} the teacher by "
              f"{abs(d):.1f} mIoU. probe.py's 0.87 was agreement with that teacher, "
              f"whose own mIoU vs GT is {res['teacher']['miou']*100:.1f}.")

    out = Path(args.out or (_SCRIPT_ROOT.parent / "output" / "ade20k-eval"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"semantic_{args.variant}.json").write_text(json.dumps(
        {"config": vars(args), "results": res}, indent=2, default=str))
    print(f"\n  wrote {out}/semantic_{args.variant}.json")


# ---------------------------------------------------------------------------
# task: instance
# ---------------------------------------------------------------------------
def decode_instance(png: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ADE20K instance PNG -> (class, instance-id).

    Verified against the released val set: R = class id (index into the 100 *thing*
    classes, NOT the 150 semantic ids), G = per-image instance index, B unused (always 0).
    cv2 hands back BGR, so R is channel 2 and G is channel 1. Images whose channels are
    all zero are simply images with no countable things -- skipped by the caller.
    """
    if png.ndim == 2:
        return png.astype(np.int32), np.zeros_like(png, np.int32)
    cls = png[:, :, 2].astype(np.int32)
    inst = png[:, :, 1].astype(np.int32)
    return cls, inst


def run_instance(args):
    base = ade_root(Path(args.root).expanduser())
    inst_dir = Path(args.root).expanduser() / "annotations_instance" / "validation"
    if not inst_dir.is_dir():
        raise SystemExit(f"instance annotations not found: {inst_dir}\n"
                         "  extract annotations_instance.tar next to ADEChallengeData2016")
    backbone, embed_dim, device, dtype = load_backbone(args.variant)

    imgs = sorted((base / "images" / "validation").glob("*.jpg"))
    rng = np.random.default_rng(0)
    if args.num and args.num < len(imgs):
        imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), args.num, replace=False))]

    same, diff = [], []
    used = 0
    for ip in imgs:
        ap = inst_dir / f"{ip.stem}.png"
        if not ap.exists():
            continue
        f, rgb, (h, w) = feature_grid(backbone, ip, args.size, args.mode, device, dtype)
        png = cv2.imread(str(ap), cv2.IMREAD_COLOR)
        if png is None:
            continue
        cls, inst = decode_instance(png)
        cls = resize_label(cls.astype(np.int32), rgb.shape)
        inst = resize_label(inst.astype(np.int32), rgb.shape)
        # patch-level majority for both channels; a patch is usable only if it sits
        # cleanly inside one instance (mixed patches would blur the comparison)
        cl = patch_labels(cls, h, w, int(cls.max()) + 1).reshape(h, w)
        il = patch_labels(inst, h, w, int(inst.max()) + 1).reshape(h, w)
        feats = f.reshape(-1, embed_dim)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        cl, il = cl.ravel(), il.ravel()

        for c in np.unique(cl):
            if c == 0:
                continue
            m = (cl == c) & (il > 0)
            ids = np.unique(il[m])
            if len(ids) < 2:                     # need >= 2 instances of a class to compare
                continue
            idx = np.where(m)[0]
            if len(idx) > args.max_patches:
                idx = rng.choice(idx, args.max_patches, replace=False)
            sub, sid = feats[idx], il[idx]
            sim = sub @ sub.T
            eq = sid[:, None] == sid[None, :]
            triu = np.triu(np.ones_like(sim, bool), 1)
            same.append(sim[triu & eq]); diff.append(sim[triu & ~eq])
        used += 1
        if used % 20 == 0:
            print(f"  {used} images, {sum(len(s) for s in same)} same / "
                  f"{sum(len(d) for d in diff)} diff pairs")

    same = np.concatenate([s for s in same if len(s)]) if same else np.array([])
    diff = np.concatenate([d for d in diff if len(d)]) if diff else np.array([])
    if not len(same) or not len(diff):
        raise SystemExit("no usable instance pairs — try a larger --num")

    # rank-based AUC: P(sim(same-instance) > sim(different-instance-same-class))
    allv = np.concatenate([same, diff])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    n1, n0 = len(same), len(diff)
    auc = float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    print(f"\n[instance separability — LingBot-{args.variant}, {used} images]")
    print(f"  same-instance pairs      {n1:>9d}   mean cos {same.mean():.4f}")
    print(f"  diff-instance same-class {n0:>9d}   mean cos {diff.mean():.4f}")
    print(f"  ROC AUC                  {auc:.4f}")
    verdict = ("features are essentially semantic-only — a head cannot split touching "
               "instances from them; you need an instance-aware signal (detector/VLM points)"
               if auc < 0.60 else
               "instance identity is weakly present — a trained head may separate instances"
               if auc < 0.75 else
               "instance identity is strongly present — worth training an instance head on "
               "these frozen features")
    print(f"  => {verdict}")

    out = Path(args.out or (_SCRIPT_ROOT.parent / "output" / "ade20k-eval"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"instance_{args.variant}.json").write_text(json.dumps({
        "config": vars(args), "images": used, "auc": auc,
        "same_mean": float(same.mean()), "diff_mean": float(diff.mean()),
        "n_same": n1, "n_diff": n0, "verdict": verdict}, indent=2, default=str))
    print(f"  wrote {out}/instance_{args.variant}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="task", required=True)
    for name in ("semantic", "instance"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="/home/play/Code/datasets/ade20k")
        p.add_argument("--variant", default="small", help="small/base/large/giant")
        p.add_argument("--size", type=int, default=512)
        p.add_argument("--mode", default="square", choices=["square", "shortest"])
        p.add_argument("--num", type=int, default=200)
        p.add_argument("--out", default=None)
    s = sub.choices["semantic"]
    s.add_argument("--val-num", type=int, default=100)
    s.add_argument("--steps", type=int, default=600)
    s.add_argument("--teacher", default="mask2former-large",
                   help="teacher to score against the same GT (the probe.py ceiling)")
    s.add_argument("--no-teacher", action="store_true")
    s.add_argument("--mapping", default="explicit", choices=["explicit", "keyword"],
                   help="ADE->Klass LUT. 'keyword' reproduces taxonomy.keyword_klass's "
                        "buggy substring matching, to measure what it costs.")
    i = sub.choices["instance"]
    i.add_argument("--max-patches", type=int, default=400,
                   help="patch cap per class per image (pair count grows quadratically)")
    args = ap.parse_args()
    (run_semantic if args.task == "semantic" else run_instance)(args)


if __name__ == "__main__":
    main()
