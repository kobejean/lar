"""Train a walkable-SURFACE head on the frozen LingBot backbone.

Complements `footprint_head.py`, which learns ground *geometry* (free / hidden / footprint)
from DEM renders. This learns what the ground *is* — path, grass, dirt, stairs, water — the
distinction the BEV surface map needs and the one the current stack is measurably worst at.

**Why it exists.** Mask2Former labels this park's lawn `earth`, not `grass` (26.5% vs 11.5% of
pixels), and its GRASS IoU against ADE ground truth is **9.2** — near useless. A frozen-LingBot
linear probe scored **7.8** on the same class, so neither model is usable for the ground split
today. Two independent reasons to own this head rather than distil the teacher:

  * measured: LingBot + a linear head already *beats* Mask2Former on FURNITURE (24.6 vs 9.0),
    so distillation would import the teacher's blind spots (see `ade20k_eval.py`);
  * licensing: Mask2Former's **weights are CC-BY-NC** (its repo LICENSE is MIT, which covers
    code only — MODEL_ZOO.md states the NC term for every checkpoint). It cannot ship.

The backbone stays frozen and Apache-licensed; only this small MLP trains.

**Why the earlier probe scored 7.8, and what is different here.** That was a *linear* head over
all 14 classes with 120 training images and no class balancing — and GRASS is rare, so a
frequency-weighted loss simply ignores it. This narrows the problem to the ground split, uses
an MLP, trains on far more data, and weights classes by inverse-sqrt frequency so the rare
surface classes survive the loss.

**Licensing of the supervision matters more than usual here.** ADE20K *images* are not clear
for training a shipped model — they are fine to evaluate against. So `--data ade20k` exists to
validate the approach and to produce comparable numbers, and any checkpoint it yields is
research-only. A shippable head wants GOOSE (CC-BY-SA, and its `low_grass`/`high_grass`/
`gravel`/`soil`/`asphalt` classes are a far better fit for a park than ADE's coarse
`earth`/`grass` split) plus hand labels from `relabel.py` on our own frames. The loaders are
kept behind `--data` so swapping the corpus does not touch the model code.

Run (from repo root):
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/ground_head.py --data ade20k --train-num 800 --val-num 200
  # then look at what it does on real park frames, which is the actual target domain:
  uv run --extra segmentation --with omegaconf --with /home/play/Code/lingbot-vision \
    python script/backbone/ground_head.py --predict-session maguro-park-after-itchy \
      --head output/ground-head/ground_head.pt --predict-num 8
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
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SCRIPT_ROOT))
sys.path.insert(0, str(_SCRIPT_ROOT / "semantic_bev"))

from taxonomy import Klass  # noqa: E402
from ade20k_eval import (  # noqa: E402
    _ADE_KLASS, ade_root, feature_grid, list_split, load_backbone, load_class_names,
    patch_labels, resize_label,
)

# Ground split. Deliberately NOT the full 14-class taxonomy: the question is what a walkable
# cell is made of, and collapsing everything else into one NOT_GROUND class stops the head
# spending capacity on distinctions the BEV surface map never reads.
NOT_GROUND, G_PATH, G_GRASS, G_TERRAIN, G_STAIRS, G_WATER = range(6)
GNAMES = ["not-ground", "path/paved", "grass", "terrain/dirt", "stairs", "water"]
GCOLORS = np.array([(40, 40, 44), (238, 220, 170), (104, 176, 92),
                    (168, 142, 108), (206, 150, 210), (86, 140, 200)], np.uint8)

# our Klass -> ground class
KLASS_TO_GROUND = {
    int(Klass.PATH): G_PATH, int(Klass.PAVEMENT): G_PATH,
    int(Klass.GRASS): G_GRASS, int(Klass.TERRAIN): G_TERRAIN,
    int(Klass.STAIRS): G_STAIRS, int(Klass.WATER): G_WATER,
}


def ade_ground_lut(names: list[str]) -> np.ndarray:
    """ADE id -> ground class, via the hand-written ADE table (never keyword matching)."""
    lut = np.zeros(151, np.uint8)
    for i, k in _ADE_KLASS.items():
        lut[i] = KLASS_TO_GROUND.get(int(k), NOT_GROUND)
    return lut


class GroundHead(torch.nn.Module):
    """Small MLP on frozen patch tokens. Kept tiny on purpose — if this cannot separate grass
    from dirt, the fault is in the features or the labels, not in head capacity."""

    def __init__(self, dim: int, hidden: int, n_cls: int = 6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
            torch.nn.Linear(hidden // 2, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def iou_table(inter, union, tag):
    valid = union > 0
    ious = inter / np.maximum(union, 1e-9)
    miou = float(ious[valid].mean()) if valid.any() else 0.0
    print(f"\n[{tag}]  mIoU {miou * 100:.1f}")
    for i in np.argsort(-union):
        if union[i] <= 0:
            continue
        print(f"    {GNAMES[i]:<14s} IoU {ious[i] * 100:5.1f}   support {int(union[i]):>10d}")
    return miou, {GNAMES[i]: float(ious[i]) for i in range(len(GNAMES)) if union[i] > 0}


def accumulate(pred, gt, inter, union):
    for k in range(len(GNAMES)):
        p, g = pred == k, gt == k
        inter[k] += np.logical_and(p, g).sum()
        union[k] += np.logical_or(p, g).sum()


def train(args):
    base = ade_root(Path(args.root).expanduser())
    lut = ade_ground_lut(load_class_names(base))
    backbone, dim, device, dtype = load_backbone(args.variant)
    tr = list_split(base, "training", args.train_num, seed=0)
    va = list_split(base, "validation", args.val_num, seed=1)
    print(f"[ground_head] {len(tr)} train / {len(va)} val  dim={dim} device={device}")

    X, Y = [], []
    for i, (ip, ap) in enumerate(tr):
        f, rgb, (h, w) = feature_grid(backbone, ip, args.size, args.mode, device, dtype)
        gt = lut[resize_label(cv2.imread(str(ap), cv2.IMREAD_GRAYSCALE), rgb.shape)]
        X.append(f.reshape(-1, dim)); Y.append(patch_labels(gt, h, w, len(GNAMES)))
        if (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(tr)}")
    X = np.concatenate(X); Y = np.concatenate(Y)
    cnt = np.bincount(Y, minlength=len(GNAMES))
    print("  patches: " + ", ".join(f"{n}:{c}" for n, c in zip(GNAMES, cnt)))

    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(Y)

    # Inverse-sqrt frequency. Plain inverse frequency swings too hard on classes with a few
    # hundred patches and the head starts hallucinating them everywhere; unweighted, the rare
    # surface classes are simply ignored -- which is how the earlier probe scored 7.8 on grass.
    wt = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32)
    wt = (wt / wt.mean()).to(device)
    print("  class weights: " + ", ".join(f"{n}:{w:.2f}" for n, w in zip(GNAMES, wt.tolist())))

    head = GroundHead(dim, args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    lossf = torch.nn.CrossEntropyLoss(weight=wt)
    n = len(Xt)
    head.train()
    for step in range(args.steps):
        idx = torch.randint(0, n, (min(args.batch, n),))
        xb, yb = Xt[idx].to(device), Yt[idx].to(device)
        opt.zero_grad()
        loss = lossf(head(xb), yb)
        loss.backward(); opt.step(); sched.step()
        if (step + 1) % max(1, args.steps // 6) == 0:
            print(f"  step {step + 1}/{args.steps} loss={loss.item():.4f}")
    head.eval()

    hi, hu = np.zeros(len(GNAMES)), np.zeros(len(GNAMES))
    ti, tu = np.zeros(len(GNAMES)), np.zeros(len(GNAMES))
    seg = None
    if not args.no_teacher:
        from segmentation import make_segmenter
        seg = make_segmenter(args.teacher)
    for i, (ip, ap) in enumerate(va):
        f, rgb, (h, w) = feature_grid(backbone, ip, args.size, args.mode, device, dtype)
        gt = lut[resize_label(cv2.imread(str(ap), cv2.IMREAD_GRAYSCALE), rgb.shape)]
        with torch.no_grad():
            xv = torch.tensor((f.reshape(-1, dim) - mu) / sd, dtype=torch.float32, device=device)
            pp = head(xv).argmax(1).cpu().numpy().reshape(h, w).astype(np.uint8)
        pred = cv2.resize(pp, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        accumulate(pred.ravel(), gt.ravel(), hi, hu)
        if seg is not None:
            tk = seg.segment(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            tg = np.vectorize(lambda k: KLASS_TO_GROUND.get(int(k), NOT_GROUND))(tk).astype(np.uint8)
            accumulate(tg.ravel(), gt.ravel(), ti, tu)
        if (i + 1) % 50 == 0:
            print(f"  val {i + 1}/{len(va)}")

    hm, hres = iou_table(hi, hu, f"LingBot-{args.variant} + ground MLP  vs GT")
    res = {"head": hres}
    if seg is not None:
        tm, tres = iou_table(ti, tu, f"{args.teacher} (CC-BY-NC, cannot ship)  vs GT")
        res["teacher"] = tres
        g_h, g_t = hres.get("grass", 0) * 100, tres.get("grass", 0) * 100
        print(f"\n  GRASS IoU: head {g_h:.1f} vs teacher {g_t:.1f}  ({g_h - g_t:+.1f})")
        print(f"  mIoU     : head {hm * 100:.1f} vs teacher {tm * 100:.1f}")

    out = Path(args.out or (_SCRIPT_ROOT.parent / "output" / "ground-head"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "mu": mu, "sd": sd, "dim": dim,
                "hidden": args.hidden, "variant": args.variant, "size": args.size,
                "mode": args.mode, "names": GNAMES,
                "provenance": "trained on ADE20K images — research/validation only, not shippable"},
               out / "ground_head.pt")
    (out / "ground_head.json").write_text(json.dumps({"config": vars(args), "results": res},
                                                     indent=2, default=str))
    print(f"\n  wrote {out}/ground_head.pt (+ .json)")


def predict(args):
    """Run a trained head on real park frames — the target domain, where the teacher fails."""
    from lar_session import Session
    from colmap_io import read_model
    from footprint2d import sample_frames

    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    backbone, dim, device, dtype = load_backbone(ck["variant"])
    head = GroundHead(dim, ck["hidden"]).to(device)
    head.load_state_dict(ck["state"]); head.eval()
    mu, sd = ck["mu"], ck["sd"]

    s = Session(args.predict_session)
    frames = sample_frames(read_model(str(s.colmap_model)), args.predict_num)
    out = Path(args.out or (_SCRIPT_ROOT.parent / "output" / "ground-head")) / "park"
    out.mkdir(parents=True, exist_ok=True)
    tally = np.zeros(len(GNAMES), np.int64)
    for img in frames:
        p = s.images / img.name
        f, rgb, (h, w) = feature_grid(backbone, p, ck["size"], ck["mode"], device, dtype)
        with torch.no_grad():
            xv = torch.tensor((f.reshape(-1, dim) - mu) / sd, dtype=torch.float32, device=device)
            pp = head(xv).argmax(1).cpu().numpy().reshape(h, w).astype(np.uint8)
        pred = cv2.resize(pp, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        tally += np.bincount(pred.ravel(), minlength=len(GNAMES))
        col = GCOLORS[pred]
        blend = (0.45 * rgb + 0.55 * col).astype(np.uint8)
        cv2.imwrite(str(out / f"{Path(img.name).stem}_ground.png"),
                    np.hstack([rgb, blend])[:, :, ::-1])
    tot = max(tally.sum(), 1)
    print(f"[ground_head] {len(frames)} park frames ->  " +
          ", ".join(f"{n} {100 * c / tot:.1f}%" for n, c in zip(GNAMES, tally) if c))
    print(f"  wrote {out}/*_ground.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="ade20k", choices=["ade20k"],
                    help="supervision corpus. ADE20K validates the approach but its images are "
                         "not clear for a shipped model; GOOSE + relabel.py labels come next.")
    ap.add_argument("--root", default="/home/play/Code/datasets/ade20k")
    ap.add_argument("--variant", default="small")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--mode", default="square", choices=["square", "shortest"])
    ap.add_argument("--train-num", type=int, default=800)
    ap.add_argument("--val-num", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--teacher", default="mask2former-large",
                    help="segmenter kind for the baseline column (make_segmenter names, not "
                         "HF paths). CC-BY-NC weights: comparison only, never shipped.")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--head", default=None, help="checkpoint for --predict-session")
    ap.add_argument("--predict-session", default=None)
    ap.add_argument("--predict-num", type=int, default=8)
    args = ap.parse_args()
    if args.predict_session:
        if not args.head:
            raise SystemExit("--predict-session needs --head")
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
