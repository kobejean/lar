"""Fetch the pretrained weights that live under ``models/`` (gitignored).

``models/`` holds a few hundred MB of third-party checkpoints that have no business in git.
Everything here is downloaded from its original host and verified by MD5, so a fresh clone
can reproduce the directory with one command instead of a description of where to click.

Only weights that are *not* pulled automatically belong here. Anything reached through
``transformers.from_pretrained`` / ``torch.hub`` already caches itself under
``~/.cache/huggingface`` and needs no entry.

Run (from repo root):
  uv run python script/download_models.py            # fetch whatever is missing
  uv run python script/download_models.py --list     # show what is known, and its state
  uv run python script/download_models.py --force footprints-handheld
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# name -> (url, md5 of the download, path that must exist afterwards, note)
MODELS: dict[str, tuple[str, str, str, str]] = {
    "footprints-handheld": (
        "https://storage.googleapis.com/niantic-lon-static/research/footprints/handheld.zip",
        "ab97945cf8f8f9e8d9bdedf8961506b6",
        "handheld/model.pth",
        "Niantic 'Footprints and Free Space from a Single Color Image' (CVPR 2020), "
        "handheld variant — the closest baseline to our single-image ground head. "
        "Non-commercial research licence: baseline comparison only, never shipped.",
    ),
    "footprints-kitti": (
        "https://storage.googleapis.com/niantic-lon-static/research/footprints/kitti.zip",
        "a52e3b04bffd86f62c62cf8859c47798",
        "kitti/model.pth",
        "Same, KITTI-trained (driving). Only useful for reproducing their published numbers.",
    ),
    "footprints-matterport": (
        "https://storage.googleapis.com/niantic-lon-static/research/footprints/matterport.zip",
        "e28929d0819392d2178c880725531c4e",
        "matterport/model.pth",
        "Same, Matterport-trained (indoor).",
    ),
}

DEFAULT = ["footprints-handheld"]


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def have(name: str) -> bool:
    return (MODELS_DIR / MODELS[name][2]).exists()


def fetch(name: str, force: bool = False) -> bool:
    url, md5, marker, _ = MODELS[name]
    target = MODELS_DIR / marker
    if target.exists() and not force:
        print(f"  [have] {name} -> {target.relative_to(REPO_ROOT)}")
        return True
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {name}\n         {url}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "download.zip"
        last = [-1]

        def hook(count, bsize, total):
            if total <= 0:
                return
            pct = min(100, int(count * bsize * 100 / total))
            if pct >= last[0] + 10:
                last[0] = pct
                print(f"         {pct:3d}%  ({total/1e6:.0f} MB)", flush=True)

        try:
            urllib.request.urlretrieve(url, tmp, hook)
        except Exception as e:                      # noqa: BLE001 - report and continue
            print(f"         FAILED: {e}")
            return False

        got = md5sum(tmp)
        if got != md5:
            # A corrupt/truncated archive that silently unzips is worse than no archive.
            print(f"         MD5 MISMATCH: expected {md5}, got {got} — discarded")
            return False
        print("         md5 ok, extracting")
        with zipfile.ZipFile(tmp) as z:
            z.extractall(MODELS_DIR)

    if not target.exists():
        print(f"         extracted but {marker} is missing — archive layout changed?")
        return False
    print(f"         -> {target.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", default=None,
                    help=f"models to fetch (default: {' '.join(DEFAULT)}; 'all' for everything)")
    ap.add_argument("--list", action="store_true", help="show known models and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.list:
        print(f"models dir: {MODELS_DIR}")
        for n, (_, _, marker, note) in MODELS.items():
            mark = "have" if have(n) else "   -"
            star = " *" if n in DEFAULT else "  "
            print(f"  [{mark}]{star}{n:<24s} {marker}\n            {note}")
        print("\n  * = fetched by default. ")
        return 0

    names = args.names or DEFAULT
    if names == ["all"]:
        names = list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}\nknown: {', '.join(MODELS)}")
        return 2

    print(f"models dir: {MODELS_DIR}")
    ok = [fetch(n, args.force) for n in names]
    n_ok = sum(ok)
    print(f"\n{n_ok}/{len(names)} ready"
          + ("" if n_ok == len(names) else "  — see failures above"))
    if n_ok and shutil.disk_usage(MODELS_DIR).free < 1 << 30:
        print("warning: under 1 GB free on this filesystem")
    return 0 if n_ok == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
