"""Pluggable semantic segmenters.

All segmenters map a BGR image to a dense (H, W) uint8 array of :class:`~taxonomy.Klass`
ids. The rest of the pipeline only sees class ids, so the backend is swappable:

  - :class:`HeuristicSegmenter` -- zero-dependency colour rules. Not accurate, but it
    exercises the full pipeline on real imagery without a GPU/model download, which is how
    we validate plumbing.
  - :class:`ClipSegSegmenter` -- real open-vocab segmentation (CLIPSeg). Prompt-driven per
    the taxonomy. Requires the optional ``segmentation`` dependency group (torch,
    transformers). This is the increment-1 "real" backend; Grounded-SAM-2 can slot in
    later behind the same interface.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from taxonomy import Klass, PromptTable, color_lut, keyword_klass


class Segmenter:
    """Interface: BGR image (H, W, 3) uint8 -> class-id map (H, W) uint8."""

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def segment_file(self, path: str | Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image {path}")
        return self.segment(img)


class HeuristicSegmenter(Segmenter):
    """Cheap HSV colour rules. For plumbing validation only -- NOT a real segmenter."""

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        H, W = v.shape
        out = np.full((H, W), int(Klass.UNKNOWN), dtype=np.uint8)

        rows = np.arange(H)[:, None] * np.ones((1, W))
        lower_half = rows > H * 0.45  # ground tends to sit in the lower image

        greenish = (h > 35) & (h < 90) & (s > 60)
        blueish = (h > 95) & (h < 130) & (s > 60)
        grayish = (s < 45)
        brownish = (h > 10) & (h < 30) & (s > 40) & (v < 200)

        out[greenish & ~lower_half] = int(Klass.TREE)
        out[greenish & lower_half] = int(Klass.GRASS)
        out[brownish & lower_half] = int(Klass.TERRAIN)
        out[grayish & lower_half] = int(Klass.PATH)
        out[blueish] = int(Klass.WATER)
        return out


class ClipSegSegmenter(Segmenter):
    """Open-vocab per-pixel segmentation via CLIPSeg (CIDAS/clipseg-rd64-refined).

    Runs one forward pass over all taxonomy prompts, upsamples the per-prompt heatmaps to
    the image size, and takes the arg-max prompt (mapped back to its Klass). Pixels whose
    best prompt score is below ``threshold`` stay UNKNOWN.
    """

    def __init__(self, model_name: str = "CIDAS/clipseg-rd64-refined",
                 threshold: float = 0.30, device: str | None = None, batch: int = 8):
        import torch  # lazy: only needed for the real backend
        from transformers import AutoProcessor, CLIPSegForImageSegmentation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device).eval()
        self.threshold = threshold
        self.batch = batch
        self.table = PromptTable.build()
        self.prompt_klass = np.array([int(k) for k in self.table.prompt_klass], dtype=np.uint8)

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        prompts = self.table.prompts

        heat = np.empty((len(prompts), H, W), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(prompts), self.batch):
                chunk = prompts[i:i + self.batch]
                inputs = self.processor(
                    text=chunk,
                    images=[rgb] * len(chunk),
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                logits = self.model(**inputs).logits  # (B, h, w)
                if logits.ndim == 2:
                    logits = logits[None]
                probs = torch.sigmoid(logits).unsqueeze(1)  # (B,1,h,w)
                up = torch.nn.functional.interpolate(
                    probs, size=(H, W), mode="bilinear", align_corners=False
                )[:, 0]
                heat[i:i + len(chunk)] = up.cpu().numpy()

        best = heat.argmax(axis=0)
        best_score = heat.max(axis=0)
        out = self.prompt_klass[best]
        out[best_score < self.threshold] = int(Klass.UNKNOWN)
        return out.astype(np.uint8)


class OneFormerSegmenter(Segmenter):
    """Closed-set dense semantic segmentation via OneFormer (MIT), trained on ADE20K.

    ADE20K's 150 classes cover real park "stuff" (tree/grass/earth/path/road/water/...), which
    we remap to our taxonomy by keyword. Higher-quality region boundaries than CLIPSeg, but a
    *fixed* vocabulary (no custom prompts).
    """

    def __init__(self, model_name: str = "shi-labs/oneformer_ade20k_swin_tiny",
                 device: str | None = None):
        import torch  # lazy
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(self.device).eval()

        # Build ADE-id -> Klass LUT once from the model's own label map.
        id2label = self.model.config.id2label
        n = max(int(i) for i in id2label) + 1
        self.lut = np.zeros(n, dtype=np.uint8)
        for i, name in id2label.items():
            self.lut[int(i)] = int(keyword_klass(name))

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        inputs = self.processor(images=rgb, task_inputs=["semantic"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
        ade = seg.cpu().numpy().astype(np.int64)
        return self.lut[ade].astype(np.uint8)


class Mask2FormerSegmenter(Segmenter):
    """Closed-set dense semantic segmentation via Mask2Former (MIT), ADE20K. Same ADE->Klass
    remap as OneFormer; different architecture, so boundary quality can differ."""

    def __init__(self, model_name: str = "facebook/mask2former-swin-large-ade-semantic",
                 device: str | None = None):
        import torch  # lazy
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device).eval()
        id2label = self.model.config.id2label
        n = max(int(i) for i in id2label) + 1
        self.lut = np.zeros(n, dtype=np.uint8)
        for i, name in id2label.items():
            self.lut[int(i)] = int(keyword_klass(name))

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
        return self.lut[seg.cpu().numpy().astype(np.int64)].astype(np.uint8)


class LingBotSegmenter(Segmenter):
    """Frozen LingBot-Vision ViT features -> optional AnyUp upsample -> trained
    linear head -> dense taxonomy mask.

    The head + normalization + config come from a checkpoint written by
    ``script/backbone/probe.py linprobe --save-head`` (trained on OUR taxonomy via
    Mask2Former pseudo-labels, so no ADE->Klass keyword remap). Requires the
    ``lingbot_vision`` package importable; if the head was trained with AnyUp,
    also the ``anyup`` repo (path stored in the ckpt, override with ``anyup_src``).
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet, matches
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # lingbot load_image

    def __init__(self, ckpt: str, device: str | None = None,
                 anyup_src: str | None = None):
        import sys
        import torch  # lazy
        from lingbot_vision import extract_patch_tokens, load_pretrained_backbone

        self.torch = torch
        self._extract = extract_patch_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.size = int(c["size"])
        self.upsample = c["upsample"]
        self.up_size = int(c["up_size"])
        self.mu = torch.tensor(c["mu"], dtype=torch.float32, device=self.device)
        self.sd = torch.tensor(c["sd"], dtype=torch.float32, device=self.device)

        backbone, _ = load_pretrained_backbone(
            variant=c["variant"], device=self.device, dtype=self.dtype)
        self.backbone = backbone
        self.patch_size = backbone.patch_size

        head = torch.nn.Linear(int(c["embed_dim"]), int(c["n_klass"]))
        head.load_state_dict(c["head"])
        self.head = head.to(self.device).eval()

        self.up = None
        if self.upsample == "anyup":
            src = anyup_src or c.get("anyup_src")
            if src and src not in sys.path:
                sys.path.insert(0, src)
            from anyup.model import AnyUp
            urls = {
                "multi": "https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth",
                "paper": "https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth",
            }
            m = AnyUp().to(self.device).eval()
            m.load_state_dict(torch.hub.load_state_dict_from_url(
                urls[c["anyup_ckpt"]], map_location=self.device, progress=False))
            self.up = m

    def _preprocess(self, image_bgr: np.ndarray):
        """BGR array -> [1,3,s,s] ImageNet-normalized tensor (lingbot square mode)."""
        s = max(self.patch_size, (self.size // self.patch_size) * self.patch_size)
        rgb = cv2.cvtColor(cv2.resize(image_bgr, (s, s), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._MEAN) / self._STD
        return self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = image_bgr.shape[:2]
        img_norm = self._preprocess(image_bgr)
        with torch.no_grad():
            tokens, (h, w) = self._extract(self.backbone, img_norm, self.device, self.dtype)
            if self.up is None:
                feat = tokens[0].float()                       # [h*w, C]
                gh, gw = h, w
            else:
                lr = tokens[0].float().reshape(1, h, w, -1).permute(0, 3, 1, 2).contiguous()
                guide = img_norm.to(self.device).float()
                hr = self.up(guide, lr, output_size=(self.up_size, self.up_size), q_chunk_size=256)
                feat = hr[0].permute(1, 2, 0).reshape(-1, hr.shape[1])  # [up*up, C]
                gh, gw = self.up_size, self.up_size
            x = (feat - self.mu) / self.sd
            pred = self.head(x).argmax(1).to(torch.uint8).cpu().numpy().reshape(gh, gw)
        return cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def save_mask_png(mask: np.ndarray, path: str | Path) -> None:
    """Write a class-id map as a colour PNG (viewable) alongside the raw ids in the R channel."""
    lut = np.array(color_lut(), dtype=np.uint8)
    rgb = lut[mask]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


# Single source of truth for valid backend names (CLI choices derive from this).
SEGMENTER_KINDS = ("heuristic", "clipseg", "oneformer", "oneformer-large",
                   "mask2former", "mask2former-large", "lingbot")


def make_segmenter(kind: str, **kw) -> Segmenter:
    if kind == "heuristic":
        return HeuristicSegmenter()
    if kind == "clipseg":
        return ClipSegSegmenter(**kw)
    if kind == "oneformer":
        return OneFormerSegmenter(**kw)
    if kind == "oneformer-large":
        return OneFormerSegmenter(model_name="shi-labs/oneformer_ade20k_swin_large", **kw)
    if kind == "mask2former":
        return Mask2FormerSegmenter(**kw)
    if kind == "mask2former-large":
        return Mask2FormerSegmenter(model_name="facebook/mask2former-swin-large-ade-semantic", **kw)
    if kind == "lingbot":
        return LingBotSegmenter(**kw)  # requires ckpt=<linprobe --save-head output>
    raise ValueError(f"unknown segmenter kind {kind!r} "
                     f"(heuristic/clipseg/oneformer[-large]/mask2former[-large]/lingbot)")
