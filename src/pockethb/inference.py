"""Inference pipeline: load HF bundle, embed photos, optionally personalise.

Designed to be the single import-and-use class for chunks 6–8.

    from pockethb.inference import InferenceSession
    sess = InferenceSession.from_hub()                     # loads bubbaonbubba/pockethb-base
    raw_hb = sess.predict_aggregate(photo_paths)           # global prediction
    sess.calibrate(photo_paths, true_hb_g_per_dL=15.3)     # fit affine bias correction
    personal_hb = sess.predict_aggregate(photo_paths)      # now personalised
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .calibration import AffineCalibrator, PersonalHead
from .embed import _prep_crop, load_backbone
from .preprocess import shades_of_gray


def _load_image(src) -> np.ndarray:
    """Accept str / Path / PIL.Image / numpy array. Return HxWx3 uint8."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (str, Path)):
        return np.asarray(Image.open(src).convert("RGB"))
    if isinstance(src, Image.Image):
        return np.asarray(src.convert("RGB"))
    raise TypeError(f"unsupported image source: {type(src)}")


@dataclass
class InferenceResult:
    raw_per_image: np.ndarray            # global prediction per input photo (g/dL)
    raw_aggregate: float                 # global prediction at session level (mean+std agg)
    personal_per_image: np.ndarray | None = None    # post-calibration per photo
    personal_aggregate: float | None = None         # post-calibration session level
    method: str = "global"               # "global" | "affine" | "mlp"
    n_photos: int = 0
    notes: str = ""


class InferenceSession:
    """Carries the global model bundle + (optional) per-user calibrator."""

    def __init__(self, bundle: dict, device: str = "cpu"):
        self.bundle = bundle
        self.backbone_name = bundle["backbone_name"]
        self.image_size = int(bundle["image_size"])
        self.sog_p = int(bundle["shades_of_gray_p"])
        self.blender = bundle["blender"]
        self.device = device
        self._backbone = None
        self.calibrator: AffineCalibrator | None = None
        self.personal_head: PersonalHead | None = None

    @classmethod
    def from_hub(cls, repo_id: str = "bubbaonbubba/pockethb-base", device: str = "cpu") -> "InferenceSession":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename="pockethb_base.pkl")
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(bundle, device=device)

    @classmethod
    def from_pkl(cls, path: str | Path, device: str = "cpu") -> "InferenceSession":
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(bundle, device=device)

    def _get_backbone(self):
        if self._backbone is None:
            self._backbone = load_backbone(self.backbone_name, device=self.device)
        return self._backbone

    @torch.no_grad()
    def embed_image(self, image) -> np.ndarray:
        """Apply Shades-of-Gray + resize + normalise → frozen backbone → 768-d feature."""
        img = _load_image(image)
        tensor = _prep_crop(img, apply_sog=True).unsqueeze(0).to(self.device)
        feat = self._get_backbone()(tensor).cpu().numpy()[0]
        return feat

    @torch.no_grad()
    def embed_many(self, images) -> np.ndarray:
        """Embed a list of images. Returns (n, 768) array."""
        feats = np.stack([self.embed_image(img) for img in images], axis=0)
        return feats

    def _aggregate(self, embs: np.ndarray) -> np.ndarray:
        """Apply the same mean+std per-patient aggregation the global model was trained with."""
        if embs.ndim == 1:
            embs = embs[None, :]
        if embs.shape[0] == 1:
            agg = np.concatenate([embs[0], np.zeros_like(embs[0])])
        else:
            agg = np.concatenate([embs.mean(axis=0), embs.std(axis=0, ddof=0)])
        return agg.astype(np.float32).reshape(1, -1)

    def predict_per_image(self, images) -> np.ndarray:
        """Per-image global prediction (each photo treated as its own session)."""
        embs = self.embed_many(images)
        preds = []
        for i in range(embs.shape[0]):
            agg = self._aggregate(embs[i : i + 1])
            preds.append(float(self.blender.predict(agg)[0]))
        return np.array(preds, dtype=np.float64)

    def predict_aggregate(self, images) -> float:
        """One Hb estimate from a session: aggregate all photos via mean+std and predict once."""
        embs = self.embed_many(images)
        agg = self._aggregate(embs)
        raw = float(self.blender.predict(agg)[0])
        if self.calibrator and self.calibrator.fitted:
            return float(self.calibrator.predict(np.array([raw]))[0])
        return raw

    def calibrate(self, images, true_hb_g_per_dL) -> AffineCalibrator:
        """Fit per-user affine calibration against a known bloodwork reading.

        true_hb_g_per_dL: scalar (single anchor) or array (multiple paired anchors).
        """
        per = self.predict_per_image(images)
        if np.isscalar(true_hb_g_per_dL):
            targets = np.full(len(per), float(true_hb_g_per_dL))
        else:
            targets = np.asarray(true_hb_g_per_dL, dtype=np.float64).ravel()
        self.calibrator = AffineCalibrator().fit(per, targets)
        return self.calibrator

    def calibrate_mlp(self, images, true_hb_g_per_dL, **head_kwargs) -> PersonalHead:
        """Fit a per-user MLP head on top of the frozen embeddings."""
        embs = self.embed_many(images)
        if np.isscalar(true_hb_g_per_dL):
            targets = np.full(embs.shape[0], float(true_hb_g_per_dL))
        else:
            targets = np.asarray(true_hb_g_per_dL, dtype=np.float64).ravel()
        self.personal_head = PersonalHead(in_dim=embs.shape[1], **head_kwargs).fit(embs, targets)
        return self.personal_head

    def run(self, images, true_hb_g_per_dL: float | None = None) -> InferenceResult:
        """Full session-level inference. If true_hb_g_per_dL is given, also fits + applies affine calibration."""
        raw_per = self.predict_per_image(images)
        raw_agg = float(np.mean(raw_per))

        if true_hb_g_per_dL is not None:
            cal = self.calibrate(images, true_hb_g_per_dL)
            personal_per = cal.predict(raw_per)
            personal_agg = float(np.mean(personal_per))
            return InferenceResult(
                raw_per_image=raw_per,
                raw_aggregate=raw_agg,
                personal_per_image=personal_per,
                personal_aggregate=personal_agg,
                method=f"affine_{cal.mode}",
                n_photos=len(raw_per),
                notes=f"calibrator: a={cal.a:.3f} b={cal.b:+.3f} anchors={cal.n_anchors_used}",
            )

        return InferenceResult(
            raw_per_image=raw_per,
            raw_aggregate=raw_agg,
            method="global",
            n_photos=len(raw_per),
            notes="no calibration applied",
        )
