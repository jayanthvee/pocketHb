"""Inference pipeline: load HF bundle, embed photos, optionally personalise.

Designed to be the single import-and-use class for chunks 6–8.

    from pockethb.inference import InferenceSession
    sess = InferenceSession.from_hub()                     # loads bubbaonbubba/pockethb-base
    raw_hb = sess.predict_aggregate(photo_paths, bboxes=[(x1,y1,x2,y2), ...])
    sess.calibrate(photo_paths, true_hb_g_per_dL=15.3, bboxes=[...])
    personal_hb = sess.predict_aggregate(photo_paths, bboxes=[...])

The training pipeline embeds bbox-cropped nail patches (~50px regions inside an
800x600 photo). Inference MUST do the same — pass a bbox per image, or pass an
already-cropped PIL Image whose extent is roughly nail-shaped. Passing a raw
4032x3024 iPhone frame with no bbox is treated as caller error and warned.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .calibration import AffineCalibrator, PersonalHead
from .embed import _prep_crop, load_backbone
from .preprocess import shades_of_gray

Bbox = tuple[int, int, int, int]  # (x1, y1, x2, y2)

# heuristic: a nail crop fed to embed_image should not be much larger than
# this in either dimension. anything bigger is almost certainly a full-frame
# photo, which is off-distribution for the model.
_MAX_CROP_DIM_BEFORE_WARN = 600


def _load_image(src) -> np.ndarray:
    """Accept str / Path / PIL.Image / numpy array. Return HxWx3 uint8."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (str, Path)):
        with Image.open(src) as im:
            return np.asarray(im.convert("RGB"))
    if isinstance(src, Image.Image):
        return np.asarray(src.convert("RGB"))
    raise TypeError(f"unsupported image source: {type(src)}")


def _apply_bbox(img: np.ndarray, bbox: Bbox | None) -> np.ndarray:
    """Crop img to bbox (clipped to bounds) or return as-is."""
    if bbox is None:
        return img
    x1, y1, x2, y2 = bbox
    H, W = img.shape[:2]
    x1c, x2c = max(0, min(int(x1), W)), max(0, min(int(x2), W))
    y1c, y2c = max(0, min(int(y1), H)), max(0, min(int(y2), H))
    if x1c >= x2c or y1c >= y2c:
        raise ValueError(f"bbox {bbox} clips to empty region inside {W}x{H} image")
    return img[y1c:y2c, x1c:x2c]


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
    def embed_image(self, image, bbox: Bbox | None = None) -> np.ndarray:
        """Apply Shades-of-Gray + resize + normalise → frozen backbone → 768-d feature.

        If bbox is given, crops to (x1, y1, x2, y2) before the SoG + resize. If
        bbox is None and the input image is much larger than a nail crop should
        be, warns once — the model was trained on small nail patches, not whole
        photos.
        """
        img = _load_image(image)
        crop = _apply_bbox(img, bbox)
        if bbox is None and max(crop.shape[:2]) > _MAX_CROP_DIM_BEFORE_WARN:
            warnings.warn(
                f"embed_image() called on a {crop.shape[1]}x{crop.shape[0]} image with "
                "no bbox. The model was trained on ~50px nail crops; whole-frame photos "
                "are off-distribution. Pass bbox=(x1,y1,x2,y2) or pre-crop the nail.",
                stacklevel=2,
            )
        tensor = _prep_crop(crop, apply_sog=True).unsqueeze(0).to(self.device)
        feat = self._get_backbone()(tensor).cpu().numpy()[0]
        return feat

    @torch.no_grad()
    def embed_many(self, images, bboxes: list[Bbox | None] | None = None) -> np.ndarray:
        """Embed a list of images. Returns (n, 768) array.

        bboxes: optional list aligned to images. None entries → embed whole image.
        """
        if bboxes is None:
            bboxes = [None] * len(images)
        if len(bboxes) != len(images):
            raise ValueError(f"bboxes ({len(bboxes)}) must align with images ({len(images)})")
        feats = np.stack(
            [self.embed_image(img, bbox=bb) for img, bb in zip(images, bboxes)],
            axis=0,
        )
        return feats

    def _aggregate_bag(self, embs: np.ndarray) -> np.ndarray:
        """Per-patient mean+std aggregation — matches what the global model was trained on.

        Requires n>=2 embeddings. For single-image input, see _aggregate_single (OOD).
        """
        if embs.ndim == 1 or embs.shape[0] < 2:
            n = 1 if embs.ndim == 1 else embs.shape[0]
            raise ValueError(
                f"_aggregate_bag requires n>=2 embeddings, got {n}. "
                "For single-image inference use _aggregate_single (off-distribution)."
            )
        agg = np.concatenate([embs.mean(axis=0), embs.std(axis=0, ddof=0)])
        return agg.astype(np.float32).reshape(1, -1)

    def _aggregate_single(self, emb: np.ndarray) -> np.ndarray:
        """Single-image aggregation — [emb, zeros]. OOD relative to training (which had
        nonzero std from bag-of-3 crops). Emits a UserWarning at every call.
        """
        warnings.warn(
            "single-image aggregation produces [mean, zeros], off-distribution for a "
            "blender trained on bag-of-3 patients. Use predict_per_image() with >=2 "
            "photos for leave-one-out bag aggregation instead.",
            stacklevel=3,
        )
        if emb.ndim == 2:
            emb = emb[0]
        agg = np.concatenate([emb, np.zeros_like(emb)])
        return agg.astype(np.float32).reshape(1, -1)

    def predict_per_image(
        self,
        images,
        bboxes: list[Bbox | None] | None = None,
        mode: str = "loo",
    ) -> np.ndarray:
        """Per-image global prediction.

        mode='loo' (default): for each image i, build a bag from the OTHER n-1
            embeddings and predict on [mean, std] of that bag. Stays inside the
            training distribution (provided n-1 >= 2 ideally; n-1=1 still zero-stds).
            Measures stability of the bag estimate to removal of any single photo.
        mode='single': each image predicted as its own bag-of-1 → [emb, zeros]
            aggregation. OOD. Each call emits a UserWarning. Kept for diagnostics
            and reproducing pre-fix behavior.
        """
        embs = self.embed_many(images, bboxes=bboxes)
        n = embs.shape[0]
        if mode == "loo":
            if n < 2:
                raise ValueError(
                    "predict_per_image(mode='loo') requires >=2 photos. Got 1. "
                    "Use predict_aggregate() with a single bag, or mode='single' "
                    "if you want the off-distribution single-image prediction."
                )
            preds = []
            for i in range(n):
                bag = np.delete(embs, i, axis=0)
                if bag.shape[0] < 2:
                    # n=2 → bag-of-1, fall back to single (OOD) aggregation
                    agg = self._aggregate_single(bag[0])
                else:
                    agg = self._aggregate_bag(bag)
                preds.append(float(self.blender.predict(agg)[0]))
            return np.array(preds, dtype=np.float64)
        elif mode == "single":
            preds = []
            for i in range(n):
                agg = self._aggregate_single(embs[i])
                preds.append(float(self.blender.predict(agg)[0]))
            return np.array(preds, dtype=np.float64)
        else:
            raise ValueError(f"unknown mode {mode!r}; use 'loo' or 'single'")

    def predict_aggregate(self, images, bboxes: list[Bbox | None] | None = None) -> float:
        """One Hb estimate from a session: aggregate all photos via mean+std and predict once.

        Canonical inference call — keeps the input distribution matched to training.
        Requires n>=2 photos. For n=1 falls back to OOD single-image aggregation
        with a warning.
        """
        embs = self.embed_many(images, bboxes=bboxes)
        if embs.shape[0] < 2:
            agg = self._aggregate_single(embs[0])
        else:
            agg = self._aggregate_bag(embs)
        raw = float(self.blender.predict(agg)[0])
        if self.calibrator and self.calibrator.fitted:
            return float(self.calibrator.predict(np.array([raw]))[0])
        return raw

    def calibrate(
        self,
        images,
        true_hb_g_per_dL,
        bboxes: list[Bbox | None] | None = None,
    ) -> AffineCalibrator:
        """Fit per-user affine calibration against a known bloodwork reading.

        Single-anchor path (true_hb_g_per_dL is scalar): aggregates ALL photos into
        one bag, predicts once, fits bias-only against that single (raw, true) point.
        This matches training distribution; the OOD per-image vectors used by the
        pre-fix path are no longer touched here.

        Requires n>=2 photos in the session.

        For multi-anchor / multi-session calibration (slope + bias), use
        calibrate_sessions() which takes a list of session photo-lists plus a list
        of true Hb values.
        """
        if not np.isscalar(true_hb_g_per_dL):
            raise ValueError(
                "calibrate() takes a scalar true_hb_g_per_dL (single bloodwork anchor). "
                "For multi-session calibration with multiple anchors, use calibrate_sessions()."
            )
        embs = self.embed_many(images, bboxes=bboxes)
        if embs.shape[0] < 2:
            raise ValueError(
                "calibrate() requires >=2 photos in the session so the bag aggregation "
                "matches training distribution. Got 1."
            )
        agg = self._aggregate_bag(embs)
        raw = float(self.blender.predict(agg)[0])
        self.calibrator = AffineCalibrator().fit(
            np.array([raw]), np.array([float(true_hb_g_per_dL)])
        )
        return self.calibrator

    def calibrate_sessions(
        self,
        sessions: list,
        true_hbs: list[float],
        bboxes_per_session: list[list[Bbox | None] | None] | None = None,
    ) -> AffineCalibrator:
        """Fit per-user affine (slope + bias) from multiple bloodwork anchors.

        sessions: list of K image-lists (each K >= 2 photos taken at one bloodwork draw)
        true_hbs: list of K g/dL values, one per session
        bboxes_per_session: optional, list of K bbox-lists aligned with sessions

        With K >= 2 distinct true_hbs, fits a full affine. With K == 1 or all true_hbs
        identical, falls back to bias-only (same as calling calibrate() with one session).
        """
        if len(sessions) != len(true_hbs):
            raise ValueError(f"sessions ({len(sessions)}) and true_hbs ({len(true_hbs)}) must align")
        if bboxes_per_session is None:
            bboxes_per_session = [None] * len(sessions)
        raws = []
        for sess_imgs, sess_bboxes in zip(sessions, bboxes_per_session):
            if len(sess_imgs) < 2:
                raise ValueError("each session needs >=2 photos for bag aggregation")
            embs = self.embed_many(sess_imgs, bboxes=sess_bboxes)
            agg = self._aggregate_bag(embs)
            raws.append(float(self.blender.predict(agg)[0]))
        self.calibrator = AffineCalibrator().fit(
            np.asarray(raws, dtype=np.float64),
            np.asarray(true_hbs, dtype=np.float64),
        )
        return self.calibrator

    def calibrate_mlp(self, images, true_hb_g_per_dL, bboxes: list[Bbox | None] | None = None, **head_kwargs) -> PersonalHead:
        """Fit a per-user MLP head on top of the frozen embeddings."""
        embs = self.embed_many(images, bboxes=bboxes)
        if np.isscalar(true_hb_g_per_dL):
            targets = np.full(embs.shape[0], float(true_hb_g_per_dL))
        else:
            targets = np.asarray(true_hb_g_per_dL, dtype=np.float64).ravel()
        self.personal_head = PersonalHead(in_dim=embs.shape[1], **head_kwargs).fit(embs, targets)
        return self.personal_head

    def _predict_loo(self, embs: np.ndarray) -> np.ndarray:
        """Leave-one-out per-image predictions from a precomputed embedding matrix."""
        n = embs.shape[0]
        preds = []
        for i in range(n):
            bag = np.delete(embs, i, axis=0)
            agg = self._aggregate_single(bag[0]) if bag.shape[0] < 2 else self._aggregate_bag(bag)
            preds.append(float(self.blender.predict(agg)[0]))
        return np.array(preds, dtype=np.float64)

    def run(self, images, true_hb_g_per_dL: float | None = None, bboxes: list[Bbox | None] | None = None) -> InferenceResult:
        """Full session-level inference. Embeds once, computes both the canonical
        bag-aggregate prediction (raw_aggregate) and per-image leave-one-out
        predictions (raw_per_image). If true_hb_g_per_dL is given, fits and applies
        bias-only affine calibration against the bag-aggregate prediction.

        Requires n>=2 photos. n=1 falls back to single-image OOD aggregation with
        warnings (the only available path with one photo).
        """
        embs = self.embed_many(images, bboxes=bboxes)
        n = embs.shape[0]

        if n >= 2:
            raw_per = self._predict_loo(embs)
            raw_agg = float(self.blender.predict(self._aggregate_bag(embs))[0])
        else:
            single_pred = float(self.blender.predict(self._aggregate_single(embs[0]))[0])
            raw_per = np.array([single_pred])
            raw_agg = single_pred

        if true_hb_g_per_dL is not None:
            if n < 2:
                raise ValueError(
                    "calibration requires >=2 photos in the session. "
                    "Got 1 — call without true_hb_g_per_dL for an uncalibrated single-photo estimate."
                )
            self.calibrator = AffineCalibrator().fit(
                np.array([raw_agg]), np.array([float(true_hb_g_per_dL)])
            )
            cal = self.calibrator
            personal_per = cal.predict(raw_per)
            personal_agg = float(cal.predict(np.array([raw_agg]))[0])
            return InferenceResult(
                raw_per_image=raw_per,
                raw_aggregate=raw_agg,
                personal_per_image=personal_per,
                personal_aggregate=personal_agg,
                method=f"affine_{cal.mode}",
                n_photos=n,
                notes=f"calibrator: a={cal.a:.3f} b={cal.b:+.3f} anchors={cal.n_anchors_used}",
            )

        return InferenceResult(
            raw_per_image=raw_per,
            raw_aggregate=raw_agg,
            method="global",
            n_photos=n,
            notes="no calibration applied",
        )
