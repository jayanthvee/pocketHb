"""Frozen-backbone embedding extraction via timm.

The BNAIC 2025 paper (Rudokaite et al.) showed that, at small dataset scale
(n<200 subjects), freezing the backbone and treating it as a fixed feature
extractor beats fine-tuning by a large margin. ConvNeXt-Tiny was their best
backbone; EfficientNetV2-S was a close second.

This module loads such a backbone, strips the classifier head, and returns
a per-image embedding via global average pooling on the last feature map.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

from .preprocess import shades_of_gray

# default to ConvNeXt-Tiny — Rudokaite 2025 best backbone (MAE 0.603 mmol/L)
# alt: "tf_efficientnetv2_s.in21k_ft_in1k" (their second best, MAE 0.613)
DEFAULT_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_backbone(name: str = DEFAULT_BACKBONE, device: str = "cpu") -> nn.Module:
    """Load a frozen, classifier-stripped timm backbone. Returns module in eval mode."""
    import timm  # imported lazily so the rest of the package works without timm

    model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def _prep_crop(crop_uint8: np.ndarray, apply_sog: bool = True) -> torch.Tensor:
    """Preprocess a single crop: optional Shades-of-Gray → resize → ImageNet normalise."""
    img = crop_uint8
    if apply_sog:
        img = shades_of_gray(img, p=6)
    pil = Image.fromarray(img).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
    return tensor


@torch.no_grad()
def embed_crops(
    backbone: nn.Module,
    crops,
    batch_size: int = 16,
    apply_sog: bool = True,
    device: str = "cpu",
    progress: bool = True,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Run the frozen backbone on a list of Crop objects.

    Returns:
        embeddings: (n_crops, feature_dim) float32 ndarray
        patient_ids: list[int] aligned with embeddings rows
        crop_idxs: list[int] aligned with embeddings rows
    """
    embeddings = []
    pids: list[int] = []
    cidxs: list[int] = []

    it = range(0, len(crops), batch_size)
    if progress:
        it = tqdm(it, desc="embedding", total=(len(crops) + batch_size - 1) // batch_size)

    for start in it:
        batch_crops = crops[start : start + batch_size]
        batch_x = torch.stack([_prep_crop(c.image, apply_sog=apply_sog) for c in batch_crops])
        batch_x = batch_x.to(device, non_blocking=True)
        feat = backbone(batch_x)  # (B, feature_dim)
        embeddings.append(feat.cpu().numpy().astype(np.float32))
        pids.extend(int(c.patient_id) for c in batch_crops)
        cidxs.extend(int(c.crop_idx) for c in batch_crops)

    return np.concatenate(embeddings, axis=0), pids, cidxs


def aggregate_per_patient(
    embeddings: np.ndarray,
    patient_ids: list[int],
) -> tuple[np.ndarray, list[int]]:
    """For each patient, aggregate crop embeddings by mean and std → 2d-dim vector.

    BNAIC §2.4: 'aggregated by element-wise mean and standard deviation, yielding
    2×d–dimensional participant-level vectors. This design captures both central
    tendency and variability across crops, improving robustness to image quality
    variation.'

    Returns:
        patient_vectors: (n_patients, 2 * feature_dim) float32
        patient_id_order: list[int] aligned with rows
    """
    pids_arr = np.asarray(patient_ids)
    unique_pids = sorted(set(patient_ids))
    rows = []
    for pid in unique_pids:
        mask = pids_arr == pid
        sub = embeddings[mask]
        if sub.shape[0] < 1:
            warnings.warn(f"patient {pid} has 0 crops, skipping")
            continue
        if sub.shape[0] == 1:
            # std is undefined for a single sample → fill with zeros (no variability info)
            agg = np.concatenate([sub.mean(axis=0), np.zeros(sub.shape[1], dtype=np.float32)])
        else:
            agg = np.concatenate([sub.mean(axis=0), sub.std(axis=0, ddof=0)])
        rows.append(agg.astype(np.float32))
    return np.stack(rows, axis=0), unique_pids
