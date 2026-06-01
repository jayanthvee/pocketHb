"""PyTorch Dataset wrappers for the nail-crop Hb regression task."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .data import Crop, iter_crops

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = False):
    """Augmentation pipeline.

    `train=True` adds geometric augmentation + mild brightness jitter only. Colour
    jitter (hue/saturation) is intentionally OFF — it would destroy the pallor signal.
    """
    base = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
    if train:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.10, contrast=0.0, saturation=0.0, hue=0.0),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(base)


class NailCropDataset(Dataset):
    """In-memory dataset over a list of `Crop` objects. Small enough (≤720) to fit."""

    def __init__(self, crops: list[Crop], train: bool = False, image_size: int = 224):
        self.crops = list(crops)
        self.transform = build_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int):
        c = self.crops[idx]
        img_arr = c.image  # (H, W, 3) uint8
        x = self.transform(img_arr)
        y = torch.tensor(c.hb_g_per_dL, dtype=torch.float32)
        return x, y, c.patient_id, c.crop_idx


def crops_for_patients(
    df,
    patient_ids: list[int],
    region: str = "nail",
) -> list[Crop]:
    """Eagerly materialise crops into memory for the given patient subset."""
    return list(iter_crops(df, patient_ids=patient_ids, region=region))
