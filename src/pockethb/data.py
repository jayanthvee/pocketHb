"""
Data loading for the Nature Sci Data 2024 fingernail+Hb dataset.

Conventions:
- Hb is always exposed in g/dL (the raw CSV is g/L; we divide by 10 on load).
- Bboxes are (x1, y1, x2, y2) in image pixel coords.
- "Crops" mean per-nail RGB arrays in shape (H, W, 3) dtype uint8.
- Splits are subject-disjoint: a PATIENT_ID lives in exactly one of train/val/test.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_ROOT = Path("data/extracted")


@dataclass
class Crop:
    patient_id: int
    region: str          # "nail" or "skin"
    crop_idx: int        # 0, 1, or 2 — which of the 3 bboxes per image
    hb_g_per_dL: float
    image: np.ndarray    # (H, W, 3) uint8


def load_metadata(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Load and parse metadata.csv. Returns a DataFrame with parsed bboxes and g/dL Hb."""
    root = Path(root)
    df = pd.read_csv(root / "metadata.csv")
    df["hb_g_per_dL"] = df["HB_LEVEL_GperL"] / 10.0
    df["nail_bboxes"] = df["NAIL_BOUNDING_BOXES"].apply(ast.literal_eval)
    df["skin_bboxes"] = df["SKIN_BOUNDING_BOXES"].apply(ast.literal_eval)
    df["image_path"] = df["PATIENT_ID"].apply(lambda pid: root / "photo" / f"{pid}.jpg")
    return df


def subject_disjoint_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split PATIENT_IDs into train/val/test. Returns dict of lists of patient IDs."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    pids = df["PATIENT_ID"].unique().tolist()
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        "train": sorted(pids[:n_train]),
        "val": sorted(pids[n_train : n_train + n_val]),
        "test": sorted(pids[n_train + n_val :]),
    }


def iter_crops(
    df: pd.DataFrame,
    patient_ids: list[int] | None = None,
    region: str = "nail",
) -> Iterator[Crop]:
    """Yield Crop objects. If patient_ids given, restricts to those subjects."""
    bbox_col = {"nail": "nail_bboxes", "skin": "skin_bboxes"}[region]
    if patient_ids is not None:
        df = df[df["PATIENT_ID"].isin(patient_ids)]
    for _, row in df.iterrows():
        img = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        H, W = img.shape[:2]
        for i, (x1, y1, x2, y2) in enumerate(row[bbox_col]):
            # some dataset labels have y1 > y2 (or x1 > x2); normalise.
            x1, x2 = sorted((int(x1), int(x2)))
            y1, y2 = sorted((int(y1), int(y2)))
            # NOTE: the public Nature 2024 release contains 600x800 images, but
            # many skin bboxes were labelled in a taller (~700+ tall) source frame
            # and now reach below the image bottom edge. Clip to image bounds and
            # use whatever pixels survive — most skin bboxes only overshoot by a
            # few rows, so the in-bounds remainder is still a usable skin patch.
            x1c = max(0, min(x1, W))
            x2c = max(0, min(x2, W))
            y1c = max(0, min(y1, H))
            y2c = max(0, min(y2, H))
            crop = img[y1c:y2c, x1c:x2c].copy()
            if crop.size == 0:
                continue
            yield Crop(
                patient_id=int(row["PATIENT_ID"]),
                region=region,
                crop_idx=i,
                hb_g_per_dL=float(row["hb_g_per_dL"]),
                image=crop,
            )


def mean_rgb_features(crop: np.ndarray) -> np.ndarray:
    """Per-crop colour feature vector. Returns shape (6,):
    [R_mean, G_mean, B_mean, R_norm, G_norm, B_norm] where norms are channel/sum.
    Normalised channels are robust to overall brightness; absolute channels carry pallor signal.
    """
    flat = crop.reshape(-1, 3).astype(np.float64)
    means = flat.mean(axis=0)  # (3,)
    total = means.sum() + 1e-8
    norms = means / total
    return np.concatenate([means / 255.0, norms])


def build_feature_table(
    df: pd.DataFrame,
    patient_ids: list[int],
    region: str = "nail",
) -> pd.DataFrame:
    """For each crop, return one row of features + label.

    Columns:
        patient_id, crop_idx, hb_g_per_dL, R, G, B, rN, gN, bN
    """
    rows = []
    for c in iter_crops(df, patient_ids=patient_ids, region=region):
        feats = mean_rgb_features(c.image)
        rows.append({
            "patient_id": c.patient_id,
            "crop_idx": c.crop_idx,
            "hb_g_per_dL": c.hb_g_per_dL,
            "R": feats[0],
            "G": feats[1],
            "B": feats[2],
            "rN": feats[3],
            "gN": feats[4],
            "bN": feats[5],
        })
    return pd.DataFrame(rows)
