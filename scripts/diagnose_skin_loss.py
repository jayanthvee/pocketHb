"""Trace exactly why so many skin crops disappear in iter_crops."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ast

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "extracted" / "metadata.csv"
PHOTOS = ROOT / "data" / "extracted" / "photo"


def main() -> int:
    df = pd.read_csv(META)
    df["skin_bboxes"] = df["SKIN_BOUNDING_BOXES"].apply(ast.literal_eval)

    n_total = 0
    n_oob = 0
    n_partial_clipped = 0
    n_zero = 0
    n_ok = 0
    n_image_size = {}
    examples_oob = []

    for _, row in df.iterrows():
        pid = int(row["PATIENT_ID"])
        img_path = PHOTOS / f"{pid}.jpg"
        img = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        n_image_size.setdefault((H, W), 0)
        n_image_size[(H, W)] += 1

        for i, (x1, y1, x2, y2) in enumerate(row["skin_bboxes"]):
            n_total += 1
            x1n, x2n = sorted((int(x1), int(x2)))
            y1n, y2n = sorted((int(y1), int(y2)))
            zero_in_coords = (x2n - x1n) * (y2n - y1n) == 0
            if zero_in_coords:
                n_zero += 1
                continue
            # check actual slice
            x1c = max(0, min(x1n, W))
            x2c = max(0, min(x2n, W))
            y1c = max(0, min(y1n, H))
            y2c = max(0, min(y2n, H))
            slice_area = (x2c - x1c) * (y2c - y1c)
            if slice_area == 0:
                # bbox fully outside image
                n_oob += 1
                if len(examples_oob) < 5:
                    examples_oob.append((pid, i, (x1n, y1n, x2n, y2n), (H, W)))
                continue
            if slice_area != (x2n - x1n) * (y2n - y1n):
                # bbox clipped
                n_partial_clipped += 1
            n_ok += 1

    print(f"total skin bboxes processed: {n_total}")
    print(f"zero-area in coords: {n_zero}")
    print(f"out-of-bounds (empty slice after clipping to image): {n_oob}")
    print(f"partially clipped (some pixels outside image, slice still non-empty): {n_partial_clipped}")
    print(f"fully in-bounds, non-empty: {n_ok}")
    print()
    print("image size distribution:")
    for sz, cnt in sorted(n_image_size.items()):
        print(f"  H={sz[0]} W={sz[1]}: {cnt} images")
    print()
    if examples_oob:
        print("first few OOB examples (pid, crop_idx, bbox, image HxW):")
        for ex in examples_oob:
            print(f"  pid={ex[0]} crop_idx={ex[1]} bbox={ex[2]} image_HxW={ex[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
