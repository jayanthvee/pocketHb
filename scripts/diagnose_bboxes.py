"""One-shot diagnostic: count degenerate bboxes in metadata.csv for both regions."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "extracted" / "metadata.csv"


def has_any_valid(boxes) -> bool:
    for x1, y1, x2, y2 in boxes:
        x1n, x2n = sorted((int(x1), int(x2)))
        y1n, y2n = sorted((int(y1), int(y2)))
        if (x2n - x1n) * (y2n - y1n) > 0:
            return True
    return False


def count_degen(boxes) -> int:
    n = 0
    for x1, y1, x2, y2 in boxes:
        x1n, x2n = sorted((int(x1), int(x2)))
        y1n, y2n = sorted((int(y1), int(y2)))
        if (x2n - x1n) * (y2n - y1n) == 0:
            n += 1
    return n


def main() -> int:
    df = pd.read_csv(META)
    df["nail"] = df["NAIL_BOUNDING_BOXES"].apply(ast.literal_eval)
    df["skin"] = df["SKIN_BOUNDING_BOXES"].apply(ast.literal_eval)

    n_nail_raw = sum(len(b) for b in df["nail"])
    n_skin_raw = sum(len(b) for b in df["skin"])
    print(f"raw nail bboxes in csv: {n_nail_raw}")
    print(f"raw skin bboxes in csv: {n_skin_raw}")

    nail_deg = sum(count_degen(b) for b in df["nail"])
    skin_deg = sum(count_degen(b) for b in df["skin"])
    print(f"degenerate nail (zero area even after coord swap): {nail_deg}")
    print(f"degenerate skin (zero area even after coord swap): {skin_deg}")

    n_pts_nail = df["nail"].apply(has_any_valid).sum()
    n_pts_skin = df["skin"].apply(has_any_valid).sum()
    print(f"patients with >=1 valid nail bbox: {n_pts_nail}/{len(df)}")
    print(f"patients with >=1 valid skin bbox: {n_pts_skin}/{len(df)}")

    n_pts_skin_dead = (df["skin"].apply(lambda b: not has_any_valid(b))).sum()
    print(f"patients with ALL skin bboxes degenerate (no usable skin): {n_pts_skin_dead}")

    # also: how many skin bbox entries per row?
    skin_counts = df["skin"].apply(len)
    print(f"\nskin bboxes per row: min={skin_counts.min()} max={skin_counts.max()} mean={skin_counts.mean():.2f}")
    print(f"row counts by skin bbox count:")
    print(skin_counts.value_counts().sort_index().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
