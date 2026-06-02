"""Run the personalization pipeline on the user's iPhone photos.

CORRECTED v2: aggregate photos into bags of `BAG_SIZE` to match the training
distribution (the model was trained on patient vectors = mean+std over 3 crops
per patient, with non-zero std). The original v1 ran each photo individually
through aggregate(), which fed the model std=zeros — out-of-distribution input
that caused the predictions to collapse near the dataset mean for all inputs.

Output:
  user_data/_results.json   — numbers
  user_data/_results.png    — chart
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pockethb.calibration import AffineCalibrator
from pockethb.inference import InferenceSession

USER_HB = 15.3
PHOTO_DIR = Path("user_data")
BUNDLE = Path("weights/pockethb_base.pkl")
RESULTS_JSON = PHOTO_DIR / "_results.json"
RESULTS_PNG = PHOTO_DIR / "_results.png"
BAG_SIZE = 3                  # matches training (3 nail crops per patient)
SEED = 42


def main() -> int:
    photos = sorted(p for p in PHOTO_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    if not photos:
        print("no photos in user_data/", flush=True)
        return 1
    print(f"found {len(photos)} photos", flush=True)

    sess = InferenceSession.from_pkl(BUNDLE)

    print(f"embedding {len(photos)} photos through frozen ConvNeXt-Tiny (CPU)...", flush=True)
    t0 = time.time()
    embs = sess.embed_many([Image.open(p).convert("RGB") for p in photos])
    print(f"  done in {time.time() - t0:.1f}s  embeddings shape: {embs.shape}", flush=True)

    # 1) one prediction over ALL photos (matches single-patient training shape)
    full_agg = np.concatenate([embs.mean(axis=0), embs.std(axis=0, ddof=0)]).reshape(1, -1).astype(np.float32)
    raw_full = float(sess.blender.predict(full_agg)[0])
    print(f"\nFULL-BAG prediction (n={len(photos)} photos as one patient): {raw_full:.3f} g/dL")
    print(f"  bias vs truth ({USER_HB}): {raw_full - USER_HB:+.3f}")

    # 2) bags of BAG_SIZE — gives a per-bag distribution
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(photos))
    n_bags = len(photos) // BAG_SIZE
    bag_preds = []
    for b in range(n_bags):
        idx = order[b * BAG_SIZE : (b + 1) * BAG_SIZE]
        bag = embs[idx]
        agg = np.concatenate([bag.mean(axis=0), bag.std(axis=0, ddof=0)]).reshape(1, -1).astype(np.float32)
        bag_preds.append(float(sess.blender.predict(agg)[0]))
    bag_preds = np.array(bag_preds, dtype=np.float64)

    raw_bag_mean = float(bag_preds.mean())
    raw_bag_std = float(bag_preds.std())
    raw_bag_mae = float(np.mean(np.abs(bag_preds - USER_HB)))
    print(f"\nBAG-OF-{BAG_SIZE} predictions (n={n_bags} bags from {n_bags * BAG_SIZE} photos):")
    print(f"  per-bag: min={bag_preds.min():.3f} max={bag_preds.max():.3f} mean={raw_bag_mean:.3f} std={raw_bag_std:.3f}")
    print(f"  per-bag MAE vs your truth: {raw_bag_mae:.3f} g/dL")
    print(f"  bias (mean - truth): {raw_bag_mean - USER_HB:+.3f} g/dL")

    # 3) personalization: bias-only fit on the per-bag predictions
    cal = AffineCalibrator().fit(bag_preds, np.full(n_bags, USER_HB))
    personal_bags = cal.predict(bag_preds)
    personal_mae = float(np.mean(np.abs(personal_bags - USER_HB)))
    personal_std = float(personal_bags.std())
    print(f"\nAFTER affine personalization (bias-only since single anchor):")
    print(f"  calibrator: mode={cal.mode}  a={cal.a:.4f}  b={cal.b:+.4f}  n_anchors={cal.n_anchors_used}")
    print(f"  per-bag personalised: mean={personal_bags.mean():.3f} std={personal_std:.3f}")
    print(f"  per-bag MAE vs truth: {personal_mae:.3f} g/dL")

    print(f"\nDELTA vs pre-cal:  MAE {raw_bag_mae:.3f} -> {personal_mae:.3f}  ({personal_mae - raw_bag_mae:+.3f})")
    print(f"Note: bias-only calibration cannot reduce per-bag *variance*, only the mean offset.")

    # comparison with the (buggy) per-photo path for reference
    print(f"\nFor comparison only (DEPRECATED per-photo inference path, off-distribution):")
    per_photo = sess.predict_per_image([Image.open(p).convert("RGB") for p in photos])
    print(f"  per-photo (std-from-zeros input): min={per_photo.min():.3f} max={per_photo.max():.3f} std={per_photo.std():.3f}")
    print(f"  per-photo std of 0.11 g/dL is the collapsed-output artifact, not a real signal-to-noise number.")

    # save
    results = {
        "user_anchor_hb_g_per_dL": USER_HB,
        "n_photos": int(len(photos)),
        "bag_size": BAG_SIZE,
        "n_bags": int(n_bags),
        "full_bag": {
            "prediction_g_per_dL": raw_full,
            "bias_vs_truth": raw_full - USER_HB,
        },
        "bagged_raw": {
            "per_bag": [round(float(x), 4) for x in bag_preds],
            "mean": raw_bag_mean,
            "std": raw_bag_std,
            "mae_vs_truth": raw_bag_mae,
            "bias_vs_truth": raw_bag_mean - USER_HB,
        },
        "bagged_personalized": {
            "per_bag": [round(float(x), 4) for x in personal_bags],
            "mean": float(personal_bags.mean()),
            "std": personal_std,
            "mae_vs_truth": personal_mae,
            "calibrator": {"mode": cal.mode, "a": cal.a, "b": cal.b, "n_anchors": cal.n_anchors_used},
        },
        "deprecated_per_photo": {
            "explanation": "per-photo inference feeds the model std=zeros for the second half of the 1536-d patient vector. The model was never trained on that distribution and collapses near the dataset mean.",
            "per_photo": [round(float(x), 4) for x in per_photo],
            "std": float(per_photo.std()),
        },
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_JSON}")

    # chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    idx = np.arange(n_bags)
    ax.axhline(USER_HB, color="black", ls="--", lw=1.2, label=f"your truth = {USER_HB}")
    ax.scatter(idx, bag_preds, color="red", s=70, alpha=0.7, label=f"raw  mean={raw_bag_mean:.2f}  std={raw_bag_std:.2f}")
    ax.scatter(idx, personal_bags, color="green", s=70, alpha=0.7, label=f"personalised  mean={personal_bags.mean():.2f}  std={personal_std:.2f}")
    for i in idx:
        ax.plot([i, i], [bag_preds[i], personal_bags[i]], color="grey", lw=0.5, alpha=0.5)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"bag {i+1}" for i in idx], rotation=0, fontsize=9)
    ax.set_ylabel("Hb estimate (g/dL)")
    ax.set_title(f"pocketHb chunk 7 (bag-of-{BAG_SIZE}, n={n_bags} bags from {len(photos)} photos)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    width = 0.35
    ax.bar(idx - width / 2, bag_preds, width, color="red", alpha=0.6, label="raw")
    ax.bar(idx + width / 2, personal_bags, width, color="green", alpha=0.6, label="personalised")
    ax.axhline(USER_HB, color="black", ls="--", lw=1.2, label=f"truth = {USER_HB}")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"bag {i+1}" for i in idx], rotation=0, fontsize=9)
    ax.set_ylabel("Hb estimate (g/dL)")
    ax.set_title("raw vs personalised per bag")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=120)
    print(f"wrote {RESULTS_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
