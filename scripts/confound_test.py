"""Confound diagnostic: is the predictive signal in nail tissue or in
acquisition/background context?

Tests 4 feature sources through the SAME pipeline (frozen ConvNeXt-Tiny + mean+std
per-patient aggregation + stratified 5-fold CV by Hb quantile, 3 seeds each):

  nails_fixed : bbox-corrected real fingernails (the swap-on-load path)
  nails_buggy : original (pre-fix) bboxes on as-stored image — bottom-region paper/table
  full_frame  : the entire 800x600 photo, no cropping
  bg_corner   : a fixed top-left 160x160 background patch (never any nail)

If full_frame and bg_corner also predict Hb, the apparent signal is an acquisition
confound rather than nail physiology. If only nails_fixed predicts, the method works.

What we found on Yakimov n=250 (Nature Sci Data 2024):
  nails_fixed   R² = -0.058   no signal from actual nails
  nails_buggy   R² = +0.288   the apparent signal was here, NOT in the nails
  full_frame    R² = +0.189   partial — full frame includes the artifact region
  bg_corner     R² = -0.130   no signal from clean top-corner background

Conclusion: the apparent signal of the (broken) pipeline lives specifically in
the bottom-of-frame region the buggy bbox happened to point at — not in the nail
tissue, not in arbitrary background. Mechanism unidentified; the standardized
Yakimov rig (single camera, fixed LED, white-reference normalised) argues against
gross camera/lighting/date confound. Most plausible candidates: demographic leakage,
patient-specific paper/hand positioning, regression-to-the-mean.

Outputs the 4-variant table to stdout and JSON.

NOT a claim about Mannino PNAS 2025 (n=9061, different cohort) or BNAIC 2025
(n=159 Sanquin, different cohort).
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score

from pockethb.data import load_metadata, Crop
from pockethb.embed import load_backbone, embed_crops, aggregate_per_patient
from pockethb.regressor import stratified_kfold_cv


def crops_for(kind: str, df) -> list[Crop]:
    """Build per-patient crops from the dataset under one of 4 feature regimes.

    nails_fixed : transpose image so the (y,x,y,x) CSV coords land on real nails
                  (this is the correct interpretation per the bbox-swap audit fix)
    nails_buggy : use bboxes on as-stored image (broken — lands on paper/table)
    full_frame  : whole image as a single "crop" per patient
    bg_corner   : fixed 160x160 top-left patch per patient
    """
    out = []
    for _, row in df.iterrows():
        img = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        pid = int(row["PATIENT_ID"])
        hb = float(row["hb_g_per_dL"])

        if kind == "full_frame":
            out.append(Crop(pid, "full", 0, hb, img.copy()))
            continue
        if kind == "bg_corner":
            out.append(Crop(pid, "bg", 0, hb, img[0:160, 0:160].copy()))
            continue
        if kind in ("nails_fixed", "nails_buggy"):
            # nails_fixed: transposing the image is equivalent to swapping x<->y
            # in the bboxes — both put the bboxes on real nails. We keep the
            # nails_buggy path on the as-stored image to expose the original bug.
            src = np.transpose(img, (1, 0, 2)) if kind == "nails_fixed" else img
            H, W = src.shape[:2]
            for i, (x1, y1, x2, y2) in enumerate(row["nail_bboxes"]):
                # NOTE: row["nail_bboxes"] has already been swapped by load_metadata
                # for both variants, so we undo the swap when we want the buggy
                # behavior. Here we read coords as-is; for nails_buggy we are
                # treating the (post-swap, i.e. correct) tuple on an un-transposed
                # image — that places the bbox in the bottom-region of the
                # landscape frame, exactly reproducing the original bug behavior.
                x1, x2 = sorted((int(x1), int(x2)))
                y1, y2 = sorted((int(y1), int(y2)))
                x1c, x2c = max(0, min(x1, W)), max(0, min(x2, W))
                y1c, y2c = max(0, min(y1, H)), max(0, min(y2, H))
                crop = src[y1c:y2c, x1c:x2c].copy()
                if crop.size == 0:
                    continue
                out.append(Crop(pid, "nail", i, hb, crop))
    return out


def main():
    df = load_metadata("data/extracted")
    hb_map = dict(zip(df["PATIENT_ID"].astype(int), df["hb_g_per_dL"].astype(float)))
    bb = load_backbone("convnext_tiny.fb_in22k_ft_in1k", device="cpu")

    results = {}
    for kind in ["nails_fixed", "nails_buggy", "full_frame", "bg_corner"]:
        t = time.time()
        crops = crops_for(kind, df)
        embs, pids, _ = embed_crops(bb, crops, batch_size=32, apply_sog=True,
                                    device="cpu", progress=False)
        X, pid_order = aggregate_per_patient(embs, pids)
        y = np.array([hb_map[p] for p in pid_order], dtype=float)

        seed_rows = []
        for seed in (42, 0, 7):
            cv = stratified_kfold_cv(X, y, pid_order, n_splits=5, n_bins=5, seed=seed)
            pred, true = cv.oof_pred, cv.oof_true
            slope = float(np.polyfit(true, pred, 1)[0])
            seed_rows.append({
                "seed": seed,
                "MAE": round(float(mean_absolute_error(true, pred)), 3),
                "R2": round(float(r2_score(true, pred)), 3),
                "slope": round(slope, 3),
                "pearson_r": round(float(np.corrcoef(true, pred)[0, 1]), 3),
            })

        results[kind] = {
            "n_crops": len(crops),
            "n_patients": int(len(y)),
            "mean_predictor_MAE": round(float(mean_absolute_error(y, np.full_like(y, y.mean()))), 3),
            "per_seed": seed_rows,
            "mean_R2": round(float(np.mean([r["R2"] for r in seed_rows])), 3),
            "mean_pearson_r": round(float(np.mean([r["pearson_r"] for r in seed_rows])), 3),
            "secs": round(time.time() - t, 1),
        }
        print(kind, "->", json.dumps(results[kind]["per_seed"]),
              "mean_r=", results[kind]["mean_pearson_r"])

    out_path = Path("_confound_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print()
    print(f"wrote {out_path}")
    print()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
