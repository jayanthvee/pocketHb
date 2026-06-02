"""Backbone-swap sanity check: EfficientNetV2-S vs ConvNeXt-Tiny.

PURPOSE
-------
Rule out the possibility that our chunk-3 result (OOF MAE 2.09 g/dL, R² -0.05,
slope 0.469) is specific to ConvNeXt-Tiny. We swap to Rudokaite et al.'s #2
backbone (EfficientNetV2-S, ImageNet-21k ft 1k) and rerun the EXACT same pipeline:

    iter_crops (nail) → frozen EfficientNetV2-S
        → Shades-of-Gray + ImageNet norm → mean+std per-patient aggregation
        → PLS+SVR+isotonic blend (StratifiedKFold n=5, seed=42, n_bins=5)

If R² near zero and slope < 1 reproduce, the failure is in the method/data,
not in the choice of backbone.

DO NOT modify embed.py / regressor.py / data.py. This script is read-only against
the pipeline; it only flips the timm model name passed to load_backbone().
"""
from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from pockethb.data import iter_crops, load_metadata
from pockethb.embed import aggregate_per_patient, embed_crops, load_backbone
from pockethb.regressor import stratified_kfold_cv

BACKBONE_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
SEED = 42
N_SPLITS = 5
N_BINS = 5
BATCH_SIZE = 16

RESULTS_JSON = REPO_ROOT / "user_data" / "_efficientnet_test.json"
RESULTS_LOG = REPO_ROOT / "user_data" / "_efficientnet_test.log"


def run() -> dict:
    print("=" * 70)
    print("BACKBONE-SWAP SANITY CHECK")
    print(f"  backbone = {BACKBONE_NAME}")
    print(f"  seed     = {SEED}")
    print(f"  n_splits = {N_SPLITS}, n_bins = {N_BINS}")
    print("=" * 70)

    # --- 1. load nail crops ---
    print("\n[1/4] loading metadata + iterating nail crops...")
    t0 = time.time()
    df = load_metadata()
    crops = list(iter_crops(df, region="nail"))
    print(f"  loaded {len(crops)} nail crops across {df['PATIENT_ID'].nunique()} patients "
          f"in {time.time() - t0:.1f}s")

    # --- 2. load frozen EfficientNetV2-S ---
    print(f"\n[2/4] loading frozen backbone '{BACKBONE_NAME}' on CPU...")
    t0 = time.time()
    backbone = load_backbone(BACKBONE_NAME, device="cpu")
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"  loaded in {time.time() - t0:.1f}s   ({n_params/1e6:.2f} M params, all frozen)")

    # --- 3. embed all crops ---
    print(f"\n[3/4] embedding {len(crops)} crops (SoG + ImageNet norm, batch={BATCH_SIZE})...")
    t0 = time.time()
    embeddings, crop_pids, _ = embed_crops(
        backbone,
        crops,
        batch_size=BATCH_SIZE,
        apply_sog=True,
        device="cpu",
        progress=True,
    )
    embed_secs = time.time() - t0
    feat_dim = embeddings.shape[1]
    print(f"  embedded in {embed_secs:.1f}s  → embeddings shape = {embeddings.shape} "
          f"(feature_dim = {feat_dim})")

    # --- 4. aggregate per patient (mean + std) ---
    print("\n[4/4] aggregating per-patient (mean+std)...")
    X, pids = aggregate_per_patient(embeddings, crop_pids)
    print(f"  patient feature matrix: {X.shape}  ({2*feat_dim} = 2 × {feat_dim})")

    # align labels to patient order
    hb_by_pid = dict(zip(df["PATIENT_ID"].astype(int), df["hb_g_per_dL"].astype(float)))
    y = np.array([hb_by_pid[int(p)] for p in pids], dtype=np.float64)
    print(f"  labels y: n={len(y)}  mean={y.mean():.3f}  std={y.std(ddof=0):.3f}  "
          f"min={y.min():.2f}  max={y.max():.2f}")

    # --- run stratified 5-fold CV ---
    print("\n" + "=" * 70)
    print("running stratified_kfold_cv (n_splits=5, n_bins=5, seed=42)...")
    print("=" * 70)
    t0 = time.time()
    cv = stratified_kfold_cv(X, y, pids, n_splits=N_SPLITS, n_bins=N_BINS, seed=SEED)
    cv_secs = time.time() - t0
    print(f"  CV completed in {cv_secs:.1f}s")

    # --- metrics ---
    oof_mae = float(mean_absolute_error(cv.oof_true, cv.oof_pred))
    oof_rmse = float(np.sqrt(np.mean((cv.oof_true - cv.oof_pred) ** 2)))
    oof_r2 = float(r2_score(cv.oof_true, cv.oof_pred))

    # regression: predicted (y-axis) vs true (x-axis)
    slope, intercept = np.polyfit(cv.oof_true, cv.oof_pred, 1)
    pearson_r = float(np.corrcoef(cv.oof_true, cv.oof_pred)[0, 1])

    print("\n" + "=" * 70)
    print(f"RESULTS — {BACKBONE_NAME}")
    print("=" * 70)
    print(f"  OOF MAE  = {oof_mae:.4f} g/dL")
    print(f"  OOF RMSE = {oof_rmse:.4f} g/dL")
    print(f"  OOF R2   = {oof_r2:.4f}")
    print(f"  slope (pred vs true) = {slope:.4f}")
    print(f"  intercept            = {intercept:.4f}")
    print(f"  Pearson r            = {pearson_r:.4f}")

    print("\n--- per-fold breakdown ---")
    print(f"  {'fold':>4} {'n_tr':>5} {'n_te':>5} {'MAE':>8} {'RMSE':>8} {'R2':>8} "
          f"{'pls_nc':>7} {'svr_C':>7} {'svr_g':>9} {'w_pls':>7}")
    for fm in cv.fold_metrics:
        print(f"  {fm['fold']:>4d} {fm['n_train']:>5d} {fm['n_test']:>5d} "
              f"{fm['MAE']:>8.4f} {fm['RMSE']:>8.4f} {fm['R2']:>+8.4f} "
              f"{fm['pls_n_components']:>7d} {fm['svr_C']:>7.2f} {str(fm['svr_gamma']):>9s} "
              f"{fm['weight_pls']:>7.3f}")

    # --- comparison vs ConvNeXt-Tiny baseline ---
    print("\n--- comparison vs ConvNeXt-Tiny baseline (chunk 3) ---")
    print(f"  {'metric':<16} {'ConvNeXt-T':>12} {'EfficientNetV2-S':>20}  delta")
    print(f"  {'OOF MAE (g/dL)':<16} {2.09:>12.4f} {oof_mae:>20.4f}  {oof_mae - 2.09:+.4f}")
    print(f"  {'OOF R²':<16} {-0.05:>12.4f} {oof_r2:>20.4f}  {oof_r2 - (-0.05):+.4f}")
    print(f"  {'slope':<16} {0.469:>12.4f} {slope:>20.4f}  {slope - 0.469:+.4f}")

    # qualitative pattern check: R² near zero AND slope < 1
    pattern_reproduces_strict = (oof_r2 < 0.1) and (slope < 1.0)
    pattern_dramatically_better = (oof_r2 > 0.3) or (slope > 0.7)
    if pattern_dramatically_better:
        verdict = ("DRAMATIC IMPROVEMENT — EfficientNetV2-S produces meaningfully better numbers. "
                   "INVESTIGATE whether ConvNeXt-Tiny specifically has an issue.")
    elif pattern_reproduces_strict:
        verdict = ("REPRODUCES — R² near zero and slope < 1 with EfficientNetV2-S too. "
                   "Failure is in the method/data, not the backbone choice.")
    else:
        verdict = ("BORDERLINE — neither dramatically better nor clean reproduction. "
                   "Manual judgement needed.")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  {verdict}")
    print("=" * 70)

    results = {
        "backbone": BACKBONE_NAME,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_bins": N_BINS,
        "n_crops": int(len(crops)),
        "n_patients": int(len(y)),
        "feature_dim": int(feat_dim),
        "patient_matrix_shape": list(X.shape),
        "backbone_params_M": float(n_params / 1e6),
        "embed_seconds": float(embed_secs),
        "cv_seconds": float(cv_secs),
        "oof_metrics": {
            "MAE_g_per_dL": oof_mae,
            "RMSE_g_per_dL": oof_rmse,
            "R2": oof_r2,
            "slope_pred_vs_true": float(slope),
            "intercept_pred_vs_true": float(intercept),
            "pearson_r": pearson_r,
        },
        "fold_metrics": cv.fold_metrics,
        "labels_summary": {
            "n": int(len(y)),
            "mean": float(y.mean()),
            "std_ddof0": float(y.std(ddof=0)),
            "min": float(y.min()),
            "max": float(y.max()),
        },
        "convnext_baseline_for_reference": {
            "OOF_MAE_g_per_dL": 2.09,
            "OOF_R2": -0.05,
            "slope_pred_vs_true": 0.469,
        },
        "qualitative_pattern_reproduces": bool(pattern_reproduces_strict),
        "dramatic_improvement_flag": bool(pattern_dramatically_better),
        "verdict": verdict,
    }
    return results


def main() -> int:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Capture all stdout (including tqdm via stderr-default — tqdm writes to stderr
    # by default, so it won't enter our buffer; we capture the readable prints only).
    buf = io.StringIO()

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    tee = Tee(sys.stdout, buf)
    with redirect_stdout(tee):
        try:
            results = run()
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}")
            raise

    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    RESULTS_LOG.write_text(buf.getvalue(), encoding="utf-8")

    print(f"\nwrote {RESULTS_JSON}")
    print(f"wrote {RESULTS_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
