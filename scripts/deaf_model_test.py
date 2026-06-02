"""Pre-registered deaf-model diagnostic for pocketHb.

Pre-registration locked 2026-06-02 (this docstring is the contract).

PURPOSE: disambiguate why R² ≈ 0 on chunk 3.
- Outcome A: model responsive on training, comparable user variability, user residual low → DEMOGRAPHIC BIAS supported.
- Outcome B: responsive, comparable variability, user residual inside cloud → bias is generic, demographic story weakens.
- Outcome C: responsive, user within-subject std much smaller than dataset → USER-SPECIFIC DEAFNESS, demographic story holds (different mechanism).
- Outcome D: dataset across-subject std < 1.34 → near-deaf on EVERYONE, the model never learned anything.

DECISION RULE (locked, do not move thresholds after seeing data):
  responsiveness:    bag-of-2 across-subject std ≥ 0.5 × true_Hb_std (= 1.34 g/dL)
  user comparability:user within-subject std ≤ 2× dataset within-subject std → comparable
                     ≤ 0.5× → user-specific deafness (outcome C)
                     >  2×  → user is genuinely more variable
  bin density:       if n patients in [14.3, 16.3] ≥ 25, use percentile-within-bin
                     else, use residual-based comparison (preferred default)
  demographic bias:  user residual below 10th percentile of dataset residual distribution

CAVEATS (pre-committed):
1. Bag-of-2 is OOD for a blender trained exclusively on bag-of-3. Within-subject and responsiveness
   numbers measured slightly off the regressor's training regime.
2. Per-patient within-subject std from n=3 LOO is noisy. Report only the averaged-over-250 number,
   not individual patient stds.
3. Dataset within-subject std from n=3 LOO is likely upward-biased (small-sample std estimator).
   If outcome C lands, it could be an artifact of the noisy dataset estimator rather than true
   user-specific deafness. This caveat decides A vs C.
4. Demographic-bias adjudication at x=15.3 happens at the high-Hb edge of the dataset
   (max = 16.9). High-leverage region of the linear regression line, wide CI. Borderline residuals
   should be read cautiously.
5. ALL FOUR OUTCOMES MEAN "the model fails, here's which way." None is "good for the model."
   A and C are "good for the paper" (publishable demographic finding); B and D are not.
   Do not conflate these two senses of "good outcome" in the writeup.
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

from pockethb.data import load_metadata, iter_crops
from pockethb.inference import InferenceSession

USER_HB = 15.3
PHOTO_DIR = Path("user_data")
BUNDLE = Path("weights/pockethb_base.pkl")
RESULTS_JSON = PHOTO_DIR / "_deaf_test_results.json"
RESULTS_PNG = PHOTO_DIR / "_deaf_test_figure.png"
SEED = 42

# locked thresholds
TRUE_HB_STD = 2.67           # from chunk 3
RESPONSIVENESS_THRESHOLD = 0.5 * TRUE_HB_STD   # = 1.335
USER_DEAFNESS_RATIO = 0.5    # outcome C
USER_OUTLIER_RATIO = 2.0     # outcome "more variable than dataset"
HIGH_HB_BIN = (14.3, 16.3)
DENSE_BIN_N = 25
RESIDUAL_PERCENTILE = 10     # for demographic-bias finding
N_USER_SUBSAMPLES = 100


def aggregate(embs: np.ndarray) -> np.ndarray:
    """Same aggregation as training: [mean, std] over the bag's embeddings."""
    if embs.shape[0] == 1:
        agg = np.concatenate([embs[0], np.zeros_like(embs[0])])
    else:
        agg = np.concatenate([embs.mean(axis=0), embs.std(axis=0, ddof=0)])
    return agg.astype(np.float32).reshape(1, -1)


def main() -> int:
    print("=" * 70)
    print("DEAF-MODEL DIAGNOSTIC — pocketHb chunk 7 successor")
    print("=" * 70)
    print("Pre-registration: see docstring. Rule is locked. No threshold movement.\n")

    rng = np.random.default_rng(SEED)
    sess = InferenceSession.from_pkl(BUNDLE)

    # ---- DATASET ----
    print("[1/4] embedding 250 dataset patients...")
    df = load_metadata()
    crops_by_patient: dict[int, list[np.ndarray]] = {}
    all_crops = []
    for c in iter_crops(df, region="nail"):
        crops_by_patient.setdefault(c.patient_id, []).append(c.image)
        all_crops.append((c.patient_id, c.image))

    t0 = time.time()
    all_embs = sess.embed_many([img for _, img in all_crops])
    print(f"  embedded {all_embs.shape[0]} crops across {len(crops_by_patient)} patients in {time.time()-t0:.1f}s")

    # group embeddings by patient (preserve order)
    emb_by_patient: dict[int, np.ndarray] = {}
    cursor = 0
    for pid, crops in crops_by_patient.items():
        emb_by_patient[pid] = all_embs[cursor : cursor + len(crops)]
        cursor += len(crops)

    hb_by_pid = dict(zip(df["PATIENT_ID"].astype(int), df["hb_g_per_dL"].astype(float)))

    # only patients with exactly 3 crops give a clean bag-of-3 + 3 LOO bag-of-2
    pids_3crop = sorted(p for p, e in emb_by_patient.items() if e.shape[0] == 3)
    pids_with_crops = sorted(emb_by_patient.keys())
    print(f"  patients with 3 crops (used for clean LOO): {len(pids_3crop)}")
    print(f"  patients with any crops (used for headline scatter): {len(pids_with_crops)}\n")

    # ---- HEADLINE: bag-of-3 prediction per patient (training-regime, all patients with ≥1 crop) ----
    print("[2/4] dataset bag-of-3 (or full-set) predictions for headline scatter...")
    pred_bag3 = {}
    for pid in pids_with_crops:
        e = emb_by_patient[pid]
        agg = aggregate(e)
        pred_bag3[pid] = float(sess.blender.predict(agg)[0])

    truths = np.array([hb_by_pid[p] for p in pids_with_crops])
    preds_headline = np.array([pred_bag3[p] for p in pids_with_crops])

    # linear regression line on all 250 (for the residual-based rule)
    slope, intercept = np.polyfit(truths, preds_headline, 1)
    residuals_dataset = preds_headline - (slope * truths + intercept)
    print(f"  headline bag-of-3 mean: {preds_headline.mean():.3f}  std: {preds_headline.std():.3f}")
    print(f"  regression line: pred = {slope:.3f} * true + {intercept:+.3f}")
    print(f"  residual std (around the line): {residuals_dataset.std():.3f}\n")

    # ---- bag-of-2 LOO per 3-crop patient ----
    print("[3/4] dataset bag-of-2 LOO (3 predictions per 3-crop patient)...")
    loo_preds_by_patient = {}
    for pid in pids_3crop:
        e = emb_by_patient[pid]
        loo_preds = []
        for drop in range(3):
            keep_idx = [i for i in range(3) if i != drop]
            agg = aggregate(e[keep_idx])
            loo_preds.append(float(sess.blender.predict(agg)[0]))
        loo_preds_by_patient[pid] = np.array(loo_preds)

    # dataset within-subject std at bag-of-2
    per_patient_within_std = np.array([loo_preds_by_patient[p].std(ddof=0) for p in pids_3crop])
    dataset_within_std_mean = float(per_patient_within_std.mean())
    dataset_within_std_median = float(np.median(per_patient_within_std))

    # dataset across-subject std at bag-of-2: use first LOO bag (omit crop 0) per patient
    pred_bag2_across = np.array([loo_preds_by_patient[p][0] for p in pids_3crop])
    dataset_across_bag2_std = float(pred_bag2_across.std(ddof=0))
    print(f"  dataset bag-of-2 across-subject std: {dataset_across_bag2_std:.3f}")
    print(f"  dataset within-subject std (mean over {len(pids_3crop)} patients of 3-LOO std): {dataset_within_std_mean:.3f}")
    print(f"  dataset within-subject std (median): {dataset_within_std_median:.3f}\n")

    # ---- USER ----
    print("[4/4] user 27-photo embeddings + bag-of-2 + bag-of-3 subsamples...")
    user_photos = sorted(p for p in PHOTO_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    if not user_photos:
        print("no user photos!")
        return 1
    t0 = time.time()
    user_embs = sess.embed_many([Image.open(p).convert("RGB") for p in user_photos])
    print(f"  embedded {user_embs.shape[0]} user photos in {time.time()-t0:.1f}s")

    # user bag-of-2 within-subject std: 100 random pairs
    user_bag2_preds = []
    for _ in range(N_USER_SUBSAMPLES):
        idx = rng.choice(user_embs.shape[0], size=2, replace=False)
        agg = aggregate(user_embs[idx])
        user_bag2_preds.append(float(sess.blender.predict(agg)[0]))
    user_bag2_preds = np.array(user_bag2_preds)
    user_within_std = float(user_bag2_preds.std(ddof=0))

    # user bag-of-3 predictions distribution (for overlay): 100 random triples
    user_bag3_preds = []
    for _ in range(N_USER_SUBSAMPLES):
        idx = rng.choice(user_embs.shape[0], size=3, replace=False)
        agg = aggregate(user_embs[idx])
        user_bag3_preds.append(float(sess.blender.predict(agg)[0]))
    user_bag3_preds = np.array(user_bag3_preds)
    user_bag3_median = float(np.median(user_bag3_preds))
    user_bag3_mean = float(user_bag3_preds.mean())
    print(f"  user bag-of-2 within-subject std (n={N_USER_SUBSAMPLES}): {user_within_std:.3f}")
    print(f"  user bag-of-3 median (n={N_USER_SUBSAMPLES} subsamples): {user_bag3_median:.3f}")
    print(f"  user bag-of-3 mean: {user_bag3_mean:.3f}")
    print(f"  user bag-of-3 std: {user_bag3_preds.std(ddof=0):.3f}\n")

    # ---- BIN CHECK ----
    high_bin_mask = (truths >= HIGH_HB_BIN[0]) & (truths <= HIGH_HB_BIN[1])
    n_in_high_bin = int(high_bin_mask.sum())
    print(f"BIN CHECK: patients with true Hb in {HIGH_HB_BIN}: n = {n_in_high_bin}")
    use_residual_rule = n_in_high_bin < DENSE_BIN_N
    if use_residual_rule:
        print(f"  → using residual-based comparison (n < {DENSE_BIN_N})\n")
    else:
        print(f"  → using percentile-within-bin rule (n ≥ {DENSE_BIN_N})\n")

    # ---- APPLY DECISION RULE ----
    print("=" * 70)
    print("DECISION RULE OUTCOME")
    print("=" * 70)

    responsive = dataset_across_bag2_std >= RESPONSIVENESS_THRESHOLD
    print(f"1. Responsiveness on training distribution:")
    print(f"   dataset bag-of-2 across-subject std = {dataset_across_bag2_std:.3f}")
    print(f"   threshold (0.5 × {TRUE_HB_STD}) = {RESPONSIVENESS_THRESHOLD:.3f}")
    print(f"   → {'PASS (responsive)' if responsive else 'FAIL (near-deaf)'}")

    user_to_dataset_ratio = user_within_std / dataset_within_std_mean if dataset_within_std_mean > 0 else float("inf")
    print(f"\n2. User vs dataset within-subject variability:")
    print(f"   user within std = {user_within_std:.3f}")
    print(f"   dataset within std (mean) = {dataset_within_std_mean:.3f}")
    print(f"   ratio (user / dataset) = {user_to_dataset_ratio:.3f}")
    user_specific_deafness = user_to_dataset_ratio <= USER_DEAFNESS_RATIO
    user_more_variable = user_to_dataset_ratio > USER_OUTLIER_RATIO
    print(f"   → user-specific deafness if ratio ≤ {USER_DEAFNESS_RATIO}: {user_specific_deafness}")
    print(f"   → user more variable than dataset if ratio > {USER_OUTLIER_RATIO}: {user_more_variable}")

    # demographic bias adjudication
    print(f"\n3. Demographic-bias adjudication (anchor at true_Hb = {USER_HB}):")
    if use_residual_rule:
        user_expected_line = slope * USER_HB + intercept
        user_residual = user_bag3_median - user_expected_line
        threshold_residual = float(np.percentile(residuals_dataset, RESIDUAL_PERCENTILE))
        demographic_bias_supported = user_residual < threshold_residual
        print(f"   user bag-of-3 median: {user_bag3_median:.3f}")
        print(f"   regression line at x={USER_HB}: {user_expected_line:.3f}")
        print(f"   user residual: {user_residual:+.3f}")
        print(f"   dataset residual {RESIDUAL_PERCENTILE}th percentile: {threshold_residual:+.3f}")
        print(f"   → demographic bias supported: {demographic_bias_supported}")
    else:
        # percentile-within-bin
        bin_preds = preds_headline[high_bin_mask]
        bin_threshold = float(np.percentile(bin_preds, RESIDUAL_PERCENTILE))
        demographic_bias_supported = user_bag3_median < bin_threshold
        print(f"   user bag-of-3 median: {user_bag3_median:.3f}")
        print(f"   {RESIDUAL_PERCENTILE}th percentile of bin predictions: {bin_threshold:.3f}")
        print(f"   → demographic bias supported: {demographic_bias_supported}")

    # outcome assignment
    print("\n" + "=" * 70)
    if not responsive:
        outcome = "D"
        narrative = "near-deaf on EVERYONE — global model is essentially a mean-predictor; the −3.68 user bias is partly compression-toward-mean artifact"
    elif user_specific_deafness:
        outcome = "C"
        narrative = "user-specific deafness — model responsive on training subjects but compressing on user; demographic story holds via 'collapse' mechanism"
    elif demographic_bias_supported:
        outcome = "A"
        narrative = "demographic-bias story holds — model responsive on training, user has comparable variability, user residual is significantly low at x=15.3"
    else:
        outcome = "B"
        narrative = "demographic angle weakened — model responsive, user variability comparable, but user residual inside the dataset distribution; the −3.68 is generic model error not skin-tone-specific"
    print(f"OUTCOME: {outcome}")
    print(f"NARRATIVE: {narrative}")
    print("=" * 70)
    print("\nREMINDER: all four outcomes mean 'the model fails, here's which way.'")
    print("A and C are good for the PAPER (publishable demographic finding); B and D are not.")
    print("None is good for the MODEL. The model is dead in all four. — mentor")

    # ---- SAVE JSON ----
    results = {
        "user_anchor_hb_g_per_dL": USER_HB,
        "true_hb_std_g_per_dL": TRUE_HB_STD,
        "responsiveness_threshold": RESPONSIVENESS_THRESHOLD,
        "dataset_n_total": int(len(pids_with_crops)),
        "dataset_n_3crop_used_for_loo": int(len(pids_3crop)),
        "bag_of_2_across_subject_std": dataset_across_bag2_std,
        "responsive_on_training": bool(responsive),
        "dataset_within_subject_std_mean": dataset_within_std_mean,
        "dataset_within_subject_std_median": dataset_within_std_median,
        "dataset_within_caveat": "n=3 LOO per patient → upward-biased std estimator. Averaging 250 stabilizes the mean but the individual stds are noisy. This number decides A vs C.",
        "user_within_subject_std": user_within_std,
        "user_n_bag2_subsamples": N_USER_SUBSAMPLES,
        "user_to_dataset_within_ratio": user_to_dataset_ratio,
        "user_bag3_predictions": {
            "median": user_bag3_median,
            "mean": user_bag3_mean,
            "std": float(user_bag3_preds.std(ddof=0)),
            "n_subsamples": N_USER_SUBSAMPLES,
        },
        "high_hb_bin_count": n_in_high_bin,
        "high_hb_bin": list(HIGH_HB_BIN),
        "used_residual_rule": use_residual_rule,
        "regression_line": {"slope": float(slope), "intercept": float(intercept)},
        "demographic_bias_supported": bool(demographic_bias_supported),
        "outcome": outcome,
        "narrative": narrative,
        "model_is_dead_reminder": "All four outcomes mean 'model fails, here's which way.' None is good for the model.",
    }
    if use_residual_rule:
        results["user_residual_at_15p3"] = float(user_bag3_median - (slope * USER_HB + intercept))
        results["dataset_residual_10pct_threshold"] = float(np.percentile(residuals_dataset, RESIDUAL_PERCENTILE))
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_JSON}")

    # ---- FIGURE ----
    fig, ax = plt.subplots(figsize=(11, 7))
    # scatter: 250 patients
    ax.scatter(truths, preds_headline, alpha=0.45, s=35, color="steelblue", label=f"dataset (n={len(pids_with_crops)})")
    # regression line
    xs = np.linspace(truths.min() - 0.5, max(truths.max(), USER_HB) + 0.5, 200)
    ax.plot(xs, slope * xs + intercept, color="darkblue", lw=1.3, ls="--",
            label=f"regression: y = {slope:.3f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}")
    # y=x line for reference
    ax.plot(xs, xs, color="grey", lw=1, ls=":", label="y = x (perfect)")
    # user's bag-of-3 distribution at x=15.3
    ax.scatter([USER_HB] * len(user_bag3_preds), user_bag3_preds, alpha=0.3, s=20, color="red",
               label=f"user bag-of-3 (n={N_USER_SUBSAMPLES} subsamples)")
    ax.scatter([USER_HB], [user_bag3_median], color="red", s=180, marker="*", edgecolor="black",
               linewidth=1.5, zorder=5, label=f"user median = {user_bag3_median:.2f}")
    # vertical line at user's anchor + horizontal line at expected
    ax.axvline(USER_HB, color="red", lw=0.5, ls=":", alpha=0.5)
    if use_residual_rule:
        ax.axhline(slope * USER_HB + intercept, color="darkred", lw=0.5, ls=":", alpha=0.5)

    ax.set_xlabel("true Hb (g/dL)")
    ax.set_ylabel("predicted Hb (g/dL)")
    ax.set_title(f"pocketHb deaf-model diagnostic — outcome {outcome}\n"
                 f"dataset bag-of-2 across-std = {dataset_across_bag2_std:.2f} (threshold {RESPONSIVENESS_THRESHOLD:.2f})  |  "
                 f"user/dataset within ratio = {user_to_dataset_ratio:.2f}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=130)
    print(f"wrote {RESULTS_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
