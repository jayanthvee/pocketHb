"""Follow-up to deaf_model_test.py addressing the mentor's confound critique.

Two checks added:
  (A) Edge-matched within-subject std — compare user's 0.10 to the LOO bag-of-2
      within-subject std for ONLY the 68 dataset patients in the user's true-Hb
      bin (14.3–16.3), NOT the global 250-patient average. The original outcome
      C was confounded: user sits at the top of the Hb range, where a
      shrinkage-to-mean model (slope 0.47) compresses hardest, naturally
      producing lower within-subject variance for ANY subject at that
      position. We need bin-matched comparison to isolate user-specific
      collapse from edge-of-axis collapse.

  (B) Bootstrap CI on the responsiveness PASS — 1.370 vs threshold 1.335 is
      a 2.6% margin. Bootstrap the 222-patient first-LOO bag-of-2 predictions
      to get a 95% CI on the across-subject std. If 1.335 sits inside the CI,
      the PASS is marginal and the writeup should say so.

Verbal correction: even if C survives the bin-matched check, n=1 darker-skin
subject cannot earn the word "demographic" — call it "this subject" instead.
"Demographic" requires option 2 (multiple darker-skin users with CBCs).

Output: appended to user_data/_deaf_test_results.json under 'followup' key.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from pockethb.data import load_metadata, iter_crops
from pockethb.inference import InferenceSession

USER_HB = 15.3
PHOTO_DIR = Path("user_data")
BUNDLE = Path("weights/pockethb_base.pkl")
RESULTS_JSON = PHOTO_DIR / "_deaf_test_results.json"
SEED = 42
HIGH_HB_BIN = (14.3, 16.3)
N_BOOTSTRAP = 1000


def aggregate(embs: np.ndarray) -> np.ndarray:
    if embs.shape[0] == 1:
        agg = np.concatenate([embs[0], np.zeros_like(embs[0])])
    else:
        agg = np.concatenate([embs.mean(axis=0), embs.std(axis=0, ddof=0)])
    return agg.astype(np.float32).reshape(1, -1)


def main() -> int:
    rng = np.random.default_rng(SEED)
    sess = InferenceSession.from_pkl(BUNDLE)

    # ---- re-embed dataset (cached on disk via timm but recomputes through blender) ----
    print("[1/3] re-embedding dataset for follow-up checks...")
    df = load_metadata()
    crops_by_patient: dict[int, list[np.ndarray]] = {}
    for c in iter_crops(df, region="nail"):
        crops_by_patient.setdefault(c.patient_id, []).append(c.image)

    flat_crops = []
    pids_flat = []
    for pid, crops in crops_by_patient.items():
        for img in crops:
            flat_crops.append(img)
            pids_flat.append(pid)

    t0 = time.time()
    all_embs = sess.embed_many(flat_crops)
    print(f"  done in {time.time()-t0:.1f}s")

    emb_by_patient: dict[int, np.ndarray] = {}
    cursor = 0
    for pid, crops in crops_by_patient.items():
        emb_by_patient[pid] = all_embs[cursor : cursor + len(crops)]
        cursor += len(crops)

    hb_by_pid = dict(zip(df["PATIENT_ID"].astype(int), df["hb_g_per_dL"].astype(float)))
    pids_3crop = sorted(p for p, e in emb_by_patient.items() if e.shape[0] == 3)

    # ---- (A) bin-matched within-subject std ----
    print("\n[2/3] check A — edge-matched within-subject std")
    pids_in_bin = [p for p in pids_3crop if HIGH_HB_BIN[0] <= hb_by_pid[p] <= HIGH_HB_BIN[1]]
    print(f"  patients in [{HIGH_HB_BIN[0]}, {HIGH_HB_BIN[1]}] with 3 crops: {len(pids_in_bin)}")

    bin_within_stds = []
    for pid in pids_in_bin:
        e = emb_by_patient[pid]
        loo_preds = []
        for drop in range(3):
            keep = [i for i in range(3) if i != drop]
            agg = aggregate(e[keep])
            loo_preds.append(float(sess.blender.predict(agg)[0]))
        bin_within_stds.append(float(np.std(loo_preds, ddof=0)))
    bin_within_mean = float(np.mean(bin_within_stds))
    bin_within_median = float(np.median(bin_within_stds))
    print(f"  bin-matched dataset within-subject std (mean of {len(pids_in_bin)} patients): {bin_within_mean:.4f}")
    print(f"  bin-matched dataset within-subject std (median):                            {bin_within_median:.4f}")

    # also compute global for reference (recompute, don't trust state)
    global_within_stds = []
    for pid in pids_3crop:
        e = emb_by_patient[pid]
        loo = [float(sess.blender.predict(aggregate(e[[i for i in range(3) if i != d]]))[0]) for d in range(3)]
        global_within_stds.append(float(np.std(loo, ddof=0)))
    global_within_mean = float(np.mean(global_within_stds))
    print(f"  GLOBAL dataset within-subject std (mean of {len(pids_3crop)} patients):    {global_within_mean:.4f}  (sanity vs earlier 0.683)")

    # user within-subject from chunk 7 followup: re-load it for an apples comparison
    user_photos = sorted(p for p in PHOTO_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    user_embs = sess.embed_many([Image.open(p).convert("RGB") for p in user_photos])
    user_bag2 = []
    for _ in range(100):
        idx = rng.choice(user_embs.shape[0], size=2, replace=False)
        user_bag2.append(float(sess.blender.predict(aggregate(user_embs[idx]))[0]))
    user_within_std = float(np.std(user_bag2, ddof=0))
    print(f"\n  user within-subject std (n=100 bag-of-2 subsamples): {user_within_std:.4f}")

    user_to_bin_ratio = user_within_std / bin_within_mean if bin_within_mean > 0 else float("inf")
    user_to_global_ratio = user_within_std / global_within_mean if global_within_mean > 0 else float("inf")
    print(f"\n  ratio user/bin-matched dataset: {user_to_bin_ratio:.3f}")
    print(f"  ratio user/global dataset:      {user_to_global_ratio:.3f}  (the confounded number from chunk 7)")

    # decision under the bin-matched comparison (use the same pre-registered 0.5 threshold)
    c_under_bin_match = user_to_bin_ratio <= 0.5
    print(f"\n  outcome C under bin-matched rule (ratio ≤ 0.5): {'YES — C survives' if c_under_bin_match else 'NO — C does not survive bin-matched test'}")
    if not c_under_bin_match:
        print("  → fold into 'model-wide compression at the edges, illustrated by this single subject'")
        print("    (slope-0.47 finding becomes the headline; n=1 cannot earn the word 'demographic')")

    # ---- (B) bootstrap responsiveness ----
    print("\n[3/3] check B — bootstrap CI on responsiveness std")
    # base: first LOO bag-of-2 per 3-crop patient
    base_preds = []
    for pid in pids_3crop:
        e = emb_by_patient[pid]
        agg = aggregate(e[[1, 2]])  # omit crop 0, consistent with chunk-7 successor
        base_preds.append(float(sess.blender.predict(agg)[0]))
    base_preds = np.array(base_preds)
    base_std = float(base_preds.std(ddof=0))
    print(f"  point estimate (matches earlier): {base_std:.4f}")

    boot = np.empty(N_BOOTSTRAP)
    n = len(base_preds)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        boot[b] = float(base_preds[idx].std(ddof=0))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    threshold = 0.5 * 2.67
    print(f"  bootstrap 95% CI for across-subject std: [{lo:.4f}, {hi:.4f}]")
    print(f"  threshold for PASS:                       {threshold:.4f}")
    pass_robust = lo >= threshold
    pass_marginal = (lo < threshold <= hi)
    pass_fail = hi < threshold
    if pass_robust:
        verdict = "PASS robust (entire CI ≥ threshold)"
    elif pass_marginal:
        verdict = f"PASS marginal — threshold {threshold:.4f} sits INSIDE the CI [{lo:.4f}, {hi:.4f}]; report as 'marginal responsiveness'"
    elif pass_fail:
        verdict = f"PASS appears fragile — entire CI BELOW threshold; would reclassify as FAIL on resampling"
    else:
        verdict = "ambiguous"
    print(f"  → {verdict}")

    # ---- save followup to json ----
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else {}
    payload["followup"] = {
        "edge_matched_within_subject_std": {
            "high_hb_bin": list(HIGH_HB_BIN),
            "n_patients_in_bin_with_3_crops": int(len(pids_in_bin)),
            "bin_within_mean": bin_within_mean,
            "bin_within_median": bin_within_median,
            "global_within_mean_sanity": global_within_mean,
            "user_within_std_for_comparison": user_within_std,
            "ratio_user_to_bin_matched": user_to_bin_ratio,
            "ratio_user_to_global_old": user_to_global_ratio,
            "outcome_C_survives_bin_match": bool(c_under_bin_match),
        },
        "responsiveness_bootstrap": {
            "point_estimate": base_std,
            "ci_lower_2p5": lo,
            "ci_upper_97p5": hi,
            "threshold": threshold,
            "verdict": verdict,
            "n_bootstrap_samples": N_BOOTSTRAP,
        },
        "verbal_correction": (
            "Even if C survives the bin-matched test, n=1 darker-skin subject cannot "
            "earn the word 'demographic.' The honest phrasing is 'this subject.' "
            "'Demographic' would require option 2 (multiple darker-skin users with paired CBCs)."
        ),
        "interpretation_status": "PROVISIONAL — pending mentor sign-off on the bin-matched + bootstrap follow-up.",
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote followup section to {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
