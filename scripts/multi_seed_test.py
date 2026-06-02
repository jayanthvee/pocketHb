"""P3: multi-seed confirmation of the chunk-3 result.

Re-runs chunk 3 with seeds [0, 1, 42, 100, 999] and reports the distribution
of OOF MAE, R², and predicted-vs-true regression slope. Shows whether the
collapse is consistent across seeds or whether seed 42 was an unlucky pick.

Note: backbone embeddings are deterministic across seeds (no augmentation here).
The seed only affects the StratifiedKFold split and the SVR / PLS inner-CV
random states. So variance across seeds reflects sensitivity of the regressor
to specific fold assignments.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pockethb.data import load_metadata, iter_crops
from pockethb.embed import DEFAULT_BACKBONE, aggregate_per_patient
from pockethb.inference import InferenceSession
from pockethb.regressor import stratified_kfold_cv

CHECKS_JSON = Path("user_data/_implementation_checks.json")
SEEDS = [0, 1, 42, 100, 999]


def main() -> int:
    sess = InferenceSession.from_pkl("weights/pockethb_base.pkl")

    print("loading data + embedding all 250 patients once...")
    df = load_metadata()
    crops_by_patient: dict[int, list] = {}
    for c in iter_crops(df, region="nail"):
        crops_by_patient.setdefault(c.patient_id, []).append(c.image)

    flat, pid_for = [], []
    for p, cs in crops_by_patient.items():
        for img in cs:
            flat.append(img)
            pid_for.append(p)

    t0 = time.time()
    all_embs = sess.embed_many(flat)
    print(f"embedded {all_embs.shape[0]} crops in {time.time()-t0:.1f}s")

    X, pid_order = aggregate_per_patient(all_embs, pid_for)
    hb = dict(zip(df["PATIENT_ID"].astype(int), df["hb_g_per_dL"].astype(float)))
    y = np.array([hb[p] for p in pid_order], dtype=np.float64)
    print(f"X: {X.shape}  y: {y.shape}")

    results_per_seed = []
    for s in SEEDS:
        print(f"\n--- seed {s} ---")
        result = stratified_kfold_cv(X, y, pids=pid_order, n_splits=5, n_bins=5, seed=s)
        mae = mean_absolute_error(result.oof_true, result.oof_pred)
        rmse = float(np.sqrt(mean_squared_error(result.oof_true, result.oof_pred)))
        r2 = r2_score(result.oof_true, result.oof_pred)
        slope, intercept = np.polyfit(result.oof_true, result.oof_pred, 1)
        print(f"  OOF MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  slope={slope:.4f}")
        results_per_seed.append({
            "seed": s,
            "MAE": float(mae),
            "RMSE": rmse,
            "R2": float(r2),
            "slope": float(slope),
            "intercept": float(intercept),
        })

    maes = np.array([r["MAE"] for r in results_per_seed])
    r2s = np.array([r["R2"] for r in results_per_seed])
    slopes = np.array([r["slope"] for r in results_per_seed])

    print("\n" + "=" * 60)
    print(f"summary over {len(SEEDS)} seeds:")
    print(f"  MAE  : mean={maes.mean():.4f}  std={maes.std():.4f}  range=[{maes.min():.4f}, {maes.max():.4f}]")
    print(f"  R²   : mean={r2s.mean():.4f}  std={r2s.std():.4f}  range=[{r2s.min():.4f}, {r2s.max():.4f}]")
    print(f"  slope: mean={slopes.mean():.4f}  std={slopes.std():.4f}  range=[{slopes.min():.4f}, {slopes.max():.4f}]")

    consistent = (r2s.max() < 0.20) and (slopes.max() < 0.7)
    if consistent:
        verdict = "CONSISTENT — collapse to mean-regression reproduces across all 5 seeds. Result is not a single-seed artifact."
    else:
        verdict = f"INCONSISTENT — at least one seed produced R² up to {r2s.max():.3f} or slope up to {slopes.max():.3f}. Investigate."
    print(f"\nVERDICT: {verdict}")
    print("=" * 60)

    payload = json.loads(CHECKS_JSON.read_text(encoding="utf-8")) if CHECKS_JSON.exists() else {}
    payload["multi_seed_robustness"] = {
        "seeds": SEEDS,
        "per_seed": results_per_seed,
        "summary": {
            "MAE_mean": float(maes.mean()),
            "MAE_std": float(maes.std()),
            "R2_mean": float(r2s.mean()),
            "R2_std": float(r2s.std()),
            "slope_mean": float(slopes.mean()),
            "slope_std": float(slopes.std()),
        },
        "verdict": verdict,
    }
    CHECKS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {CHECKS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
