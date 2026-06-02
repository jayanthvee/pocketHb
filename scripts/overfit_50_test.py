"""P2: overfit-50-patients positive control.

Train the chunk-3 pipeline on 50 patients used as BOTH train and test.
If the pipeline is implemented correctly it should overfit hard:
  train MAE near zero, R² near 1, slope near 1.

If it can't memorize 50 samples it's used to train on, the bug is upstream
(features misaligned, labels broken, regressor unable to fit, etc.).
If it can memorize them but fails on real held-out CV, the failure is
generalization (which is the method-bound finding we're claiming).

Output: stdout + appended to user_data/_implementation_checks.json
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
from pockethb.embed import DEFAULT_BACKBONE, aggregate_per_patient, embed_crops, load_backbone
from pockethb.inference import InferenceSession
from pockethb.regressor import fit_blender

CHECKS_JSON = Path("user_data/_implementation_checks.json")
SEED = 42


def main() -> int:
    rng = np.random.default_rng(SEED)
    sess = InferenceSession.from_pkl("weights/pockethb_base.pkl")

    # load metadata + pick 50 patients with 3 crops each
    df = load_metadata()
    crops_by_patient: dict[int, list] = {}
    for c in iter_crops(df, region="nail"):
        crops_by_patient.setdefault(c.patient_id, []).append(c.image)
    pids_3 = [p for p, cs in crops_by_patient.items() if len(cs) == 3]
    chosen = sorted(rng.choice(pids_3, size=50, replace=False).tolist())
    print(f"selected 50 patients with 3 crops each (from {len(pids_3)} available)")

    # embed
    print("embedding 150 crops via frozen ConvNeXt-Tiny...")
    flat = []
    pid_for = []
    for p in chosen:
        for img in crops_by_patient[p]:
            flat.append(img)
            pid_for.append(p)
    t0 = time.time()
    embs = sess.embed_many(flat)
    print(f"  done in {time.time()-t0:.1f}s, shape {embs.shape}")

    # aggregate per patient
    X, pid_order = aggregate_per_patient(embs, pid_for)
    hb = dict(zip(df["PATIENT_ID"].astype(int), df["hb_g_per_dL"].astype(float)))
    y = np.array([hb[p] for p in pid_order], dtype=np.float64)
    print(f"X shape: {X.shape}  y shape: {y.shape}")

    # fit on all 50, predict on all 50 (overfit test)
    print("\nfitting blender on all 50 patients, predicting on the same 50...")
    blender = fit_blender(X, y)
    pred = blender.predict(X)

    mae = mean_absolute_error(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = r2_score(y, pred)
    slope, intercept = np.polyfit(y, pred, 1)

    print(f"\nOVERFIT (in-sample) metrics on the 50 patients:")
    print(f"  MAE  = {mae:.4f} g/dL")
    print(f"  RMSE = {rmse:.4f} g/dL")
    print(f"  R²   = {r2:.4f}")
    print(f"  slope (predicted vs true linear regression) = {slope:.4f}")
    print(f"  intercept = {intercept:+.4f}")
    print(f"  blender: pls_n={blender.pls_n_components}  svr_C={blender.svr_C}  svr_gamma={blender.svr_gamma}  weight_pls={blender.weight_pls:.2f}")

    # verdict
    print("\n" + "=" * 60)
    if mae < 0.4 and r2 > 0.8 and slope > 0.8:
        verdict = "PASS — pipeline can fit 50 training patients tightly. Implementation can learn; chunk-3 generalization failure is method-bound."
    elif mae < 1.0 and r2 > 0.4:
        verdict = "PARTIAL PASS — pipeline shows real overfit signal but not as tight as expected. Investigate regularization / blender behavior."
    else:
        verdict = "FAIL — pipeline cannot fit 50 patients it trained on. There is an upstream bug in features/labels/aggregation/regressor that must be found before any publication."
    print(f"VERDICT: {verdict}")
    print("=" * 60)

    payload = json.loads(CHECKS_JSON.read_text(encoding="utf-8")) if CHECKS_JSON.exists() else {}
    payload["overfit_50_positive_control"] = {
        "n_patients": 50,
        "in_sample_MAE": float(mae),
        "in_sample_RMSE": rmse,
        "in_sample_R2": float(r2),
        "in_sample_slope": float(slope),
        "in_sample_intercept": float(intercept),
        "blender_params": {
            "pls_n_components": blender.pls_n_components,
            "svr_C": blender.svr_C,
            "svr_gamma": str(blender.svr_gamma),
            "weight_pls": float(blender.weight_pls),
        },
        "verdict": verdict,
    }
    CHECKS_JSON.parent.mkdir(exist_ok=True)
    CHECKS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {CHECKS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
