"""Independent reproduction — uses ONLY src/ code + data/ + weights/pockethb_base.pkl.
Does not read docs/notebooks/README/metrics.json or analysis scripts."""
import pickle
import numpy as np

from pockethb.data import load_metadata, iter_crops
from pockethb.embed import load_backbone, embed_crops, aggregate_per_patient
from pockethb.regressor import stratified_kfold_cv
from sklearn.metrics import mean_absolute_error, r2_score

CACHE = "_repro_emb_cache.npz"

# ---- 1. build per-patient feature matrix from nail crops (bag-of-3, mean+std) ----
df = load_metadata()
import os
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    X, pids = z["X"], list(z["pids"])
    print("loaded cached embeddings", X.shape)
else:
    crops = list(iter_crops(df, region="nail"))
    print("total nail crops:", len(crops))
    backbone = load_backbone("convnext_tiny.fb_in22k_ft_in1k", device="cpu")
    emb, crop_pids, _ = embed_crops(backbone, crops, batch_size=16, apply_sog=True,
                                    device="cpu", progress=True)
    X, pids = aggregate_per_patient(emb, crop_pids)
    np.savez(CACHE, X=X, pids=np.array(pids))
    print("patient feature matrix:", X.shape)

# ---- align labels to patient order ----
hb_by_pid = dict(zip(df["PATIENT_ID"].astype(int), df["hb_g_per_dL"].astype(float)))
y = np.array([hb_by_pid[int(p)] for p in pids], dtype=np.float64)
assert len(y) == 250, len(y)

# ============================================================
# RESULT 1: OOF 5-fold CV MAE & R2 (repo's own stratified_kfold_cv, seed 42)
# ============================================================
cv = stratified_kfold_cv(X, y, pids, n_splits=5, n_bins=5, seed=42)
oof_mae = mean_absolute_error(cv.oof_true, cv.oof_pred)
oof_r2 = r2_score(cv.oof_true, cv.oof_pred)
print("\n=== RESULT 1: out-of-fold 5-fold CV (n=250) ===")
print(f"  OOF MAE = {oof_mae:.4f} g/dL")
print(f"  OOF R2  = {oof_r2:.4f}")

# ============================================================
# RESULT 2/3: global model (shipped blender) per-patient bag-of-3 predictions,
#             then linear regression pred(y) vs true(x)
# ============================================================
bundle = pickle.load(open("weights/pockethb_base.pkl", "rb"))
blender = bundle["blender"]
preds = blender.predict(X)            # X already mean+std bag-of-3 per patient
print("\n=== RESULT 2: per-patient global predictions (bag-of-3) ===")
print(f"  n={len(preds)}  pred mean={preds.mean():.3f}  pred std={preds.std(ddof=0):.4f}"
      f"  min={preds.min():.3f} max={preds.max():.3f}")

# linear fit: predicted (y-axis) vs true (x-axis)
slope, intercept = np.polyfit(y, preds, 1)
yhat = slope * y + intercept
ss_res = np.sum((preds - yhat) ** 2)
ss_tot = np.sum((preds - preds.mean()) ** 2)
fit_r2 = 1 - ss_res / ss_tot
print("\n=== RESULT 3: linear regression  pred_Hb (y) ~ true_Hb (x) ===")
print(f"  slope     = {slope:.4f}")
print(f"  intercept = {intercept:.4f}")
print(f"  R2        = {fit_r2:.4f}")
print(f"  (sanity) Pearson r = {np.corrcoef(y, preds)[0,1]:.4f}")

# ============================================================
# RESULT 4: std of true Hb labels (250 patients)
# ============================================================
print("\n=== RESULT 4: std of true Hb labels (n=250) ===")
print(f"  population std (ddof=0) = {y.std(ddof=0):.4f} g/dL")
print(f"  sample std     (ddof=1) = {y.std(ddof=1):.4f} g/dL")
