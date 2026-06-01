"""Per-patient Hb regressor following Rudokaite et al. BNAIC 2025.

Pipeline:
    standardise → PLS  ──┐
                          ├─ isotonic-calibrate each → weighted blend
    standardise → SVR  ──┘                                 │
                                                           ▼
                                                       Hb estimate

Hyperparameters tuned by inner CV inside fit(). All sklearn — fast on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class FittedBlender:
    scaler: StandardScaler
    pls: PLSRegression
    svr: SVR
    iso_pls: IsotonicRegression
    iso_svr: IsotonicRegression
    weight_pls: float  # weight in [0,1]; svr weight is 1-w
    pls_n_components: int
    svr_C: float
    svr_gamma: float | str

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_std = self.scaler.transform(X)
        p_pls = self.pls.predict(X_std).ravel()
        p_svr = self.svr.predict(X_std)
        c_pls = self.iso_pls.predict(p_pls)
        c_svr = self.iso_svr.predict(p_svr)
        return self.weight_pls * c_pls + (1.0 - self.weight_pls) * c_svr


def _tune_pls(X: np.ndarray, y: np.ndarray, max_components: int = 20) -> tuple[PLSRegression, int]:
    """Inner 5-fold CV to pick PLS n_components."""
    max_components = min(max_components, X.shape[1], len(y) - 1)
    best_n, best_mae = 1, float("inf")
    inner = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=0)
    for n in range(1, max_components + 1):
        maes = []
        for tr, te in inner.split(X):
            m = PLSRegression(n_components=n, scale=False)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te]).ravel()
            maes.append(mean_absolute_error(y[te], pred))
        mae = float(np.mean(maes))
        if mae < best_mae:
            best_mae, best_n = mae, n
    final = PLSRegression(n_components=best_n, scale=False)
    final.fit(X, y)
    return final, best_n


def _tune_svr(X: np.ndarray, y: np.ndarray) -> tuple[SVR, float, float | str]:
    """Inner 3-fold CV over a small grid for SVR(RBF). Returns fitted SVR."""
    Cs = [0.5, 1.0, 2.0, 4.0]
    gammas = ["scale", 1e-3, 3e-3, 1e-2]
    inner = KFold(n_splits=3, shuffle=True, random_state=0)
    best = (None, float("inf"), 1.0, "scale")
    for C in Cs:
        for g in gammas:
            maes = []
            for tr, te in inner.split(X):
                m = SVR(kernel="rbf", C=C, gamma=g)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                maes.append(mean_absolute_error(y[te], pred))
            mae = float(np.mean(maes))
            if mae < best[1]:
                best = (None, mae, C, g)
    final = SVR(kernel="rbf", C=best[2], gamma=best[3])
    final.fit(X, y)
    return final, best[2], best[3]


def _pick_blend_weight(pls_pred: np.ndarray, svr_pred: np.ndarray, y: np.ndarray) -> float:
    """Pick w in [0,1] that minimises blended MAE on the training fold (after iso-calibration)."""
    weights = np.linspace(0.0, 1.0, 21)
    best_w, best_mae = 0.5, float("inf")
    for w in weights:
        blend = w * pls_pred + (1.0 - w) * svr_pred
        mae = mean_absolute_error(y, blend)
        if mae < best_mae:
            best_mae, best_w = mae, float(w)
    return best_w


def fit_blender(X: np.ndarray, y: np.ndarray) -> FittedBlender:
    """Fit the full pipeline on a training fold and return a deployable blender."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pls, n_comp = _tune_pls(X_std, y)
    svr, C, gamma = _tune_svr(X_std, y)

    # within-train predictions for isotonic calibration
    pls_raw = pls.predict(X_std).ravel()
    svr_raw = svr.predict(X_std)

    iso_pls = IsotonicRegression(out_of_bounds="clip").fit(pls_raw, y)
    iso_svr = IsotonicRegression(out_of_bounds="clip").fit(svr_raw, y)

    c_pls = iso_pls.predict(pls_raw)
    c_svr = iso_svr.predict(svr_raw)
    w = _pick_blend_weight(c_pls, c_svr, y)

    return FittedBlender(
        scaler=scaler,
        pls=pls,
        svr=svr,
        iso_pls=iso_pls,
        iso_svr=iso_svr,
        weight_pls=w,
        pls_n_components=n_comp,
        svr_C=C,
        svr_gamma=gamma,
    )


@dataclass
class CVResult:
    """Out-of-fold predictions concatenated across all folds, plus per-fold metrics."""
    oof_pred: np.ndarray
    oof_true: np.ndarray
    oof_pids: np.ndarray
    fold_metrics: list[dict] = field(default_factory=list)
    blender_per_fold: list[FittedBlender] = field(default_factory=list)


def stratified_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    pids: list[int],
    n_splits: int = 5,
    n_bins: int = 5,
    seed: int = 42,
) -> CVResult:
    """5-fold CV with Hb-stratified bins. Mirrors Rudokaite §2.6.

    Each PATIENT goes to exactly one fold (patient-disjoint by construction here —
    X is already at patient level).
    """
    from sklearn.model_selection import StratifiedKFold

    bins = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_bins + 1))[1:-1])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_pred = np.zeros_like(y, dtype=np.float64)
    oof_true = y.copy().astype(np.float64)
    oof_pids = np.asarray(pids)

    fold_metrics = []
    fitted = []
    for fold_idx, (tr, te) in enumerate(skf.split(X, bins), start=1):
        blender = fit_blender(X[tr], y[tr])
        pred = blender.predict(X[te])
        oof_pred[te] = pred

        mae = mean_absolute_error(y[te], pred)
        rmse = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y[te], pred))
        fold_metrics.append({
            "fold": fold_idx,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "MAE": float(mae),
            "RMSE": rmse,
            "R2": r2,
            "pls_n_components": blender.pls_n_components,
            "svr_C": blender.svr_C,
            "svr_gamma": str(blender.svr_gamma),
            "weight_pls": float(blender.weight_pls),
        })
        fitted.append(blender)

    return CVResult(
        oof_pred=oof_pred,
        oof_true=oof_true,
        oof_pids=oof_pids,
        fold_metrics=fold_metrics,
        blender_per_fold=fitted,
    )
