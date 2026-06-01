"""Per-user calibration layers on top of the global Hb regressor.

The global model produces biased per-user estimates (skin tone, nail thickness,
camera color profile compress or stretch the response curve). Given a user's
known bloodwork reading and a few photos at that anchor, we fit a small
correction that adapts the global prediction to that specific person.

v1 (this module): affine — fit (a, b) such that Hb_personal = a * f(x) + b.
v2 (chunks 5/7): per-user MLP head on frozen features (calibration.py +
                  model code, not in this file).

Math behind v1 (full least squares with multiple distinct anchor Hb values):

    minimise_{a,b}  Σ_j (a · p_j + b − y_j)²

    closed form: stack predictions p = [p_1, ..., p_k]ᵀ, targets y = [y_1, ..., y_k]ᵀ,
    design matrix X = [p, 1]. then [a, b]ᵀ = (XᵀX)⁻¹ Xᵀ y.

Degenerate case (all targets identical — the user has only ONE bloodwork value
and k photos at that same Hb): the system is underdetermined for slope, so we
fall back to bias-only correction:

    a = 1
    b = y* − mean(p)

bias-only is what almost every real user gets, because most people only have
one CBC reading on file. it removes the systematic offset for that person
without touching the slope (which can't be learned from one anchor anyway).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


@dataclass
class AffineCalibrator:
    """Single-user affine calibration: Hb_cal = a · Hb_pred + b."""

    a: float = 1.0
    b: float = 0.0
    fitted: bool = False
    mode: str = "identity"  # "affine" | "bias_only" | "identity"
    n_anchors_used: int = 0

    def fit(self, predictions, targets) -> "AffineCalibrator":
        """Fit (a, b) on a user's paired (base-model prediction, true Hb) samples.

        If targets vary, full least-squares affine fit.
        If targets are all the same value, falls back to bias-only correction.
        """
        p = np.asarray(predictions, dtype=np.float64).ravel()
        y = np.asarray(targets, dtype=np.float64).ravel()
        if p.shape != y.shape:
            raise ValueError(f"shape mismatch: predictions {p.shape} vs targets {y.shape}")
        if len(p) < 1:
            raise ValueError("need at least one anchor sample")

        if float(np.std(y)) < _EPS:
            # all anchors at the same Hb value → bias-only
            self.a = 1.0
            self.b = float(y.mean() - p.mean())
            self.mode = "bias_only"
        else:
            X = np.column_stack([p, np.ones_like(p)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.a = float(coef[0])
            self.b = float(coef[1])
            self.mode = "affine"

        self.fitted = True
        self.n_anchors_used = int(len(p))
        return self

    def predict(self, predictions) -> np.ndarray:
        p = np.asarray(predictions, dtype=np.float64)
        return self.a * p + self.b

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "fitted": self.fitted,
            "mode": self.mode,
            "n_anchors_used": self.n_anchors_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AffineCalibrator":
        c = cls()
        c.a = float(d["a"])
        c.b = float(d["b"])
        c.fitted = bool(d.get("fitted", False))
        c.mode = str(d.get("mode", "identity"))
        c.n_anchors_used = int(d.get("n_anchors_used", 0))
        return c
