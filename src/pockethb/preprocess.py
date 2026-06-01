"""Illumination correction for nail/skin crops.

Shades-of-Gray (Finlayson & Trezzi 2004) — Minkowski p-norm generalisation
of the gray-world assumption. Standard in photometric calibration pipelines
and what the Tilburg/Sanquin BNAIC paper used (p=6).
"""
from __future__ import annotations

import numpy as np


def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Apply Shades-of-Gray colour constancy with Minkowski exponent p.

    Args:
        img: HxWx3 uint8 OR float in [0,1].
        p: Minkowski exponent. p=1 is gray-world, p=inf is max-RGB.
            BNAIC used p=6 — a robust middle ground.

    Returns:
        Corrected HxWx3 uint8.
    """
    was_uint8 = img.dtype == np.uint8
    x = img.astype(np.float64)
    if was_uint8:
        x = x / 255.0

    # per-channel Lp norm of pixel intensities
    Lp = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)  # shape (3,)

    # normalise so the corrected image preserves overall brightness
    # scale by sqrt(3) / ||Lp||_2 so the illuminant is treated as a unit vector toward gray
    norm = np.sqrt((Lp ** 2).sum()) + 1e-12
    illuminant = Lp / norm * np.sqrt(3)

    corrected = x / illuminant
    corrected = np.clip(corrected, 0.0, 1.0)
    return (corrected * 255.0).astype(np.uint8)
