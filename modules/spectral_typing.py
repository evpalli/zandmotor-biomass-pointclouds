"""
MODULE: spectral_typing.py
PURPOSE: Spectral greenness extraction and pioneer species discrimination.
ASSIGNMENT QUESTION: A2
https://www.scribd.com/document/364828887/Visible-Atmospherically-Resistant-Index-VARI
https://www.sciencedirect.com/science/article/abs/pii/S0034425701002899
"""

import numpy as np


def compute_vari(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Visible Atmospherically Resistant Index (VARI) = (G - R) / (G + R - B)."""
    rf = r.astype(np.float32)
    gf = g.astype(np.float32)
    bf = b.astype(np.float32)

    denom = gf + rf - bf
    denom[np.abs(denom) < 1e-5] = 1e-5
    vari = (gf - rf) / denom
    return np.clip(vari, -1.0, 1.0)


def classify_pioneer_morphology(
    height_max: float, linearity_mean: float, vari_mean: float
) -> str:
    """Morphological classification rule based on physical stature and greenness."""
    if vari_mean < -0.05:
        return "Dry Driftline / Senescent Matter"
    elif height_max >= 0.28 and linearity_mean >= 0.28:
        return "Ammophila arenaria (European Marram)"
    else:
        return "Elytrigia juncea (Sand Couch)"