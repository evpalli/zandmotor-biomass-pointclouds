"""
MODULE: spectral_typing.py
PURPOSE: Spectral greenness extraction and pioneer species discrimination.
ASSIGNMENT QUESTION: A2 (Vegetation Identification & Species Typing)
https://www.scribd.com/document/364828887/Visible-Atmospherically-Resistant-Index-VARI
https://www.sciencedirect.com/science/article/abs/pii/S0034425701002899
"""

import numpy as np


def compute_vari(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes Visible Atmospherically Resistant Index (VARI):
      VARI = (Green - Red) / (Green + Red - Blue)
    Normalizes RGB channels to prevent integer overflow.
    """
    rf = r.astype(np.float32)
    gf = g.astype(np.float32)
    bf = b.astype(np.float32)

    denom = gf + rf - bf
    denom[denom == 0] = 1e-6
    return (gf - rf) / denom


def classify_pioneer_species(height: float, linearity: float, vari: float) -> str:
    """
    Applies multi-attribute decision criteria to distinguish coastal pioneer species:
      1. Ammophila arenaria (European marram grass): Tall, erect, high linearity stems.
      2. Elytrigia juncea (Sand couch): Low-profile, lateral creeping embryonic runners.
      3. Dry driftline / Senescent matter: Low photosynthetic greenness (low VARI).
    """
    if vari < 0.0:
        return "Dry Drift / Senescent Matter"
    elif height >= 0.25 and linearity >= 0.25:
        return "Ammophila arenaria (Marram Grass)"
    else:
        return "Elytrigia juncea (Sand Couch)"