"""Compares the method of the report with other CSF settings / ground models on the same data (A4: 'is this the best setting?').
Needs data/merged (run prepare_data.py first). Takes ~10 min.   python tools/compare_csf_variants.py   -> tables/A4_csf_variants.csv"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import helpers as H

q25 = lambda v: np.percentile(v, 25)
VARIANTS = {
    "report method (CSF res .5, threshold .10, rigidness 2)": {},
    "first CSF code (res .6, threshold .12, no height / shape gates)": dict(csf=dict(resolution=.6, rigidness=2, threshold=.12), gate=(-9, 99), sparse_veg=False, drop_outliers=False),
    "report rules + CSF res .6, threshold .12": dict(csf=dict(resolution=.6, rigidness=2, threshold=.12)),
    "CSF rigidness 1": dict(csf=dict(resolution=.5, rigidness=1, threshold=.10)),
    "CSF rigidness 3": dict(csf=dict(resolution=.5, rigidness=3, threshold=.10)),
    "CSF threshold .05": dict(csf=dict(resolution=.5, rigidness=2, threshold=.05)),
    "CSF threshold .15": dict(csf=dict(resolution=.5, rigidness=2, threshold=.15)),
    "CSF cloth resolution .3": dict(csf=dict(resolution=.3, rigidness=2, threshold=.10)),
    "CSF cloth resolution .75": dict(csf=dict(resolution=.75, rigidness=2, threshold=.10)),
    "CSF cloth resolution 1.0": dict(csf=dict(resolution=1.0, rigidness=2, threshold=.10)),
    "CSF cloth resolution 1.0, threshold .05": dict(csf=dict(resolution=1.0, rigidness=2, threshold=.05)),
    "DTM = 25th percentile of ground per cell": dict(dtm_how=q25),
}
W1, W2 = (72450, 451485, 72510, 451560), (72300, 451540, 72400, 451620)       # window around plant 1; window with 4 biomass samples
S = pd.read_csv(H.ROOT / "data" / "biomass_measurements.csv"); S.columns = ["x", "y", "g"]
P1, P2 = H.read_points("20250327", box=W1), H.read_points("20250327", box=W2)
rows = []
for name, opt in VARIANTS.items():
    R1 = H.separate(P1, box=W1, **opt); R1["epoch"] = "x"; t1 = H.tussocks(R1, alpha=False)
    R2 = H.separate(P2, box=W2, **opt); R2["epoch"] = "x"; v2 = R2["veg"]
    near = [int((np.hypot(v2["x"] - s.x, v2["y"] - s.y) < 1).sum()) for _, s in S[(S.x > W2[0]) & (S.x < W2[2]) & (S.y > W2[1]) & (S.y < W2[3])].iterrows()]
    g = H.gnss_dtm_check(**opt)[2].set_index("track")["RMSE (m)"] * 100
    rows.append({"variant": name, "GNSS RMSE all (cm)": g["all tracks"], "GNSS RMSE flat Track 3 (cm)": g["Track 3"], "GNSS RMSE steep Track 1 (cm)": g["Track 1"],
                 "W1 vegetation points": R1["n_veg"], "W1 points < 10 cm above ground (%)": 100 * float(np.mean(R1["veg"]["hag"] < 0.10)),
                 "W1 cover (m2)": H.cover_m2(R1), "W1 tussock volume (m3)": t1.vol_column.sum(), "W1 largest plant volume (m3)": t1.vol_column.max(),
                 "W2 veg points within 1 m of the 4 samples": str(near)})
    print(rows[-1]["variant"], flush=True)
H.table(pd.DataFrame(rows).round(1), "A4_csf_variants")
