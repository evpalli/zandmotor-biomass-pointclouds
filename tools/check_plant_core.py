"""Are the points that CSF calls 'ground' in the middle of a plant really sand? Compares the LiDAR intensity and colour of those points with
bare sand. Needs data/merged (prepare_data.py). python tools/check_plant_core.py -> tables/A4_plant_core_check.csv"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, laspy
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import helpers as H

las = laspy.read(H.MERGED_DIR / "20250327_field.laz"); x, y = np.asarray(las.x), np.asarray(las.y)
SPOTS = {"Plant 1": (72483.28, 451532.65), "Plant 3": (72458.19, 451466.98), "bare sand plain": (72415.0, 451560.0)}     # x, y; the sand plain is flat open sand
hw, rows = 10, []
for name, (cx, cy) in SPOTS.items():
    m = (abs(x - cx) < hw) & (abs(y - cy) < hw); idx = np.nonzero(m)[0]
    idx = idx[np.random.default_rng(0).random(len(idx)) < min(1, DENS := 250 * (2 * hw) ** 2 / len(idx))]
    d = {"x": x[idx], "y": y[idx], "z": np.asarray(las.z)[idx], "inten": np.asarray(las.intensity)[idx], "nr": np.asarray(las.number_of_returns)[idx],
         "rgb": np.column_stack([las.red[idx], las.green[idx], las.blue[idx]]).astype(np.float32) / 65535.0}
    keep = ~H.low_outliers(np.column_stack([d["x"], d["y"], d["z"]])); d = {k: v[keep] for k, v in d.items()}
    bx = (cx - hw, cy - hw, cx + hw, cy + hw)
    R = H.separate({k: d[k] for k in ("x", "y", "z", "rgb")}, box=bx, debug=True, drop_outliers=False); D = R["debug"]
    cl = np.zeros(len(d["x"]), int); cl[D["cand"]] = 1; cl[D["veg"]] = 2
    near = np.hypot(d["x"] - cx, d["y"] - cy) < 1.2
    base = np.percentile(d["z"][~near & (cl == 0)], 50)
    groups = {"bare sand plain (all points)": np.ones(len(cl), bool)} if name == "bare sand plain" else {
        f"{name}: vegetation (final)": near & (cl == 2),
        f"{name}: CSF 'ground' in the plant core (> 15 cm above the surrounding ground)": near & (cl == 0) & (d["z"] > base + 0.15),
        f"{name}: ground 3+ m from the plant": (np.hypot(d["x"] - cx, d["y"] - cy) > 3) & (cl == 0)}
    for g, mk in groups.items():
        rows.append({"group": g, "points": int(mk.sum()), "median intensity": int(np.median(d["inten"][mk])), "median brightness": round(float(np.median(d["rgb"][mk].mean(1))), 2),
                     "multi-return %": round(100 * float(np.mean(d["nr"][mk] > 1)), 1)})
H.table(pd.DataFrame(rows), "A4_plant_core_check")
