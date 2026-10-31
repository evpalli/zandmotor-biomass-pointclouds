"""STEP 1. Merge the .laz clips into ONE file per epoch, cropped to the study area (BOX in helpers.py).

Put all your .laz clips anywhere under data/clips/ (sub-folders are fine). The epoch is read from the 8-digit date in the
file name (e.g. 20250327). The files are read chunk by chunk, so little memory is needed. If an epoch has more than
400 points/m2 it is thinned to that (the analysis thins to 250 later anyway); the original point count and density are
saved in data/merged/epoch_stats.csv for table A1.   Run:  python prepare_data.py
"""
import re
from collections import defaultdict

import laspy
import numpy as np
import pandas as pd

import helpers as H

KEEP = ["intensity", "return_number", "number_of_returns", "classification", "red", "green", "blue"]
MAX_DENSITY = 400.0

groups = defaultdict(list)
for f in sorted(H.CLIPS_DIR.rglob("*")):
    if f.suffix.lower() in (".laz", ".las"):
        m = re.search(r"(20\d{6})", f.name)
        if m and m.group(1) in H.EPOCH_LABEL:
            groups[m.group(1)].append(f)
if not groups:
    raise SystemExit(f"No .laz files with a date in the name found under {H.CLIPS_DIR}")

x0, y0, x1, y1 = H.BOX
area = (x1 - x0) * (y1 - y0)
rng = np.random.default_rng(0)
stats = []
for epoch, files in sorted(groups.items()):
    n_header = sum(laspy.open(f).header.point_count for f in files)            # upper bound of the points in the area
    keep_frac = min(1.0, MAX_DENSITY * area / n_header)
    X, Y, Z, cols, n_raw = [], [], [], {k: [] for k in KEEP}, 0
    for f in files:
        with laspy.open(f) as r:
            names = r.header.point_format.dimension_names
            for pts in r.chunk_iterator(2_000_000):
                x, y = np.asarray(pts.x), np.asarray(pts.y)
                m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                n_raw += int(m.sum())
                if keep_frac < 1.0:
                    m &= rng.random(len(m)) < keep_frac
                if not m.any():
                    continue
                X.append(x[m]); Y.append(y[m]); Z.append(np.asarray(pts.z)[m])
                for k in KEEP:
                    cols[k].append(np.asarray(pts[k])[m] if k in names else np.zeros(m.sum(), np.uint16))
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales, hdr.offsets = np.array([0.001] * 3), np.array([72000.0, 451000.0, 0.0])
    out = laspy.LasData(hdr)
    out.x, out.y, out.z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)
    for k in KEEP:
        out[k] = np.concatenate(cols[k]).astype(out[k].dtype)
    out.write(H.MERGED_DIR / f"{epoch}_field.laz")
    stats.append({"epoch": epoch, "files": len(files), "points_in_area": n_raw, "kept_points": len(out.x)})
    print(f"{epoch}: {len(files)} files, {n_raw:,} points in the area -> kept {len(out.x):,}")
    del X, Y, Z, cols, out

pd.DataFrame(stats).to_csv(H.MERGED_DIR / "epoch_stats.csv", index=False)
