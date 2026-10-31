"""Writes the three notebooks (A, B, C). Not needed for the analysis itself.  python tools/make_notebooks.py [--run]"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"; NB.mkdir(exist_ok=True)


def make(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)


START = '''import sys; sys.path.insert(0, "..")            # so the notebook finds helpers.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import helpers as H

EP = H.epochs()                                  # all epochs found in data/merged
R = {e: H.process(e) for e in EP}                # ground / vegetation separation: slow the first time, cached afterwards
print([H.EPOCH_LABEL[e] for e in EP])'''

# ============================================================================ A
A = [
("md", "# A - Explore vegetation occurrences (A1-A4)\nRun top to bottom. Every cell is one step; the helper functions are in `helpers.py`."),
("code", START + '''
E = "20250327" if "20250327" in EP else EP[-1]       # epoch used for the examples (the date of the manual biomass samples)'''),
("md", "## A1 - Point cloud characteristics (and the effect of point density)"),
("code", '''a1 = H.describe_files(EP); H.table(a1, "A1_characteristics")'''),
("md", "Detection of low vegetation depends on density. Test: the same 60 x 60 m window of the densest epoch, thinned to different densities (the merged files keep at most 400 pts/m2)."),
("code", '''e_dense = EP[int(a1["density (pts/m2)"].idxmax())]
b = H.blocks(R[e_dense]); best = b.loc[b.cover_m2.idxmax()]                      # window with most vegetation
win = (best.x - 30, best.y - 30, best.x + 30, best.y + 30); rows = []
for dens in [30, 60, 125, 250, 400]:           # 400 = the maximum kept by prepare_data.py
    res = H.separate(H.read_points(e_dense, density=dens, box=win))
    rows.append({"density (pts/m2)": dens, "vegetation points": res["n_veg"], "vegetation cover (m2)": round(H.cover_m2(res), 1)})
H.table(pd.DataFrame(rows), "A1_density_experiment")'''),
("md", "**To write (A1):** compare the epochs (points, density, extent, attributes) and discuss the experiment above: at low density small and low plants are missed, so cover is a lower bound and epochs must be compared at the same density (that is why every epoch is thinned to 250 pts/m2)."),
("md", "## A2 - Three vegetation examples and bare sand\nTop: canopy height (m). Bottom: the points with their real RGB colour."),
("code", '''t = H.tussocks(R[E]); top = t.sort_values("vol_column", ascending=False).head(3).reset_index(drop=True)
bl = H.blocks(R[E], t); sand = bl[(bl.cover_m2 == 0) & (bl.slope_deg < 5) & (bl.data_share > .95)].sort_values("slope_deg").iloc[0]
spots = [(r.x, r.y, f"Plant {i+1}") for i, r in top.iterrows()] + [(sand.x, sand.y, "Bare sand")]
gf = R[E]["grid_fine"]; v = R[E]["veg"]; h = 2.5
sand_pts = H.read_points(E, box=(sand.x - 5, sand.y - 5, sand.x + 5, sand.y + 5))
fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
for j, (x, y, name) in enumerate(spots):
    c0, c1 = int((x - h - gf["x0"]) / gf["res"]), int((x + h - gf["x0"]) / gf["res"]); r0, r1 = int((gf["y1"] - y - h) / gf["res"]), int((gf["y1"] - y + h) / gf["res"])
    im = axs[0, j].imshow(R[E]["chm"][r0:r1, c0:c1], extent=(x-h, x+h, y-h, y+h), vmin=0, vmax=1, cmap="YlGn"); axs[0, j].set_title(name)
    if name == "Bare sand": p = sand_pts; xx, yy, cc = p["x"], p["y"], p["rgb"]
    else: m = (abs(v["x"] - x) < h) & (abs(v["y"] - y) < h); xx, yy, cc = v["x"][m], v["y"][m], v["rgb"][m]
    axs[1, j].scatter(xx, yy, c=np.clip(cc, 0, 1), s=3); axs[1, j].set_xlim(x-h, x+h); axs[1, j].set_ylim(y-h, y+h); axs[1, j].set_aspect("equal")
fig.colorbar(im, ax=axs[0, :], label="canopy height (m)", shrink=.8); H.save(fig, "A2_patches")'''),
("code", '''sb, sx = sand_pts["rgb"].mean(1), 2 * sand_pts["rgb"][:, 1] - sand_pts["rgb"][:, 0] - sand_pts["rgb"][:, 2]
H.table(pd.concat([
    top[["x", "y", "n_pts", "footprint_m2", "h_max", "h_p95", "vol_column", "linearity", "planarity", "sphericity", "brightness", "exg"]].round(3).assign(patch=["Plant 1", "Plant 2", "Plant 3"]),
    pd.DataFrame([{"patch": "Bare sand (points)", "brightness": sb.mean(), "exg": sx.mean()}]).round(3)]), "A2_patch_attributes")'''),
("md", "**To write (A2):** for each plant the most likely type (from height, footprint, shape: tall dense dome vs. low sparse tuft; there are no species labels in the field data, say so). Which attributes helped: height above ground and shape (linearity / planarity) separate plants from sand best; compare brightness / ExG of the plants with the sand in the table (dry vegetation is dark, so greenness alone is weak)."),
("md", "## A3 - One location through time\nA 12 x 12 m window around the largest plant, in every epoch (canopy height, m)."),
("code", '''x0, y0 = top.x[0], top.y[0]; h = 6; n = len(EP); cols = 4; fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(12, 3.2 * int(np.ceil(n / cols)))); axs = axs.ravel(); rows = []
for k, e in enumerate(EP):
    g = R[e]["grid_fine"]; c0, c1 = int((x0 - h - g["x0"]) / g["res"]), int((x0 + h - g["x0"]) / g["res"]); r0, r1 = int((g["y1"] - y0 - h) / g["res"]), int((g["y1"] - y0 + h) / g["res"])
    chm = R[e]["chm"][r0:r1, c0:c1]; im = axs[k].imshow(chm, extent=(x0-h, x0+h, y0-h, y0+h), vmin=0, vmax=1, cmap="YlGn"); axs[k].set_title(H.EPOCH_LABEL[e])
    rows.append({"epoch": H.EPOCH_LABEL[e], "cover_m2": round(H.cover(R[e])[r0:r1, c0:c1].sum() * g["res"] ** 2, 2), "max_canopy_height_m": round(float(np.nanmax(chm)), 2)})
for a in axs[n:]: a.axis("off")
fig.colorbar(im, ax=axs, label="canopy height (m)", shrink=.6); H.save(fig, "A3_location_epochs"); H.table(pd.DataFrame(rows), "A3_location_table")'''),
("md", "**To write (A3):** what changed and why (season, burial / erosion, different conditions or density during acquisition). Only ground changes bigger than the level of detection (notebook C) are real."),
("md", "## A4 - Method to separate vegetation from sand\n1. **Remove low outliers** (points far below their neighbours pull the cloth down).\n2. **Cloth Simulation Filter (CSF)**: coarse ground + non-ground candidates; few parameters, works on slopes.\n3. **DTM** from the ground points (median per 25 cm cell).\n4. A candidate is **vegetation** if it is **10 cm - 1.5 m above the DTM** (10 cm = about 3 x the 3 cm noise of bare sand; slightly more on slopes) **and not flat** (eigenvalue shape: planarity < 0.65 and linearity / sphericity high enough). Plant points are 3D, sand ripples are planar.\n5. Final DTM from everything that is not vegetation; gates repeated."),
("code", '''pts = H.read_points(E, box=(x0 - 6, y0 - 0.3, x0 + 6, y0 + 0.3)); hag = pts["z"] - H.sample(R[E]["dtm"], R[E]["grid_dtm"], pts["x"], pts["y"])
fig, ax = plt.subplots(figsize=(9, 3)); isv = (hag >= H.H_MIN) & (hag <= H.H_MAX)
ax.plot(pts["x"][~isv], pts["z"][~isv], ".", c="#c8a96e", ms=2, label="ground / sand"); ax.plot(pts["x"][isv], pts["z"][isv], ".", c="#2a7f3f", ms=2, label="vegetation (> 10 cm above DTM)")
xs = np.linspace(x0 - 6, x0 + 6, 300); ax.plot(xs, H.sample(R[E]["dtm"], R[E]["grid_dtm"], xs, np.full_like(xs, y0)), "k-", lw=1, label="DTM")
ax.set_xlabel("x (m)"); ax.set_ylabel("height NAP (m)"); ax.legend(markerscale=4, fontsize=7); H.save(fig, "A4_profile")'''),
("md", "**To write (A4):** why this method given the dataset: sparse low vegetation and noise of 3 cm (so a 10 cm gate), slopes (so CSF + slope-aware gate instead of one global height threshold), 250 pts/m2 (so neighbourhood shape from a 20 cm radius). Validation: the GNSS check of the ground model (separate field data) and the biomass samples (notebook C)."),
]

# ============================================================================ B
B = [
("md", "# B - Vegetation, tussocks, volume and biomass (B1-B4)"),
("code", START + '''
E = "20250327" if "20250327" in EP else EP[-1]       # epoch of the manual biomass samples
r = R[E]; t = H.tussocks(r)                          # one row per tussock'''),
("md", "## B1 - Vegetation / ground separation applied to the study area"),
("code", '''data_m2 = H.has_data(r).sum() * H.FINE_RES ** 2
print(f"Vegetation points: {100 * r['n_veg'] / r['n_points']:.2f} % of all points")
print(f"Vegetation cover: {H.cover_m2(r):.1f} m2 = {100 * H.cover_m2(r) / data_m2:.2f} % of the {data_m2:.0f} m2 that has LiDAR data")'''),
("code", '''fig, ax = plt.subplots(figsize=(7, 7)); g = r["grid_dtm"]
ax.imshow(H.hillshade(r), extent=H.extent(g), cmap="gray")
from scipy.ndimage import binary_dilation
big = binary_dilation(H.cover(r), iterations=4)                                    # enlarged for display only
ax.imshow(np.ma.masked_where(~big, big), extent=H.extent(r["grid_fine"]), cmap="autumn", vmin=0, vmax=1)
ax.set_title(f"Detected vegetation (yellow), {H.EPOCH_LABEL[E]}  -  white = no LiDAR data"); H.save(fig, "B1_vegetation_map")'''),
("code", '''sl = np.kron(r["slope"], np.ones((5, 5)))[::2, ::2][:r["grid_fine"]["nr"], :r["grid_fine"]["nc"]]      # slope on the 0.10 m grid
ok = H.has_data(r); cls = pd.cut(pd.Series(sl[ok]), [-1, 5, 10, 20, 90], labels=["0-5", "5-10", "10-20", ">20"])
tab = pd.DataFrame({"cover_%": (100 * pd.Series(H.cover(r)[ok]).groupby(cls.values, observed=True).mean()).round(2),
                    "area_m2": (pd.Series(np.ones(ok.sum())).groupby(cls.values, observed=True).sum() * H.FINE_RES ** 2).round(0)})
H.table(tab.reset_index().rename(columns={"index": "slope (deg)"}), "B1_cover_vs_slope")'''),
("md", "**To write (B1):** the vegetation fraction, the map, and obvious misclassification: vegetation below ~10 cm is missed; steep slopes and wet or rough sand can give false detections (cover increases with slope in the table: real, or a slope artefact?); poles / people were removed by height and shape rules."),
("md", "## B2 - Characterisation: tussock volume\nTussocks are found with DBSCAN on the vegetation points (points < 20 cm apart belong together). The intermediate quantity is the **canopy volume**, computed four ways: **column** (canopy height integrated over the footprint, closed at the base - main choice), **convex hull**, **voxels** and **alpha shape**."),
("code", '''vols = ["vol_column", "vol_hull", "vol_voxel", "vol_alpha"]
tot = t[vols].sum().to_frame("total volume (m3)").T.round(2); tot.insert(0, "tussocks", len(t)); H.table(tot, "B2_volume_totals")'''),
("code", '''fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, m in zip(axs, ["vol_hull", "vol_voxel", "vol_alpha"]):
    ax.loglog(t.vol_column, t[m], ".", c="#2a7f3f"); lim = [t.vol_column.min() / 2, t[vols].max().max() * 2]; ax.plot(lim, lim, "k--", lw=.6)
    ax.set_xlabel("column volume (m3)"); ax.set_ylabel(m + " (m3)")
H.save(fig, "B2_estimator_comparison"); t[vols].corr(method="spearman").round(2)'''),
("code", '''rows = []                                            # sensitivity to the key parameters
for vox in [0.03, 0.05, 0.10]:
    for ar in [0.10, 0.15, 0.25]:
        s = H.tussocks(r, voxel=vox, alpha_radius=ar); rows.append({"voxel (m)": vox, "alpha radius (m)": ar, "vol_voxel": s.vol_voxel.sum(), "vol_alpha": s.vol_alpha.sum(), "vol_column": s.vol_column.sum()})
H.table(pd.DataFrame(rows).round(2), "B2_sensitivity")'''),
("md", "**To write (B2):** column volume is the main estimate (closed base, robust to the parameters); hull over-estimates (fills space between lobes); voxel and alpha volumes change by several times with their size parameter, so they are descriptors, not unbiased volumes (see the sensitivity table)."),
("md", "## B3 - Volume to biomass\n**AGB = volume x rho.** rho (kg dry biomass per m3 of canopy volume) is set in `helpers.py` (`RHO`): **placeholder values** until fixed from literature / the manual samples (notebook C)."),
("code", '''show = pd.concat([t.nlargest(1, "vol_column"), t.iloc[[len(t) // 2]], t.nsmallest(1, "vol_column")]).copy(); show["patch"] = ["largest", "medium", "smallest"]
H.table(show[["patch", "footprint_m2", "h_max", "vol_column", "agb_low_kg", "agb_mid_kg", "agb_high_kg"]].round(3), "B3_patches")'''),
("md", "**To write (B3):** where rho comes from and its assumptions (species, dry vs fresh mass, canopy density). Are the differences between the patches logical (bigger plant = more biomass; similar plants = similar kg per m2)?"),
("md", "## B4 - Uncertainty\nTwo blocks: **structure** (how tussocks are found / measured: DBSCAN distance, minimum points, minimum height) and **conversion** (rho). Every combination is run."),
("code", '''import itertools
rows = []
for eps, ms, mh in itertools.product([0.15, 0.20, 0.30], [5, 8, 12], [0.08, 0.10, 0.15]):
    s = H.tussocks(r, eps=eps, min_samples=ms, min_height=mh, alpha=False); rows.append({"eps": eps, "min_samples": ms, "min_height": mh, "tussocks": len(s), "volume_m3": s.vol_column.sum()})
sw = H.table(pd.DataFrame(rows).round(3), "B4_parameter_sweep")
base = t.vol_column.sum()
unc = pd.DataFrame({"low": [sw.volume_m3.min(), base * H.RHO["low"]], "base": [base, base * H.RHO["mid"]], "high": [sw.volume_m3.max(), base * H.RHO["high"]]},
                   index=["structure: volume (m3) over all parameter sets", "conversion: biomass (kg) for rho low / mid / high"]).round(2)
H.table(unc.reset_index(), "B4_uncertainty_budget")'''),
("code", '''fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, p in zip(axs, ["eps", "min_samples", "min_height"]):
    sw.groupby(p).volume_m3.agg(["min", "median", "max"]).plot(ax=ax, marker="o"); ax.set_title("effect of " + p)
axs[0].set_ylabel("total volume (m3)"); H.save(fig, "B4_parameter_effects")'''),
("md", "**To write (B4):** which block dominates (compare the relative spread of the two rows) and what that means. Also name density / noise (structure) and species, moisture and dry vs fresh mass (conversion)."),
]

# ============================================================================ C
C = [
("md", "# C - Validation, change through time, research question (C1-C5)"),
("code", START + '''
TT = {e: H.tussocks(R[e], alpha=False) for e in EP}                               # tussocks of every epoch
BL = pd.concat([H.blocks(R[e], TT[e]) for e in EP], ignore_index=True)           # 5 m blocks of every epoch
BC = H.common_area(BL)                                                           # blocks with LiDAR data in ALL epochs
print(f"Common area: {len(BC[BC.epoch == EP[0]]) * H.BLOCK_M ** 2:.0f} m2")'''),
("md", "## C1 - Quality check of the biomass estimate\n### (a) Manual biomass samples (destructive sampling, Sander Vos)"),
("code", '''import os
f = H.ROOT / "data" / "biomass_measurements.csv"
S = pd.read_csv(f); S.columns = ["x", "y", "measured_g_per_m3"]          # 7 samples: x, y (RD) and dry biomass in g per m3
E = "20250327" if "20250327" in EP else EP[-1]; g = R[E]["grid_dtm"]; v = R[E]["veg"]
inside = (S.x > g["x0"]) & (S.x < g["x0"] + g["nc"] * g["res"]) & (S.y < g["y1"]) & (S.y > g["y1"] - g["nr"] * g["res"])
print(f"{inside.sum()} of {len(S)} samples lie inside the study area; epoch used: {H.EPOCH_LABEL[E]}")
if inside.sum() == 0: print("-> clip the area around the samples (see README) and change BOX in helpers.py")
rows = []
for _, s in S[inside].iterrows():
    m = np.hypot(v["x"] - s.x, v["y"] - s.y) < 1.0                              # vegetation points within 1 m of the sample
    xyh = np.column_stack([v["x"][m], v["y"][m], v["hag"][m]])
    vol = H.vol_column(xyh) if m.sum() > 5 else 0.0
    rows.append({"x": s.x, "y": s.y, "measured (g/m3)": s.measured_g_per_m3, "veg points within 1 m": int(m.sum()),
                 "LiDAR canopy volume (m3)": round(vol, 3), "max height (m)": round(float(v["hag"][m].max()), 2) if m.any() else 0.0,
                 "veg points per m3": round(m.sum() / vol, 0) if vol > 0 else np.nan})
H.table(pd.DataFrame(rows), "C1_biomass_samples")'''),
("code", '''meas = S.measured_g_per_m3 / 1000                                   # kg per m3, same unit as rho
print("Measured density (kg/m3): min %.3f | 25 %% %.3f | median %.3f | 75 %% %.3f | max %.3f" % (meas.min(), meas.quantile(.25), meas.median(), meas.quantile(.75), meas.max()))
print("Placeholder rho (kg/m3):", H.RHO)'''),
("md", "**To write (C1a):** the measured values are biomass per m3 - the same quantity as rho - so they test the conversion directly: where do low / mid / high rho sit in the measured distribution? Set rho from the measured quartiles (25 % / median / 75 %) if the sampled volume is the canopy volume (ask Sander how the volume was defined). Only a few samples: this is a plausibility check, not a calibration or machine learning."),
("md", "### (b) Noise floor of ground change and (c) agreement of the volume methods"),
("code", '''rows = []
for a, b in zip(EP[:-1], EP[1:]):
    d, lod = H.ground_change(R[a], R[b]); rows.append({"from": H.EPOCH_LABEL[a], "to": H.EPOCH_LABEL[b], "median change (m)": round(lod["median"], 3), "sigma (m)": round(lod["sigma"], 3), "LoD 95% (m)": round(lod["lod95"], 3)})
H.table(pd.DataFrame(rows), "C1_level_of_detection")'''),
("code", '''tv = H.tussocks(R[E])
print("Rank agreement of the volume estimators (Spearman):"); tv[["vol_column", "vol_hull", "vol_voxel", "vol_alpha"]].corr(method="spearman").round(2)'''),
("md", "## C2 - Maps of vegetation and biomass, and what drives them"),
("code", '''n = len(EP); cols = 3; fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(12, 3.8 * int(np.ceil(n / cols)))); axs = axs.ravel()
for k, e in enumerate(EP):
    s = BL[BL.epoch == e].copy(); s.loc[s.data_share <= .5, "cover_m2"] = np.nan; g = R[e]["grid_dtm"]
    im = axs[k].imshow(H.block_image(s, "cover_m2"), extent=H.extent(g), cmap="Greens", vmin=0, vmax=BL.cover_m2.max()); axs[k].set_title(H.EPOCH_LABEL[e]); axs[k].tick_params(labelsize=6)
for a in axs[n:]: a.axis("off")
fig.colorbar(im, ax=axs, label="vegetation cover per 5 m block (m2); white = no data", shrink=.6); H.save(fig, "C2_cover_maps")'''),
("code", '''from scipy.stats import spearmanr
rows = []
for e in EP:
    s = BC[BC.epoch == e]
    for drv in ["slope_deg", "z_nap"]:
        r_, p_ = spearmanr(s[drv], s.agb_mid_kg, nan_policy="omit"); rows.append({"epoch": H.EPOCH_LABEL[e], "driver": drv, "spearman r": round(r_, 2), "p": round(p_, 4)})
H.table(pd.DataFrame(rows), "C2_driver_correlations").pivot(index="epoch", columns="driver", values="spearman r").loc[[H.EPOCH_LABEL[e] for e in EP]]'''),
("md", "**To write (C2):** compare with what you expect for a juvenile dune (vegetation behind the beach, on higher / sheltered parts; exposure to wind and water; distance to the shoreline). z_nap = height of the block in NAP."),
("md", "## C3 - Change through time (common area only)"),
("code", '''g_ = BC.groupby("epoch")
ts = pd.DataFrame({"epoch": [H.EPOCH_LABEL[e] for e in EP], "cover_m2": [g_.cover_m2.sum()[e] for e in EP], "tussocks": [g_.n_tussocks.sum()[e] for e in EP],
                   "volume_m3": [g_.vol_m3.sum()[e] for e in EP], "biomass_mid_kg": [g_.agb_mid_kg.sum()[e] for e in EP]})
ts["biomass_low_kg"] = ts.biomass_mid_kg * H.RHO["low"] / H.RHO["mid"]; ts["biomass_high_kg"] = ts.biomass_mid_kg * H.RHO["high"] / H.RHO["mid"]
H.table(ts.round(2), "C3_timeseries")'''),
("code", '''fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(ts.epoch, ts.cover_m2, "o-", c="#2a7f3f"); ax[0].set_ylabel("vegetation cover (m2)")
ax[1].fill_between(ts.epoch, ts.biomass_low_kg, ts.biomass_high_kg, color="#2a7f3f", alpha=.25); ax[1].plot(ts.epoch, ts.biomass_mid_kg, "o-", c="#2a7f3f"); ax[1].set_ylabel("biomass (kg), rho low - high")
for a in ax: a.tick_params(axis="x", rotation=40)
H.save(fig, "C3_timeseries")'''),
("code", '''pairs = [(EP[0], EP[len(EP) // 2]), (EP[len(EP) // 2], EP[-1]), (EP[0], EP[-1])]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (a, b) in zip(axs, pairs):
    d, lod = H.ground_change(R[a], R[b]); m = np.where(np.abs(d) > lod["lod95"], d, np.nan)
    im = ax.imshow(m, extent=H.extent(R[a]["grid_dtm"]), cmap="RdBu", vmin=-1, vmax=1); ax.tick_params(labelsize=6)
    ax.set_title(f"{H.EPOCH_LABEL[a]} -> {H.EPOCH_LABEL[b]}\\nonly |change| > LoD ({lod['lod95']:.2f} m)", fontsize=8)
fig.colorbar(im, ax=axs, label="ground change (m)", shrink=.8); H.save(fig, "C3_ground_change")'''),
("code", '''a, b = BL[(BL.epoch == EP[0]) & (BL.data_share > .9)], BL[(BL.epoch == EP[-1]) & (BL.data_share > .9)]
hot = a.merge(b, on=["row", "col"], suffixes=("_first", "_last"))                 # only blocks with data in both epochs
hot = pd.DataFrame({"x": hot.x_first, "y": hot.y_first, "slope_deg": hot.slope_deg_first.round(1), "biomass_first_kg": hot.agb_mid_kg_first.round(2),
                    "biomass_last_kg": hot.agb_mid_kg_last.round(2)}); hot["change_kg"] = (hot.biomass_last_kg - hot.biomass_first_kg).round(2)
H.table(pd.concat([hot.nsmallest(5, "change_kg"), hot.nlargest(5, "change_kg")]), "C3_hotspots")'''),
("md", "**To write (C3):** total change in cover and biomass, the largest losses and gains (hotspots), and the drivers (season, sand burial / erosion, wind). Compare any big step between epochs with the level of detection and with acquisition differences before calling it real change."),
("md", "## Research question - does biomass differ between dune slopes?"),
("code", '''s = BC.copy(); s["slope_class"] = pd.cut(s.slope_deg, [-1, 5, 10, 15, 90], labels=["0-5", "5-10", "10-15", ">15"])
mean = s.groupby("slope_class", observed=True)[["agb_mid_kg", "cover_m2"]].mean(); mean.insert(0, "blocks", s.groupby("slope_class", observed=True).size())
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for e, c in zip([EP[0], EP[len(EP) // 2], EP[-1]], ["#1b9e77", "#d95f02", "#7570b3"]):
    q = s[s.epoch == e]; axs[0].plot(q.slope_deg, q.agb_mid_kg, ".", c=c, alpha=.7, label=H.EPOCH_LABEL[e])
axs[0].set_xlabel("block mean slope (deg)"); axs[0].set_ylabel("biomass per 5 m block (kg)"); axs[0].legend(fontsize=7)
mean.agb_mid_kg.plot.bar(ax=axs[1], color="#2a7f3f"); axs[1].set_ylabel("mean biomass per block (kg)")
for i, n_ in enumerate(mean.blocks): axs[1].text(i, mean.agb_mid_kg.iloc[i], f"n={n_}", ha="center", va="bottom", fontsize=7)
H.save(fig, "RQ_biomass_vs_slope"); H.table(mean.round(3).reset_index(), "RQ_slope_classes")'''),
("code", '''from scipy.stats import rankdata
def partial(x, y, z):                                  # Spearman correlation of x and y after removing the effect of z
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    return np.corrcoef(rx - np.polyval(np.polyfit(rz, rx, 1), rz), ry - np.polyval(np.polyfit(rz, ry, 1), rz))[0, 1]
rows = []
for e in EP:
    q = s[(s.epoch == e) & s.z_nap.notna()]
    r_, p_ = spearmanr(q.slope_deg, q.agb_mid_kg); rows.append({"epoch": H.EPOCH_LABEL[e], "blocks": len(q), "spearman (slope, biomass)": round(r_, 2), "p": round(p_, 4),
                                                              "partial, elevation held fixed": round(partial(q.slope_deg, q.agb_mid_kg, q.z_nap), 2)})
H.table(pd.DataFrame(rows), "RQ_slope_correlations")'''),
("md", "**To write (RQ):** direction and strength of the slope - biomass relation, whether it survives with elevation held fixed (partial correlation), and confounding: biomass depends on elevation / distance to the sea, burial and wind exposure as well as slope; vegetation detection can also be biased on steep slopes."),
("md", "## C4 - Discussion\nHow the vegetation and biomass patterns may influence dune development at the Zandmotor, and whether your data are sufficient or only indicative (number of epochs, one small area, placeholder or few-sample rho).\n\n## C5 - Reflection\nHow this LiDAR workflow compares with satellite imagery and manual sampling; recommendations to improve the estimates and the quality assessment (more biomass samples, species map, repeat GNSS on bare sand, density-matched acquisitions)."),
]

make("A_explore_vegetation.ipynb", A)
make("B_biomass.ipynb", B)
make("C_validation_and_change.ipynb", C)
print("notebooks written")
if "--run" in sys.argv:
    from nbclient import NotebookClient
    out = ROOT / "notebooks_executed"; out.mkdir(exist_ok=True)
    for f in sorted(NB.glob("*.ipynb")):
        nb = nbf.read(f, 4)
        NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, out / f.name); print("executed", f.name)
