# Vegetation biomass from UAV LiDAR (AESM5230C) - simple version

One helper file (`helpers.py`, ~400 lines, short plain functions) and three notebooks - one per part of the assignment.
No `modules/` folder.

```
helpers.py                     ALL functions + the settings you may change (top of the file)
prepare_data.py                STEP 1: merge the .laz clips -> one file per epoch
find_and_clip.py               (before step 1) cut a small area out of the big strip files
notebooks/A_explore_vegetation.ipynb          A1-A4
notebooks/B_biomass.ipynb                     B1-B4
notebooks/C_validation_and_change.ipynb       C1-C5 + research question
data/clips/                    your clipped .laz files (any sub-folders; the file name must contain the date)
data/biomass_measurements.csv  the 7 manual biomass samples (Sander Vos)
figures/  tables/              made by the notebooks (PNG / CSV for the report)
```

## Results
`figures/` and `tables/` hold the results of the run on the study area (x 72290-72510, y 451410-451640, 8 epochs Feb 2024 - Oct 2026).
`notebooks/` are saved WITH their outputs (executed), so they can be read on GitHub and appended to the report.
The big LiDAR files are not in the repository (see `.gitignore`).

## Run (Windows PowerShell, in this folder)
```
pip install -r requirements.txt
# 0) only if you still have the big strip files: cut out the study area (all epoch folders at once)
foreach ($d in Get-ChildItem data -Directory | Where-Object { $_.Name -match '^\d{8}' }) { python find_and_clip.py $d.FullName "data\clips\$($d.Name)" 72290 451410 72510 451640 --density 250 }
python prepare_data.py          # 1) merge into data/merged/<date>_field.laz (keeps at most 400 pts/m2; raw counts go to A1)
jupyter lab                     # 2) run notebooks A, B, C top to bottom
```
The first cell of each notebook runs the ground / vegetation separation for every epoch (about 1 minute per epoch) and
caches it in `data/cache`, so later runs are fast. Change a parameter in `helpers.py` -> delete `data/cache` -> rerun.

## What you may change (top of helpers.py)
`BOX` (study area), `DENSITY` (250 pts/m2), `H_MIN` / `H_MAX` (10 cm - 1.5 m), `EPS` / `MIN_SAMPLES` (tussock clustering),
`RHO` (kg biomass per m3 canopy volume: PLACEHOLDER 0.3 / 0.8 / 1.4 - set it from the manual samples / literature).

## The method in 5 lines (A4)
1 remove low outliers - 2 Cloth Simulation Filter (ground + candidates) - 3 DTM from the ground - 4 a candidate is
vegetation if it is 10 cm - 1.5 m above the DTM (a bit more on slopes) and not planar (eigenvalue shape) - 5 final DTM.
Tussocks = DBSCAN on the vegetation points; volume (column / hull / voxel / alpha); biomass = volume x rho.

## Limits to state in the report
No harvest data except 7 manual samples; vegetation < ~10 cm is missed (cover is a lower bound); species are not labelled;
time series use only the area with LiDAR data in all epochs; rho is the largest uncertainty until it is calibrated.
