"""helpers.py - ALL helper functions of the project in one file, used by the three notebooks (A, B, C).

Sections:  1 settings (edit here)  |  2 reading the data  |  3 ground / vegetation separation
           4 tussocks and volumes  |  5 maps, change through time, plots
Every function is short and has a one-line description. Nothing else is needed (no modules folder).
"""
import os
import pickle
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

# =============================================================================== 1. SETTINGS (edit here)
ROOT = Path(__file__).resolve().parent
CLIPS_DIR = ROOT / "data" / "clips"        # put your .laz clips here (any sub-folders)
MERGED_DIR = ROOT / "data" / "merged"      # made by prepare_data.py: one file per epoch
CACHE_DIR = ROOT / "data" / "cache"        # results of the ground / vegetation separation
FIG_DIR, TAB_DIR = ROOT / "figures", ROOT / "tables"
for _d in (MERGED_DIR, CACHE_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# study area in RD New (metres): x_min, y_min, x_max, y_max  (default = around the manual biomass samples)
BOX = tuple(float(v) for v in os.environ["BOX"].split(",")) if "BOX" in os.environ else (72290.0, 451410.0, 72510.0, 451640.0)
EPOCH_LABEL = {"20230927": "Sep 2023", "20240227": "Feb 2024", "20240501": "May 2024", "20240607": "Jun 2024",
               "20240717": "Jul 2024", "20250327": "Mar 2025", "20250514": "May 2025", "20260204": "Feb 2026",
               "20261007": "Oct 2026"}
DENSITY = 250.0          # every epoch is thinned to this many points / m2, so epochs are comparable
H_MIN, H_MAX = 0.10, 1.5  # vegetation = between 10 cm (sand noise is ~3 cm) and 1.5 m above the ground
EPS, MIN_SAMPLES = 0.20, 8  # DBSCAN: points closer than 20 cm belong to the same tussock
RHO = dict(low=0.3, mid=0.8, high=1.4)   # kg dry biomass per m3 of canopy volume. PLACEHOLDER - replace / tune!
BLOCK_M = 5.0            # block size for maps (m)
DTM_RES, FINE_RES = 0.25, 0.10   # raster cell sizes (m)


def epochs():
    """Epochs for which a merged file exists, in time order."""
    return [e for e in EPOCH_LABEL if (MERGED_DIR / f"{e}_field.laz").exists()]


# =============================================================================== 2. READING
def read_points(epoch, density=DENSITY, box=None, seed=0):
    """Read one merged epoch, crop to the study area and thin randomly to `density` points/m2. Returns dict x, y, z, rgb."""
    x0, y0, x1, y1 = box or BOX
    las = laspy.read(MERGED_DIR / f"{epoch}_field.laz")
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if density:
        frac = min(1.0, density * (x1 - x0) * (y1 - y0) / max(keep.sum(), 1))
        keep &= np.random.default_rng(seed).random(len(x)) < frac
    rgb = np.column_stack([las.red, las.green, las.blue])[keep].astype(np.float32)
    rgb /= 255.0 if rgb.max() <= 255 else 65535.0
    return {"x": x[keep], "y": y[keep], "z": z[keep], "rgb": rgb}


def describe_files(epoch_list):
    """A1: points, extent, density, spacing and attributes of every epoch in the study area (before thinning)."""
    f = MERGED_DIR / "epoch_stats.csv"
    st = pd.read_csv(f, dtype={"epoch": str}).set_index("epoch") if f.exists() else None
    area = (BOX[2] - BOX[0]) * (BOX[3] - BOX[1])
    rows = []
    for e in epoch_list:
        with laspy.open(MERGED_DIR / f"{e}_field.laz") as r:
            h = r.header
            n = int(st.loc[e, "points_in_area"]) if st is not None else h.point_count
            rows.append({"epoch": EPOCH_LABEL[e], "points": n,
                         "x range (m)": f"{h.mins[0]:.0f}-{h.maxs[0]:.0f}", "y range (m)": f"{h.mins[1]:.0f}-{h.maxs[1]:.0f}",
                         "z range (m)": f"{h.mins[2]:.2f}-{h.maxs[2]:.2f}", "density (pts/m2)": round(n / area, 1),
                         "spacing (cm)": round(100 / np.sqrt(n / area), 1), "has RGB": "red" in h.point_format.dimension_names})
    return pd.DataFrame(rows)


# =============================================================================== 3. GROUND / VEGETATION SEPARATION
def make_grid(res, box=None):
    """A raster grid over the study area: dict with x0 (left), y1 (top), res, nr rows, nc columns."""
    x0, y0, x1, y1 = box or BOX
    return {"x0": x0, "y1": y1, "res": res, "nr": int(np.ceil((y1 - y0) / res)), "nc": int(np.ceil((x1 - x0) / res))}


def cell_of(g, x, y):
    """Flat cell index of every point on grid g, and a mask of the points that fall inside the grid."""
    c = np.floor((x - g["x0"]) / g["res"]).astype(np.int64)
    r = np.floor((g["y1"] - y) / g["res"]).astype(np.int64)
    ok = (c >= 0) & (c < g["nc"]) & (r >= 0) & (r < g["nr"])
    return r * g["nc"] + c, ok


def cell_stat(g, x, y, v, how="median"):
    """Per-cell median / max / mean / count of values v (NaN where a cell has no points)."""
    idx, ok = cell_of(g, x, y)
    s = pd.Series(np.asarray(v)[ok]).groupby(idx[ok]).agg(how)
    out = np.full(g["nr"] * g["nc"], np.nan, np.float32)
    out[s.index.values] = s.values
    return out.reshape(g["nr"], g["nc"])


def fill_holes(a):
    """Fill empty (NaN) cells: linear interpolation, nearest value outside the data."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    r, c = np.nonzero(~bad)
    out = a.astype(np.float64).copy()
    out[bad] = griddata(np.column_stack((c, r)), a[~bad], np.column_stack(np.nonzero(bad)[::-1]), method="linear")
    left = ~np.isfinite(out)
    if left.any():
        i = ndi.distance_transform_edt(left, return_distances=False, return_indices=True)
        out[left] = out[i[0][left], i[1][left]]
    return out.astype(np.float32)


def sample(a, g, x, y):
    """Value of raster a (grid g) at positions x, y (bilinear)."""
    return ndi.map_coordinates(a.astype(np.float64), [(g["y1"] - y) / g["res"] - 0.5, (x - g["x0"]) / g["res"] - 0.5],
                               order=1, mode="nearest")


def slope_deg(dtm, res):
    """Terrain slope in degrees (DTM smoothed over ~1 m first)."""
    gy, gx = np.gradient(ndi.gaussian_filter(dtm.astype(np.float64), 1.0 / res), res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def csf_ground(xyz, resolution=0.5, rigidness=2, threshold=0.10):
    """Cloth Simulation Filter: returns the indices of ground points and of non-ground candidates."""
    import CSF
    c = CSF.CSF()
    c.params.bSloopSmooth, c.params.cloth_resolution, c.params.rigidness = True, resolution, rigidness
    c.params.class_threshold, c.params.time_step, c.params.interations = threshold, 0.65, 500   # (sic) library spelling
    c.setPointCloud(np.ascontiguousarray(xyz, dtype=np.float64))
    g, ng = CSF.VecInt(), CSF.VecInt()
    c.do_filtering(g, ng, False)
    return np.array(g, dtype=np.int64), np.array(ng, dtype=np.int64)


def low_outliers(xyz, drop=0.40):
    """True for points more than `drop` m below the median height of their 1 m cell (noise that pulls the cloth down)."""
    g = make_grid(1.0, (xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 0].max() + 1, xyz[:, 1].max() + 1))
    med = cell_stat(g, xyz[:, 0], xyz[:, 1], xyz[:, 2], "median").ravel()
    idx, ok = cell_of(g, xyz[:, 0], xyz[:, 1])
    return xyz[:, 2] < med[np.clip(idx, 0, med.size - 1)] - drop


def local_shape(xyz, idx, radius=0.20, k=48, min_neighbours=8, chunk=50_000):
    """Eigenvalue shape of the neighbourhood (radius 20 cm) of points xyz[idx]: planarity, linearity, sphericity.
    Points with too few neighbours get NaN and sparse=True."""
    tree = cKDTree(xyz)
    out = {n: np.full(len(idx), np.nan, np.float32) for n in ("planarity", "linearity", "sphericity")}
    sparse = np.zeros(len(idx), bool)
    for s in range(0, len(idx), chunk):
        sel = slice(s, s + chunk)
        d, i = tree.query(xyz[idx[sel]], k=k, distance_upper_bound=radius, workers=-1)
        ok = np.isfinite(d); i[~ok] = 0
        n = ok.sum(1); P = xyz[i]
        mean = (P * ok[..., None]).sum(1) / np.maximum(n, 1)[:, None]
        D = (P - mean[:, None, :]) * ok[..., None]
        ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", D, D) / np.maximum(n - 1, 1)[:, None, None])   # l3 <= l2 <= l1
        l3, l2, l1 = ev[:, 0], ev[:, 1], np.maximum(ev[:, 2], 1e-12)
        good = n >= min_neighbours
        out["linearity"][sel] = np.where(good, (l1 - l2) / l1, np.nan)
        out["planarity"][sel] = np.where(good, (l2 - l3) / l1, np.nan)
        out["sphericity"][sel] = np.where(good, l3 / l1, np.nan)
        sparse[sel] = ~good
    return out, sparse


def separate(p):
    """THE METHOD (A4/B1). Input: points p (x, y, z, rgb). Steps:
    1 remove low outliers  2 CSF -> ground + candidates  3 DTM from the ground  4 candidates are vegetation if they are
    10 cm - 1.5 m above the DTM (a bit more on slopes) and not flat (eigenvalue shape)  5 final DTM + rasters."""
    keep = ~low_outliers(np.column_stack([p["x"], p["y"], p["z"]]))
    x, y, z, rgb = p["x"][keep], p["y"][keep], p["z"][keep], p["rgb"][keep]
    xyz = np.column_stack([x, y, z])
    gd, gf = make_grid(DTM_RES), make_grid(FINE_RES)
    ground, cand = csf_ground(xyz)
    dtm = fill_holes(cell_stat(gd, x[ground], y[ground], z[ground]))
    hag = z - sample(dtm, gd, x, y)
    slope = slope_deg(dtm, DTM_RES)
    h_min = lambda i: H_MIN + 0.5 * DTM_RES * np.tan(np.radians(sample(slope, gd, x[i], y[i])))   # slope-aware gate
    shape, sparse = local_shape(xyz, cand)
    flat = (shape["planarity"] >= 0.65) | ((shape["linearity"] <= 0.30) & (shape["sphericity"] <= 0.15))
    is_veg = (hag[cand] >= h_min(cand)) & (hag[cand] <= H_MAX) & (sparse | ~flat)
    ground = np.concatenate([ground, cand[~is_veg]])                       # candidates that are not vegetation are ground too
    dtm = fill_holes(cell_stat(gd, x[ground], y[ground], z[ground]))      # final, cleaner DTM
    slope = slope_deg(dtm, DTM_RES)
    hag = z - sample(dtm, gd, x, y)
    veg = cand[is_veg]
    veg = veg[(hag[veg] >= h_min(veg)) & (hag[veg] <= H_MAX)]            # same gates on the final DTM
    bright, exg = rgb.mean(1), 2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    shape_v, _ = local_shape(xyz, veg)
    isveg = np.zeros(len(x), bool); isveg[veg] = True
    idx, ok = cell_of(gf, x, y)
    n = gf["nr"] * gf["nc"]
    return {"grid_dtm": gd, "grid_fine": gf, "dtm": dtm, "slope": slope,
            "chm": cell_stat(gf, x, y, np.maximum(hag, 0), "max"),
            "npts": np.bincount(idx[ok], minlength=n).reshape(gf["nr"], gf["nc"]),
            "veg_count": np.bincount(idx[ok & isveg], minlength=n).reshape(gf["nr"], gf["nc"]),
            "veg": {"x": x[veg], "y": y[veg], "z": z[veg], "hag": hag[veg], "rgb": rgb[veg], "brightness": bright[veg],
                    "exg": exg[veg], **shape_v},
            "n_points": len(x), "n_veg": len(veg)}


def process(epoch, force=False):
    """Read + separate one epoch; the result is cached in data/cache so it is only computed once."""
    f = CACHE_DIR / f"{epoch}.pkl"
    if f.exists() and not force:
        return pickle.load(open(f, "rb"))
    R = separate(read_points(epoch)); R["epoch"] = epoch
    pickle.dump(R, open(f, "wb"))
    return R


def cover(R):
    """Vegetation cover raster (True/False, 0.10 m cells). Isolated 1-2 cell false alarms are removed."""
    lab, n = ndi.label(R["veg_count"] > 0)
    size = np.bincount(lab.ravel())
    return (size[lab] >= 3) & (lab > 0)


def cover_m2(R):
    return cover(R).sum() * R["grid_fine"]["res"] ** 2


def has_data(R):
    """True where the epoch really has LiDAR points (judged on 1 m cells), on the 0.10 m grid."""
    k = int(round(1 / FINE_RES)); g = R["grid_fine"]
    occ = block_reduce(R["npts"].astype(float), k, "sum") > 0
    out = np.zeros((g["nr"], g["nc"]), bool); m = np.kron(occ, np.ones((k, k), bool))
    out[:m.shape[0], :m.shape[1]] = m[:g["nr"], :g["nc"]]
    return out


def to_dtm_grid(a_fine, R):
    """Resample a 0.10 m raster to the 0.25 m grid (nearest cell centre)."""
    gd, gf = R["grid_dtm"], R["grid_fine"]
    ri = np.minimum(((np.arange(gd["nr"]) + .5) * gd["res"] / gf["res"]).astype(int), a_fine.shape[0] - 1)
    ci = np.minimum(((np.arange(gd["nc"]) + .5) * gd["res"] / gf["res"]).astype(int), a_fine.shape[1] - 1)
    return a_fine[np.ix_(ri, ci)]


# =============================================================================== 4. TUSSOCKS AND VOLUMES
def vol_column(xyh, cell=0.10):
    """Volume = sum over 10 cm columns of (highest point above ground x cell area). Closed at the base. MAIN METHOD."""
    ij = np.floor((xyh[:, :2] - xyh[:, :2].min(0)) / cell).astype(np.int64)
    return float(pd.Series(xyh[:, 2]).groupby(ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]).max().sum() * cell * cell)


def vol_hull(xyh):
    """Convex hull volume (points are also projected down to the ground so the hull is closed at the base)."""
    base = xyh.copy(); base[:, 2] = 0
    try:
        return float(ConvexHull(np.vstack([xyh, base])).volume)
    except (QhullError, ValueError):
        return np.nan


def vol_voxel(xyh, voxel=0.05):
    """Number of occupied voxels x voxel volume (porous lower bound - depends strongly on the voxel size)."""
    ijk = np.floor((xyh - xyh.min(0)) / voxel).astype(np.int64)
    return float(len(np.unique(ijk, axis=0)) * voxel ** 3)


def vol_alpha(xyh, radius=0.15, max_pts=30000):
    """Alpha-shape volume: union of the Delaunay tetrahedra whose circumscribed sphere has radius <= `radius`."""
    P = np.unique(np.round(xyh, 3), axis=0)
    if len(P) < 5:
        return np.nan
    if len(P) > max_pts:
        P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
    try:
        s = P[Delaunay(P).simplices]
    except (QhullError, ValueError):
        return np.nan
    u, v, w = s[:, 1] - s[:, 0], s[:, 2] - s[:, 0], s[:, 3] - s[:, 0]
    det = np.einsum("ij,ij->i", u, np.cross(v, w))
    num = (u * u).sum(1)[:, None] * np.cross(v, w) + (v * v).sum(1)[:, None] * np.cross(w, u) + (w * w).sum(1)[:, None] * np.cross(u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.linalg.norm(num, axis=1) / (2 * np.abs(det))
    return float((np.abs(det)[np.isfinite(R) & (R <= radius)] / 6).sum())


def tussocks(R, eps=EPS, min_samples=MIN_SAMPLES, min_height=H_MIN, voxel=0.05, alpha_radius=0.15, alpha=True, rho=None):
    """B2/B3: find tussocks with DBSCAN on the vegetation points; one row per tussock with size, 4 volumes and biomass."""
    from sklearn.cluster import DBSCAN
    v = R["veg"]; rho = rho or RHO
    lab = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(np.column_stack([v["x"], v["y"]]))
    order = np.argsort(lab, kind="stable"); sl = lab[order]
    ids = np.unique(sl[sl >= 0])
    rows = []
    for cid, s, e in zip(ids, np.searchsorted(sl, ids), np.searchsorted(sl, ids, side="right")):
        i = order[s:e]
        if len(i) < 30:
            continue
        xyh = np.column_stack([v["x"][i], v["y"][i], v["hag"][i]]); h = xyh[:, 2]
        foot = len(np.unique(np.floor((xyh[:, :2] - xyh[:, :2].min(0)) / 0.10).astype(np.int64), axis=0)) * 0.01
        if h.max() < min_height or h.max() > H_MAX or h.max() / np.sqrt(max(foot, 1e-6)) > 1.5:
            continue          # too low (noise), too tall (pole / person) or too thin (stake) to be a tussock
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyh.T)))[::-1]
        cx, cy = xyh[:, 0].mean(), xyh[:, 1].mean()
        rows.append({"epoch": R["epoch"], "x": cx, "y": cy, "n_pts": len(i), "footprint_m2": foot, "h_max": h.max(),
                     "h_p95": np.percentile(h, 95), "linearity": (ev[0] - ev[1]) / ev[0], "planarity": (ev[1] - ev[2]) / ev[0],
                     "sphericity": ev[2] / ev[0], "brightness": v["brightness"][i].mean(), "exg": v["exg"][i].mean(),
                     "slope_deg": float(sample(R["slope"], R["grid_dtm"], np.array([cx]), np.array([cy]))[0]),
                     "vol_column": vol_column(xyh), "vol_hull": vol_hull(xyh), "vol_voxel": vol_voxel(xyh, voxel),
                     "vol_alpha": vol_alpha(xyh, alpha_radius) if alpha else np.nan})
    t = pd.DataFrame(rows)
    for k in ("low", "mid", "high"):
        t[f"agb_{k}_kg"] = t["vol_column"] * rho[k] if len(t) else []     # BIOMASS = VOLUME x RHO
    return t


# =============================================================================== 5. MAPS, CHANGE THROUGH TIME
def block_reduce(a, n, how="mean"):
    """Aggregate n x n cells (mean / sum / max); NaN-aware."""
    nr, nc = (a.shape[0] // n) * n, (a.shape[1] // n) * n
    b = a[:nr, :nc].astype(np.float64).reshape(nr // n, n, nc // n, n)
    f = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}[how]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); return f(b, axis=(1, 3))


def blocks(R, t=None, n_m=BLOCK_M):
    """Cover, tussock volume / biomass, slope, height and data coverage per 5 m block (for maps and the research question)."""
    t = tussocks(R, alpha=False) if t is None else t
    gd, gf = R["grid_dtm"], R["grid_fine"]
    nf, nd = int(round(n_m / gf["res"])), int(round(n_m / gd["res"]))
    cov = block_reduce(cover(R).astype(float), nf, "sum") * gf["res"] ** 2
    share = block_reduce(block_reduce(R["npts"].astype(float), 10, "sum") > 0, int(n_m), "mean")
    vol, agb, cnt = np.zeros(cov.shape), np.zeros(cov.shape), np.zeros(cov.shape)
    if len(t):
        c = np.floor((t.x.values - gd["x0"]) / n_m).astype(int); r = np.floor((gd["y1"] - t.y.values) / n_m).astype(int)
        ok = (r >= 0) & (r < cov.shape[0]) & (c >= 0) & (c < cov.shape[1])
        np.add.at(vol, (r[ok], c[ok]), t.vol_column.values[ok]); np.add.at(agb, (r[ok], c[ok]), t.agb_mid_kg.values[ok])
        np.add.at(cnt, (r[ok], c[ok]), 1)
    rr, cc = np.indices(cov.shape)
    return pd.DataFrame({"epoch": R["epoch"], "row": rr.ravel(), "col": cc.ravel(),
                         "x": gd["x0"] + (cc.ravel() + .5) * n_m, "y": gd["y1"] - (rr.ravel() + .5) * n_m,
                         "cover_m2": cov.ravel(), "n_tussocks": cnt.ravel(), "vol_m3": vol.ravel(), "agb_mid_kg": agb.ravel(),
                         "slope_deg": block_reduce(R["slope"], nd, "mean").ravel(), "z_nap": block_reduce(R["dtm"], nd, "mean").ravel(),
                         "data_share": share.ravel()})


def common_area(all_blocks, min_share=0.9):
    """Keep only blocks that have LiDAR data in EVERY epoch - otherwise totals through time are not comparable."""
    ok = all_blocks[all_blocks.data_share > min_share].groupby(["row", "col"]).size()
    keep = ok[ok == all_blocks.epoch.nunique()].index
    return all_blocks[pd.MultiIndex.from_frame(all_blocks[["row", "col"]]).isin(keep)]


def block_image(df, column):
    """2D array (rows x columns) of one block column - for imshow maps."""
    a = np.full((df.row.max() + 1, df.col.max() + 1), np.nan)
    a[df.row.values, df.col.values] = df[column].values
    return a


def ground_change(R1, R2, slope_max=5.0):
    """Ground height change R2 - R1 (m) where both epochs have data, and its 95 % level of detection (from flat cells)."""
    ok = to_dtm_grid(has_data(R1) & has_data(R2), R1)
    d = np.where(ok, R2["dtm"] - R1["dtm"], np.nan)
    flat = d[(R1["slope"] < slope_max) & np.isfinite(d)]
    med = np.median(flat); sigma = 1.4826 * np.median(np.abs(flat - med))
    return d, {"median": med, "sigma": sigma, "lod95": 1.96 * sigma}


def hillshade(R, az=315.0, alt=45.0):
    """Shaded relief of the DTM (for map backgrounds); NaN where the epoch has no data."""
    g = R["grid_dtm"]
    gy, gx = np.gradient(R["dtm"].astype(np.float64), g["res"])
    s, a = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    hs = np.clip(np.cos(np.radians(90 - alt)) * np.cos(s) + np.sin(np.radians(90 - alt)) * np.sin(s) * np.cos(np.radians(360 - az + 90) - a), 0, 1)
    return np.where(to_dtm_grid(has_data(R), R), hs, np.nan)


def extent(g):
    """imshow extent (x_min, x_max, y_min, y_max) of grid g."""
    return (g["x0"], g["x0"] + g["nc"] * g["res"], g["y1"] - g["nr"] * g["res"], g["y1"])


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def table(df, name):
    """Save a table as CSV (for the report) and return it for display."""
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    return df
