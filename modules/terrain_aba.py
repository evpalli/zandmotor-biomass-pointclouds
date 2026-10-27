"""
MODULE: terrain_aba.py
PURPOSE: Area-Based Approach (ABA) 2.5D raster generation and multi-epoch geomorphology.
ASSIGNMENT QUESTIONS: A1, A3, C2, C3
"""

from pathlib import Path
import laspy
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def characteristics(file_paths: list[str]) -> pd.DataFrame:
    """Computes point counts, spatial bounds, footprint area, and pulse density (Question A1)."""
    records = []
    for path_str in file_paths:
        path = Path(path_str)
        with laspy.open(path) as f:
            las = f.read()

        total_pts = len(las.points)
        x_min, x_max = float(las.x.min()), float(las.x.max())
        y_min, y_max = float(las.y.min()), float(las.y.max())
        z_min, z_max = float(las.z.min()), float(las.z.max())

        area = (x_max - x_min) * (y_max - y_min)
        density = total_pts / area if area > 0 else 0.0

        records.append({
            "Epoch": path.stem.split("_")[0],
            "File Name": path.name,
            "Total Points": total_pts,
            "X Bounds (m)": f"[{x_min:.1f}, {x_max:.1f}]",
            "Y Bounds (m)": f"[{y_min:.1f}, {y_max:.1f}]",
            "Z Range (m)": f"[{z_min:.2f}, {z_max:.2f}]",
            "Area (m²)": round(area, 2),
            "Pulse Density (pts/m²)": round(density, 2),
            "Attributes": ", ".join(list(las.point_format.dimension_names)),
        })
    return pd.DataFrame(records)


def rasters(
    coords: np.ndarray,
    ground_idx: np.ndarray,
    veg_idx: np.ndarray,
    grid_res: float = 0.10,
    bounds_override: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generates DTM, CHM, and Dune Slope rasters via vectorized NumPy binning.

    Allows bounding box override for multi-epoch co-registration.
    """
    if bounds_override is not None:
        x_min, x_max = bounds_override["x_min"], bounds_override["x_max"]
        y_min, y_max = bounds_override["y_min"], bounds_override["y_max"]
    else:
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    ncols = int(np.ceil((x_max - x_min) / grid_res)) + 1
    nrows = int(np.ceil((y_max - y_min) / grid_res)) + 1

    bounds = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "grid_res": grid_res,
        "nrows": nrows,
        "ncols": ncols,
    }

    def get_indices(pts):
        c = np.clip(
            np.floor((pts[:, 0] - x_min) / grid_res).astype(np.int64),
            0,
            ncols - 1,
        )
        r = np.clip(
            np.floor((y_max - pts[:, 1]) / grid_res).astype(np.int64),
            0,
            nrows - 1,
        )
        return r * ncols + c

    n_cells = nrows * ncols

    # 1. DTM: Minimum ground elevation
    ground_pts = coords[ground_idx]
    dtm_flat = np.full(n_cells, np.inf, dtype=np.float32)
    if len(ground_pts) > 0:
        np.minimum.at(
            dtm_flat, get_indices(ground_pts), ground_pts[:, 2].astype(np.float32)
        )
    dtm_flat[~np.isfinite(dtm_flat)] = np.nan
    dtm = dtm_flat.reshape(nrows, ncols)

    # 2. DSM: Maximum vegetation elevation
    veg_pts = coords[veg_idx]
    dsm_flat = np.full(n_cells, -np.inf, dtype=np.float32)
    if len(veg_pts) > 0:
        np.maximum.at(
            dsm_flat, get_indices(veg_pts), veg_pts[:, 2].astype(np.float32)
        )
    dsm_flat[~np.isfinite(dsm_flat)] = np.nan
    dsm = dsm_flat.reshape(nrows, ncols)

    # Gap-fill DTM with nearest neighbor
    empty_dtm = ~np.isfinite(dtm)
    if empty_dtm.all():
        raise ValueError("No ground points found inside bounding extent.")
    if empty_dtm.any():
        nearest = distance_transform_edt(
            empty_dtm, return_distances=False, return_indices=True
        )
        dtm[empty_dtm] = dtm[nearest[0][empty_dtm], nearest[1][empty_dtm]]

    # 3. CHM: DSM - DTM clipped at 0
    dsm[~np.isfinite(dsm)] = dtm[~np.isfinite(dsm)]
    chm = np.maximum(dsm - dtm, 0.0)

    # 4. Dune Slope (degrees) via elevation gradient
    dy, dx = np.gradient(dtm, grid_res)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)

    return dtm, chm, slope_deg, bounds


def epoch_differences(
    dtm_t1: np.ndarray,
    dtm_t2: np.ndarray,
    chm_t1: np.ndarray,
    chm_t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates geomorphic and canopy change grids between two coregistered dates (Questions A3, C3).

    - delta_dtm > 0: Aeolian sand accretion / burial
    - delta_dtm < 0: Wind deflation / erosion
    - delta_chm > 0: Biological growth / lateral expansion
    - delta_chm < 0: Physical canopy burial or seasonal senescence
    """
    if dtm_t1.shape != dtm_t2.shape or chm_t1.shape != chm_t2.shape:
        raise ValueError("Epoch grids must share identical pixel dimensions.")
    return dtm_t2 - dtm_t1, chm_t2 - chm_t1