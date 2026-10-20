"""
MODULE: terrain_aba.py
PURPOSE: Area-Based Approach (ABA) 2.5D raster generation and multi-epoch geomorphology.
ASSIGNMENT QUESTIONS: A1 (Pulse Density Analysis), A3 (Spatio-Temporal Dynamics), and RQ (Dune Slope)
"""

import numpy as np

def rasters(coords: np.ndarray, ground_idx: np.ndarray, veg_idx: np.ndarray, grid_res: float = 0.10):
    """
    Generates 2.5D DTM, DSM, and CHM rasters via direct vectorized binning.
    Returns: dtm, chm, slope_deg, raster_bounds
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    ncols = int(np.ceil((x_max - x_min) / grid_res)) + 1
    nrows = int(np.ceil((y_max - y_min) / grid_res)) + 1

    def map_pts_to_cells(pts):
        col_idx = np.clip(np.floor((pts[:, 0] - x_min) / grid_res).astype(np.int32), 0, ncols - 1)
        row_idx = np.clip(np.floor((pts[:, 1] - y_min) / grid_res).astype(np.int32), 0, nrows - 1)
        return row_idx, col_idx

    # DTM: lowest bare ground elevation per cell
    ground_pts = coords[ground_idx]
    flat_dtm = np.full(nrows * ncols, np.inf, dtype=np.float32)
    if len(ground_pts) > 0:
        gr, gc = map_pts_to_cells(ground_pts)
        np.minimum.at(flat_dtm, gr * ncols + gc, ground_pts[:, 2])
    flat_dtm[np.isinf(flat_dtm)] = np.nan
    dtm = flat_dtm.reshape((nrows, ncols))
    default_ground = np.nanmin(ground_pts[:, 2]) if len(ground_pts) > 0 else 0.0
    dtm = np.nan_to_num(dtm, nan=default_ground)

    # DSM: maximum canopy elevation per cell
    veg_pts = coords[veg_idx]
    flat_dsm = np.full(nrows * ncols, -np.inf, dtype=np.float32)
    if len(veg_pts) > 0:
        vr, vc = map_pts_to_cells(veg_pts)
        np.maximum.at(flat_dsm, vr * ncols + vc, veg_pts[:, 2])
    flat_dsm[np.isneginf(flat_dsm)] = np.nan
    dsm = flat_dsm.reshape((nrows, ncols))
    dsm = np.nan_to_num(dsm, nan=dtm)

    # CHM: Canopy Height Model
    chm = np.maximum(dsm - dtm, 0.0)

    # dune slope in degrees via horizontal elevation gradients (grad Z)
    dy, dx = np.gradient(dtm, grid_res)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    raster_bounds = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "grid_res": grid_res,
        "nrows": nrows, "ncols": ncols
    }

    return dtm, chm, slope_deg, raster_bounds


def compute_epoch_differences(dtm_t1: np.ndarray, dtm_t2: np.ndarray,
                              chm_t1: np.ndarray, chm_t2: np.ndarray):
    """
    Multi-temporal raster change detection (Question A3):
      - delta_dtm > 0: Aeolian sand accretion / burial around stems
      - delta_dtm < 0: Wind erosion / dune blowout
      - delta_chm > 0: Biological vegetation growth / canopy expansion
      - delta_chm < 0: Canopy burial or seasonal senescence
    """
    # delta_dtm = dtm_t2 - dtm_t1
    # delta_chm = chm_t2 - chm_t1
    pass
    # return delta_dtm, delta_chm