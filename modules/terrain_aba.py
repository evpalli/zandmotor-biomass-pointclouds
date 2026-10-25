"""
MODULE: terrain_aba.py
PURPOSE: Area-Based Approach (ABA) 2.5D raster generation and multi-epoch geomorphology.
ASSIGNMENT QUESTIONS: A1 (Pulse Density Analysis), A3 (Spatio-Temporal Dynamics), and RQ (Dune Slope)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

def cell_indexes(x, y, bounds):
    """points -> Grida: assign points to grid"""
    res = bounds["grid_res"]
    col = np.floor((x - bounds["x_min"]) / res).astype(np.int64)
    row = np.floor((y - bounds["y_min"]) / res).astype(np.int64)
    return row, col


def sample_raster(raster, bounds, x, y):
    """Grid -> points: reads the raster value under every (x, y); NaN if outside the grid."""
    row, col = cell_indexes(x, y, bounds)
    ok = (row >= 0) & (row < raster.shape[0]) & (col >= 0) & (col < raster.shape[1])
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[ok] = raster[row[ok], col[ok]]
    return out


def rasters(coords: np.ndarray, ground_idx: np.ndarray, veg_idx: np.ndarray, grid_res: float = 0.10):
    """
    Generates 2.5D DTM, DSM, and CHM rasters via direct vectorized binning.
    Returns: dtm, chm, slope_deg, raster_bounds
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max() # all row, 1st column (x) - left to right
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max() # all row, 2nd column (y)

    ncols = int(np.ceil((x_max - x_min) / grid_res)) + 1
    nrows = int(np.ceil((y_max - y_min) / grid_res)) + 1

    # dictionary storing where raster grid information
    raster_bounds = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
                    "grid_res": grid_res, "nrows": nrows, "ncols": ncols,}

    # DTM: lowest ground elevation per grid cell
    ground_pts = coords[ground_idx]

    dtm = np.full((nrows, ncols), np.nan, dtype=np.float32) # initialise empty grid

    for r in range(nrows):
        for c in range(ncols):
            x_min_cell = raster_bounds["x_min"] + c * grid_res
            x_max_cell = x_min_cell + grid_res

            y_max_cell = raster_bounds["y_max"] - r * grid_res
            y_min_cell = y_max_cell - grid_res

            inside = ((ground_pts[:, 0] >= x_min_cell) & # find ground points inside cell
                    (ground_pts[:, 0] < x_max_cell) &
                    (ground_pts[:, 1] >= y_min_cell) &
                    (ground_pts[:, 1] < y_max_cell))

            cell_ground_pts = ground_pts[inside]

            # Assign the lowest ground elevation to this cell
            if len(cell_ground_pts) > 0:
                dtm[r, c] = np.min(cell_ground_pts[:, 2])

    dtm = dtm[nearest[0], nearest[1]]
    # DSM: maximum canopy elevation per cell
    veg_pts = coords[veg_idx]
    dsm = np.full((nrows, ncols), np.nan, dtype=np.float32) # initialise empty grid dsm

    for r in range(nrows):
        for c in range(ncols):
            x_min_cell = raster_bounds["x_min"] + c * grid_res
            x_max_cell = x_min_cell + grid_res

            y_max_cell = raster_bounds["y_max"] - r * grid_res
            y_min_cell = y_max_cell - grid_res

            inside_veg = ((veg_pts[:, 0] >= x_min_cell) & # find ground points inside cell
                    (veg_pts[:, 0] < x_max_cell) &
                    (veg_pts[:, 1] >= y_min_cell) &
                    (veg_pts[:, 1] < y_max_cell))

            cell_veg_pts = veg_pts[inside_veg]

            # Assign the lowest ground elevation to this cell
            if len(cell_veg_pts) > 0:
                dsm[r, c] = np.max(cell_veg_pts[:, 2])
    # make sure of empty DTM and DSM. if empty DTM -> nearest cell, if empty DSM -> DSM = DTM (e.g. no vegetation)
    empty = np.isnan(dtm)
    if empty.any():
        nearest = distance_transform_edt(empty,return_distances=False, return_indices=True)
    empty_dsm = np.isnan(dsm)
    dsm[empty_dsm] = dtm[empty_dsm]
    dtm = dtm[nearest[0], nearest[1]]

    # CHM: Canopy Height Model
    chm = np.maximum(dsm - dtm, 0.0)

    # dune slope in degrees via horizontal elevation gradients (grad Z)
    dy, dx = np.gradient(dtm, grid_res)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

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
    if dtm_t1.shape != dtm_t2.shape or chm_t1.shape != chm_t2.shape:
        raise ValueError(
            "The two epochs must be on the same grid (same shape). rasters() builds its grid "
        )
    delta_dtm = dtm_t2 - dtm_t1
    delta_chm = chm_t2 - chm_t1
    return delta_dtm, delta_chm