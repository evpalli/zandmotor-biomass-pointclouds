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

def rasters(
    coords: np.ndarray,
    ground_idx: np.ndarray,
    veg_idx: np.ndarray,
    grid_res: float = 0.10,
):
    """
    Generate DTM, DSM, CHM and slope rasters using vectorised binning.

    DTM: minimum ground elevation in each cell.
    DSM: maximum vegetation elevation in each cell.
    CHM: DSM - DTM, clipped at zero.
    Slope: terrain slope in degrees.
    """

    # 1. Define raster bounds
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()

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

    print(
        f"Raster grid: {nrows:,} rows x {ncols:,} columns "
        f"({nrows * ncols:,} cells)",
        flush=True,
    )

    # 2. Convert coordinates to raster indices
    # Rows are measured from the northern edge.
    def get_flat_indices(points):
        cols = np.floor((points[:, 0] - x_min) / grid_res).astype(np.int64)
        rows = np.floor((y_max - points[:, 1]) / grid_res).astype(np.int64)

        # Clamp indices to valid raster bounds.
        cols = np.clip(cols, 0, ncols - 1)
        rows = np.clip(rows, 0, nrows - 1)

        return rows * ncols + cols

    n_cells = nrows * ncols

    # 3. DTM: minimum ground elevation per cell
    ground_pts = coords[ground_idx]

    dtm_flat = np.full(n_cells, np.inf, dtype=np.float32)

    if len(ground_pts) > 0:
        ground_cells = get_flat_indices(ground_pts)

        np.minimum.at(
            dtm_flat,
            ground_cells,
            ground_pts[:, 2].astype(np.float32))

    dtm_flat[~np.isfinite(dtm_flat)] = np.nan
    dtm = dtm_flat.reshape(nrows, ncols)

    # 4. DSM: maximum vegetation elevation per cell
    veg_pts = coords[veg_idx]

    dsm_flat = np.full(n_cells, -np.inf, dtype=np.float32)

    if len(veg_pts) > 0:
        veg_cells = get_flat_indices(veg_pts)

        np.maximum.at(
            dsm_flat,
            veg_cells,
            veg_pts[:, 2].astype(np.float32),
        )

    dsm_flat[~np.isfinite(dsm_flat)] = np.nan
    dsm = dsm_flat.reshape(nrows, ncols)

    # 5. Fill missing DTM cells using nearest valid DTM cells
    empty_dtm = ~np.isfinite(dtm)

    if empty_dtm.all():
        raise ValueError("Cannot build DTM: no valid ground elevations were found.")

    if empty_dtm.any():
        nearest = distance_transform_edt(empty_dtm,return_distances=False,return_indices=True)

        dtm[empty_dtm] = dtm[nearest[0][empty_dtm], nearest[1][empty_dtm]]

    # 6. Where vegetation is absent, set DSM equal to DTM
    empty_dsm = ~np.isfinite(dsm)
    dsm[empty_dsm] = dtm[empty_dsm]

    # 7. CHM
    chm = np.maximum(dsm - dtm, 0.0)

    # 8. Slope in degrees
    dy, dx = np.gradient(dtm, grid_res)

    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    return dtm, chm, slope_deg, bounds


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