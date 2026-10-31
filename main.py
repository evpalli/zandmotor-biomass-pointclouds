<<<<<<< Updated upstream
from pathlib import Path
=======
"""
MODULE: main.py
PURPOSE:
    Main pipeline for the dune LiDAR workflow.

    - A1: LAS/LAZ point-cloud characteristics.
    - C2: DTM, CHM and terrain-slope raster generation.
    - A3/C3: Optional multi-epoch DTM and CHM change detection.
    - Existing workflow: ground/vegetation classification, VARI,
      vegetation clustering, geometric features, volume estimates,
      species typing and enriched LAS/LAZ output.

EXPECTED PROJECT STRUCTURE:
    project/
        main.py
        modules/
            __init__.py
            terrain_aba.py
            ground_filter.py
            spectral_typing.py
            biomass_itd.py

EXAMPLE:
    python main.py epoch1.laz out.laz --grid-res 0.25

    python main.py epoch1.laz out.laz \
        --compare-with epoch2.laz --grid-res 0.25

NOTES:
    - This script expects terrain_aba.py to expose:
      characteristics(), rasters(), and epoch_differences().
    - The updated terrain_aba.py shown in the conversation does not
      expose sample_raster(), so a compatible sampler is implemented here.
    - GeoTIFF output is optional and requires rasterio.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import laspy
>>>>>>> Stashed changes
import numpy as np
import laspy
from scipy.spatial import cKDTree, ConvexHull
import CSF
import gc
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist

<<<<<<< Updated upstream
def initialise_grid_ABA(las):

    # Extract X, Y, and Z coordinates
    x = las.x
    y = las.y
    z = las.z
=======
from modules.biomass_itd import (
    alpha_shape_vol,
    compute_geometric_features,
    convex_hull_vol,
    voxel_3d,
    export_voxels_to_obj,
)
from modules.ground_filter import classify_ground_and_vegetation
from modules.spectral_typing import (
    classify_pioneer_morphology,
    compute_vari,
)
from modules.terrain_aba import (
    characteristics,
    rasters,
    epoch_differences,
)

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None
    from_origin = None
>>>>>>> Stashed changes

    points = np.column_stack((x, y))
    values = z

<<<<<<< Updated upstream
    # Define the grid resolution
    resolution = 2.0 
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    grid_x, grid_y = np.mgrid[x_min:x_max:resolution, y_min:y_max:resolution]

    grid_z = griddata(points, values, (grid_x, grid_y), method="linear")
    
    return grid_x, grid_y, grid_z, points, values
    '''
    auf we want to visualise it

    plt.subplot(221)
    plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
    plt.plot(points[:,0], points[:,1], 'k.', ms=1)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
    plt.title('Nearest')
    plt.subplot(223)
    plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
    plt.title('Linear')
    plt.subplot(224)
    plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    plt.title('Cubic')
    plt.gcf().set_size_inches(6, 6)
    plt.show()
    '''

def run_csf_ground_filter(coords, resolution=0.6, rigidness=2, threshold=0.12):
    """Ground separation via Cloth Simulation Filtering (CSF)."""
    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = resolution
    csf.params.rigidness = rigidness
    csf.params.class_threshold = threshold
    csf.params.time_step = 0.65
    csf.params.interations = 500

    csf.setPointCloud(coords.tolist())
    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices, False)

    return np.array(ground_indices), np.array(non_ground_indices)

def classify_grids(points, values, ground_indices, non_ground_indices, grid_x, grid_y):
    """Generates independent grids for ground and non-ground points."""

    grid_ground = griddata(points[ground_indices], values[ground_indices], (grid_x, grid_y), method="linear")
    return grid_ground, grid_non_ground

def compute_geometric_features(coords, candidate_idx, radius=0.20, min_neighbors=8):
    """Refinement computing eigenvalue descriptors (Kucak, 2022)."""
    kdtree = cKDTree(coords)
    candidate_pts = coords[candidate_idx]
    n_candidates = len(candidate_pts)

    planarity = np.zeros(n_candidates, dtype=np.float32)
    linearity = np.zeros(n_candidates, dtype=np.float32)
    sphericity = np.zeros(n_candidates, dtype=np.float32)

    for i, pt in enumerate(candidate_pts):
        neighbor_indices = kdtree.query_ball_point(pt, r=radius)
        if len(neighbor_indices) >= min_neighbors:
            neighbors = coords[neighbor_indices]
            cov = np.cov(neighbors, rowvar=False)

            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(np.maximum(eigenvalues, 1e-9))[::-1]
            l1, l2, l3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

            planarity[i] = (l2 - l3) / l1
            linearity[i] = (l1 - l2) / l1
            sphericity[i] = l3 / l1

    return planarity, linearity, sphericity

def predictory_variables(grid_ground, grid_non_ground, non_ground_coords):
    """
    - Tree height (via Canopy Height Model difference)
    - Crown diameter (via max pairwise distance on convex hull vertices)
    - Crown density (via 2D histogram of raw non-ground points)
    - ConvexHull of non-ground point cloud
    """
    # Canopy Height Model (CHM) difference
    chm = grid_non_ground - grid_ground
    tree_height = np.nanmax(chm)

    convexhull = ConvexHull(non_ground_coords)
    hull_pts = non_ground_coords[hull.vertices]
    crown_diameter = np.max(pdist(hull_pts, metric='euclidean'))

    crown_density, _, _ = np.histogram2d(non_ground_coords[:, 0], non_ground_coords[:, 1], bins=10)

    return chm, tree_height, convexhull, crown_diameter, crown_density

# chekc later if we want to keep this
def CHM(coords, veg_idx, ground_idx, grid_resolution=0.25):
    ground_pts = coords[ground_idx]
    veg_pts = coords[veg_idx]

    # Grid bounds
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    ncols = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
    nrows = int(np.ceil((y_max - y_min) / grid_resolution)) + 1

    # Map point coordinates directly to grid cell row/col indices
    def points_to_grid(pts):
        col_idx = np.floor((pts[:, 0] - x_min) / grid_resolution).astype(np.int32)
        row_idx = np.floor((pts[:, 1] - y_min) / grid_resolution).astype(np.int32)
        col_idx = np.clip(col_idx, 0, ncols - 1)
        row_idx = np.clip(row_idx, 0, nrows - 1)
        return row_idx, col_idx

    # Initialize Rasters
    dtm = np.full((nrows, ncols), np.nan, dtype=np.float32)
    dsm = np.full((nrows, ncols), np.nan, dtype=np.float32)

    if len(ground_pts) > 0:
        g_rows, g_cols = points_to_grid(ground_pts)
        flat_idx = g_rows * ncols + g_cols
        # np.minimum.at handles duplicates per cell efficiently
        flat_dtm = np.full(nrows * ncols, np.inf, dtype=np.float32)
        np.minimum.at(flat_dtm, flat_idx, ground_pts[:, 2])
        flat_dtm[np.isinf(flat_dtm)] = np.nan
        dtm = flat_dtm.reshape((nrows, ncols))

    if len(veg_pts) > 0:
        v_rows, v_cols = points_to_grid(veg_pts)
        flat_idx = v_rows * ncols + v_cols
        flat_dsm = np.full(nrows * ncols, -np.inf, dtype=np.float32)
        np.maximum.at(flat_dsm, flat_idx, veg_pts[:, 2])
        flat_dsm[np.isneginf(flat_dsm)] = np.nan
        dsm = flat_dsm.reshape((nrows, ncols))

    default_ground = np.nanmin(ground_pts[:, 2]) if len(ground_pts) > 0 else 0.0
    dtm = np.nan_to_num(dtm, nan=default_ground)
    dsm = np.nan_to_num(dsm, nan=dtm)

    chm = np.maximum(dsm - dtm, 0.0)
    
    grid_x = np.linspace(x_min, x_max, ncols)
    grid_y = np.linspace(y_min, y_max, nrows)

    return chm, grid_x, grid_y

# has not been adapted to aba style
def main(input_laz: str, output_laz: str):
    print(f"Loading point cloud: {input_laz}")
    las = laspy.read(input_laz)
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    total_points = len(coords)

    # 1. CSF Filter
    ground_idx, candidate_veg_idx = run_csf_ground_filter(coords)
    print(f"CSF ground: {len(ground_idx)}, candidate veg: {len(candidate_veg_idx)}")

    # 2. Geometric Covariance Refinement
    print("Computing eigenvalue geometric features...")
    planarity, linearity, sphericity = compute_geometric_features(coords, candidate_veg_idx)

    # 3. Filter remaining planar sand residuals
    is_true_veg = (planarity < 0.65) & ((linearity > 0.30) | (sphericity > 0.15))
    final_veg_idx = candidate_veg_idx[is_true_veg]
    residual_sand_idx = candidate_veg_idx[~is_true_veg]
    final_ground_idx = np.concatenate((ground_idx, residual_sand_idx))

    # 4. Standard ASPRS Classification
    las.classification[:] = 0
    las.classification[final_ground_idx] = 2  # Ground
    las.classification[final_veg_idx] = 4     # Medium/Low Vegetation

    # 5. Compute Predictor Variables
    height, diameter, density, volume = predictory_variables(coords, final_veg_idx, final_ground_idx, radius=0.5, min_neighbors=4)

    # 6. Generate Canopy Height Model (CHM)
    chm, grid_x, grid_y = CHM(coords, final_veg_idx, final_ground_idx, grid_resolution=0.1)

    # 5. Add custom scalar fields to the single file for CloudCompare
    print("Adding diagnostic scalar fields...")
    las.add_extra_dim(laspy.ExtraBytesParams(name="stage_csf", type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name="stage_refined", type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name="planarity", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="tree_height", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="crown_diameter", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="crown_density", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="crown_volume", type=np.float32))

    # Stage 1 flag: 1 = Candidate vegetation after CSF, 0 = Ground
    stage1 = np.zeros(total_points, dtype=np.uint8)
    stage1[candidate_veg_idx] = 1
    las.stage_csf = stage1

    # Stage 2 flag: 1 = Confirmed vegetation after eigenvalues, 0 = Ground
    stage2 = np.zeros(total_points, dtype=np.uint8)
    stage2[final_veg_idx] = 1
    las.stage_refined = stage2

    # Continuous Planarity field
    full_planarity = np.zeros(total_points, dtype=np.float32)
    full_planarity[candidate_veg_idx] = planarity
    las.planarity = full_planarity
  # height, diameter, density, volume 
    full_tree_height = np.zeros(total_points, dtype=np.float32)
    full_tree_height[final_veg_idx] = height
    las.tree_height = full_tree_height

    full_crown_diam = np.zeros(total_points, dtype=np.float32)
    full_crown_diam[final_veg_idx] = diameter
    las.crown_diameter = full_crown_diam

    full_density = np.zeros(total_points, dtype=np.float32)
    full_density[final_veg_idx] = density
    las.crown_density = full_density

    full_volume = np.zeros(total_points, dtype=np.float32)
    full_volume[final_veg_idx] = volume
    las.crown_volume = full_volume
    # Ensure output directory exists and write
    Path(output_laz).parent.mkdir(parents=True, exist_ok=True)
    las.write(output_laz)
    print(f"Classified multi-stage file saved to: {output_laz}")

=======
# LAS classification codes (ASPRS standard).
GROUND_CLASS = 2
VEGETATION_CLASSES = [3, 4, 5]

SPECIES_CODES = {
    "Dry Drift / Senescent Matter": 1,
    "Ammophila arenaria (Marram Grass)": 2,
    "Elytrigia juncea (Sand Couch)": 3,
    "Unclassified (missing RGB or too few points)": 4,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def write_dim(las, name, values, dtype):
    """Add or overwrite a per-point LAS extra dimension."""
    values = np.asarray(values, dtype=dtype)

    if len(values) != len(las.points):
        raise ValueError(
            f"Dimension '{name}' has {len(values)} values; "
            f"expected {len(las.points)}."
        )

    if name not in las.point_format.dimension_names:
        las.add_extra_dim(
            laspy.ExtraBytesParams(name=name, type=dtype)
        )

    setattr(las, name, values)


def sample_raster(raster, bounds, x, y):
    """
    Sample a north-up raster using the same indexing convention
    as terrain_aba.rasters(): row 0 is at y_max.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    cols = np.floor(
        (x - bounds["x_min"]) / bounds["grid_res"]
    ).astype(np.int64)

    rows = np.floor(
        (bounds["y_max"] - y) / bounds["grid_res"]
    ).astype(np.int64)

    cols = np.clip(cols, 0, bounds["ncols"] - 1)
    rows = np.clip(rows, 0, bounds["nrows"] - 1)

    return raster[rows, cols]


def save_rasters(prefix, dtm, chm, slope_deg, bounds, las):
    """
    Save raster arrays to NPZ and, when rasterio is installed,
    to a three-band GeoTIFF.

    terrain_aba.rasters() indexes row 0 from y_max downward,
    which already matches the standard north-up GeoTIFF layout.
    Therefore, the arrays are not vertically flipped.
    """
    prefix = Path(prefix)
    npz_path = prefix.parent / f"{prefix.name}_rasters.npz"

    np.savez_compressed(
        npz_path,
        dtm=dtm.astype(np.float32),
        chm=chm.astype(np.float32),
        slope_deg=slope_deg.astype(np.float32),
        x_min=bounds["x_min"],
        x_max=bounds["x_max"],
        y_min=bounds["y_min"],
        y_max=bounds["y_max"],
        grid_res=bounds["grid_res"],
        nrows=bounds["nrows"],
        ncols=bounds["ncols"],
    )
    logger.info("Raster NPZ saved: %s", npz_path)

    if rasterio is None:
        logger.warning(
            "rasterio is not installed; skipping GeoTIFF output. "
            "Install it with: pip install rasterio"
        )
        return

    crs = las.header.parse_crs()
    crs_wkt = crs.to_wkt() if crs is not None else None
    res = bounds["grid_res"]

    tif_path = prefix.parent / f"{prefix.name}_rasters.tif"
    transform = from_origin(
        bounds["x_min"],
        bounds["y_max"],
        res,
        res,
    )

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=bounds["nrows"],
        width=bounds["ncols"],
        count=3,
        dtype="float32",
        crs=crs_wkt,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        for band, (array, description) in enumerate(
            (
                (dtm, "DTM"),
                (chm, "CHM"),
                (slope_deg, "slope_deg"),
            ),
            start=1,
        ):
            dst.write(array.astype(np.float32), band)
            dst.set_band_description(band, description)

    logger.info("GeoTIFF saved: %s", tif_path)


def common_bounds(coords_a, coords_b):
    """Return the overlapping XY extent for two point clouds."""
    bounds = {
        "x_min": float(max(coords_a[:, 0].min(), coords_b[:, 0].min())),
        "x_max": float(min(coords_a[:, 0].max(), coords_b[:, 0].max())),
        "y_min": float(max(coords_a[:, 1].min(), coords_b[:, 1].min())),
        "y_max": float(min(coords_a[:, 1].max(), coords_b[:, 1].max())),
    }

    if (
        bounds["x_min"] >= bounds["x_max"]
        or bounds["y_min"] >= bounds["y_max"]
    ):
        raise ValueError(
            "The two epochs do not have an overlapping XY extent."
        )

    return bounds


def load_las(path):
    """Read a LAS/LAZ file and return its LAS object and XYZ array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input point cloud not found: {path}")

    las = laspy.read(path)
    if len(las.points) == 0:
        raise ValueError(f"Point cloud is empty: {path}")

    coords = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    valid = np.isfinite(coords).all(axis=1)

    if not valid.all():
        logger.warning(
            "%s contains %d invalid coordinate records; excluding them "
            "from raster and feature calculations.",
            path.name,
            int((~valid).sum()),
        )

    return las, coords, valid


# ---------------------------------------------------------------------
# Optional multi-epoch comparison
# ---------------------------------------------------------------------

def compare_epochs(
    las_a,
    coords_a,
    ground_a,
    veg_a,
    las_b,
    coords_b,
    ground_b,
    veg_b,
    output_prefix,
    grid_res,
):
    """
    Rasterize two epochs on one shared overlapping grid and calculate
    DTM and CHM differences (epoch B minus epoch A).
    """
    bounds = common_bounds(coords_a, coords_b)

    dtm_a, chm_a, slope_a, raster_bounds = rasters(
        coords_a,
        ground_a,
        veg_a,
        grid_res=grid_res,
        bounds_override=bounds,
    )

    dtm_b, chm_b, slope_b, _ = rasters(
        coords_b,
        ground_b,
        veg_b,
        grid_res=grid_res,
        bounds_override=bounds,
    )

    delta_dtm, delta_chm = epoch_differences(
        dtm_a,
        dtm_b,
        chm_a,
        chm_b,
    )

    change_path = output_prefix.parent / (
        f"{output_prefix.name}_epoch_changes.npz"
    )

    np.savez_compressed(
        change_path,
        dtm_epoch_a=dtm_a,
        dtm_epoch_b=dtm_b,
        chm_epoch_a=chm_a,
        chm_epoch_b=chm_b,
        slope_epoch_a=slope_a,
        slope_epoch_b=slope_b,
        delta_dtm=delta_dtm,
        delta_chm=delta_chm,
        x_min=raster_bounds["x_min"],
        x_max=raster_bounds["x_max"],
        y_min=raster_bounds["y_min"],
        y_max=raster_bounds["y_max"],
        grid_res=raster_bounds["grid_res"],
        nrows=raster_bounds["nrows"],
        ncols=raster_bounds["ncols"],
    )

    logger.info("Epoch change rasters saved: %s", change_path)
    logger.info(
        "DTM change: min %.3f m, max %.3f m, mean %.3f m",
        float(np.nanmin(delta_dtm)),
        float(np.nanmax(delta_dtm)),
        float(np.nanmean(delta_dtm)),
    )
    logger.info(
        "CHM change: min %.3f m, max %.3f m, mean %.3f m",
        float(np.nanmin(delta_chm)),
        float(np.nanmax(delta_chm)),
        float(np.nanmean(delta_chm)),
    )


# ---------------------------------------------------------------------
# Main single-epoch processing pipeline
# ---------------------------------------------------------------------

def main(
    input_las_path,
    output_las_path,
    grid_res=0.10,
    voxel_size=0.5,
    compute_alpha=True,
    compare_with=None,
):
    input_path = Path(input_las_path)
    out_path = Path(output_las_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = out_path.with_suffix("")

    if grid_res <= 0 or voxel_size <= 0:
        raise ValueError("grid_res and voxel_size must be greater than zero.")

    # A1: point-cloud characteristics.
    stats_paths = [str(input_path)]
    if compare_with:
        stats_paths.append(str(compare_with))

    stats = characteristics(stats_paths)
    stats_path = prefix.parent / f"{prefix.name}_characteristics.csv"
    stats.to_csv(stats_path, index=False)
    logger.info("Point-cloud characteristics saved: %s", stats_path)

    # Load primary epoch.
    las, coords, valid = load_las(input_path)
    n_points = len(las.points)

    # Ground filtering expects an input LAS/LAZ path and returns a
    # dictionary containing integer point-index arrays (not Boolean masks).
    logger.info("Starting ground/vegetation classification.")
    ground_result = classify_ground_and_vegetation(
        str(input_path),
        output_laz_path=None,
        planarity_threshold=0.65,
        linearity_threshold=0.30,
        sphericity_threshold=0.15,
    )

    # Convert the returned point indices into Boolean masks for rasters().
    ground_idx = np.zeros(n_points, dtype=bool)
    veg_idx = np.zeros(n_points, dtype=bool)
    ground_idx[np.asarray(ground_result["ground_idx"], dtype=np.int64)] = True
    veg_idx[np.asarray(ground_result["veg_idx"], dtype=np.int64)] = True

    if len(ground_idx) != n_points or len(veg_idx) != n_points:
        raise ValueError("Ground-filter masks do not match the LAS point count.")

    # Exclude invalid coordinates from the masks used for rasterization.
    ground_for_raster = ground_idx & valid
    veg_for_raster = veg_idx & valid

    ground_veg_class = np.zeros(n_points, dtype=np.uint8)
    ground_veg_class[ground_idx] = 1
    ground_veg_class[veg_idx] = 2
    write_dim(las, "ground_veg_class", ground_veg_class, np.uint8)

    # C2: DTM, CHM and slope.
    logger.info("Generating DTM, CHM and slope rasters.")
    dtm, chm, slope_deg, bounds = rasters(
        coords,
        ground_for_raster,
        veg_for_raster,
        grid_res=grid_res,
    )

    save_rasters(prefix, dtm, chm, slope_deg, bounds, las)

    # Per-point height above ground and terrain slope.
    logger.info("Sampling DTM and slope at point locations.")
    ground_z = sample_raster(
        dtm, bounds, coords[:, 0], coords[:, 1]
    )
    height_above_ground = (
        coords[:, 2] - ground_z
    ).astype(np.float32)

    slope_pt = sample_raster(
        slope_deg, bounds, coords[:, 0], coords[:, 1]
    ).astype(np.float32)

    # Set invalid point results to NaN.
    height_above_ground[~valid] = np.nan
    slope_pt[~valid] = np.nan

    write_dim(
        las, "height_above_ground", height_above_ground, np.float32
    )
    write_dim(las, "slope_deg", slope_pt, np.float32)

    # VARI spectral index.
    logger.info("Calculating VARI.")
    dimension_names = set(las.point_format.dimension_names)
    rgb_available = {"red", "green", "blue"}.issubset(dimension_names)

    if rgb_available:
        vari = compute_vari(
            np.asarray(las.red),
            np.asarray(las.green),
            np.asarray(las.blue),
        )
        vari = np.asarray(vari, dtype=np.float32)
    else:
        logger.warning(
            "RGB dimensions are missing. VARI and RGB-based species "
            "typing will be unavailable."
        )
        vari = np.full(n_points, np.nan, dtype=np.float32)

    if len(vari) != n_points:
        raise ValueError("compute_vari() must return one value per point.")

    vari[~valid] = np.nan
    write_dim(las, "vari", vari, np.float32)

    # Initialize per-point vegetation outputs.
    tussock_id = np.zeros(n_points, dtype=np.uint32)
    species_class = np.zeros(n_points, dtype=np.uint8)

    veg_planarity = np.full(n_points, np.nan, dtype=np.float32)
    veg_linearity = np.full(n_points, np.nan, dtype=np.float32)
    veg_sphericity = np.full(n_points, np.nan, dtype=np.float32)

    vol_voxel = np.full(n_points, np.nan, dtype=np.float32)
    vol_hull = np.full(n_points, np.nan, dtype=np.float32)
    vol_alpha = np.full(n_points, np.nan, dtype=np.float32)

    # Keep only valid vegetation points for clustering.
    veg_point_mask = veg_for_raster
    veg_coords = coords[veg_point_mask]
    veg_original_indices = np.flatnonzero(veg_point_mask)

    table = pd.DataFrame()

    if len(veg_coords) > 0:
        logger.info("Starting 3D vegetation voxel clustering.")

        veg_xyz = SimpleNamespace(
            x=veg_coords[:, 0],
            y=veg_coords[:, 1],
            z=veg_coords[:, 2],
        )

        (
            voxel_set,
            voxel_indices,
            components,
            component_id,
            veg_id_volume,
            total_volume,
            points,
            point_cluster_ids,
        ) = voxel_3d(
            veg_xyz,
            connectivity = 8,
            voxel_size=voxel_size
        )

        # Export voxel geometry beside the output LAS/LAZ.
        obj_path = prefix.parent / f"{prefix.name}_voxels_{connectivity}.obj"
        export_voxels_to_obj(
            voxel_set,
            voxel_size=voxel_size,
            output_path=str(obj_path),
        )
        logger.info("Voxel OBJ saved: %s", obj_path)

        # Map each vegetation point to its connected-component ID.
        unique_voxels, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True,
        )
        inverse = inverse.reshape(-1)

        unique_ids = np.fromiter(
            (components[tuple(v)] for v in unique_voxels),
            dtype=np.int64,
            count=len(unique_voxels),
        )
        veg_cluster_ids = unique_ids[inverse]

        pts = pd.DataFrame(
            {
                "cluster_id": veg_cluster_ids,
                "x": veg_coords[:, 0],
                "y": veg_coords[:, 1],
                "z": veg_coords[:, 2],
                "height_above_ground":
                    height_above_ground[veg_original_indices],
                "ground_z": ground_z[veg_original_indices],
                "slope_deg": slope_pt[veg_original_indices],
                "vari": vari[veg_original_indices],
            }
        )

        # Cluster-level geometric features.
        cluster_features = []
        for cluster_id, group in pts.groupby("cluster_id"):
            cluster_coords = group[["x", "y", "z"]].to_numpy()
            planarity, linearity, sphericity = (
                compute_geometric_features(cluster_coords)
            )
            cluster_features.append(
                {
                    "cluster_id": cluster_id,
                    "planarity": planarity,
                    "linearity": linearity,
                    "sphericity": sphericity,
                }
            )

        cluster_features = pd.DataFrame(cluster_features)

        # Cluster summary table.
        table = pts.groupby("cluster_id").agg(
            point_count=("x", "size"),
            centroid_x=("x", "mean"),
            centroid_y=("y", "mean"),
            centroid_z=("z", "mean"),
            ground_z=("ground_z", "mean"),
            height_above_ground_max=("height_above_ground", "max"),
            height_above_ground_mean=("height_above_ground", "mean"),
            slope_deg_mean=("slope_deg", "mean"),
            slope_deg_median=("slope_deg", "median"),
            vari_median=("vari", "median"),
        )

        table = table.join(
            cluster_features.set_index("cluster_id")
        )

        # Volume estimates for each cluster.
        voxels_by_cluster = defaultdict(list)
        for voxel, cluster_id in components.items():
            voxels_by_cluster[cluster_id].append(voxel)

        table["volume_voxel_m3"] = (
            pd.Series(veg_id_volume).reindex(table.index)
        )

        table["volume_convex_hull_m3"] = [
            convex_hull_vol(
                points[point_cluster_ids == cluster_id]
            )
            for cluster_id in table.index
        ]

        if compute_alpha:
            table["volume_alpha_shape_m3"] = [
                alpha_shape_vol(
                    points[point_cluster_ids == cluster_id]
                )
                for cluster_id in table.index
            ]
        else:
            table["volume_alpha_shape_m3"] = np.nan

        # Species/morphology classification.
        labels = []
        for _, row in table.iterrows():
            height = row["height_above_ground_max"]
            linearity = row["linearity"]
            vari_value = row["vari_median"]

            if np.isfinite([height, linearity, vari_value]).all():
                # Use the function imported from the current
                # spectral_typing module.
                label = classify_pioneer_morphology(
                    height,
                    linearity,
                    vari_value,
                )
            else:
                label = (
                    "Unclassified "
                    "(missing RGB or too few points)"
                )

            labels.append(label)

        table["species"] = labels
        table["species_code"] = (
            table["species"]
            .map(SPECIES_CODES)
            .fillna(4)
            .astype(np.uint8)
        )

        # Broadcast cluster-level values back to vegetation points.
        def spread(column):
            return table[column].reindex(
                veg_cluster_ids
            ).to_numpy()

        tussock_id[veg_original_indices] = veg_cluster_ids.astype(
            np.uint32
        )
        species_class[veg_original_indices] = spread(
            "species_code"
        ).astype(np.uint8)

        veg_planarity[veg_original_indices] = spread(
            "planarity"
        ).astype(np.float32)
        veg_linearity[veg_original_indices] = spread(
            "linearity"
        ).astype(np.float32)
        veg_sphericity[veg_original_indices] = spread(
            "sphericity"
        ).astype(np.float32)

        vol_voxel[veg_original_indices] = spread(
            "volume_voxel_m3"
        ).astype(np.float32)
        vol_hull[veg_original_indices] = spread(
            "volume_convex_hull_m3"
        ).astype(np.float32)
        vol_alpha[veg_original_indices] = spread(
            "volume_alpha_shape_m3"
        ).astype(np.float32)

        for species, count in table["species"].value_counts().items():
            logger.info("%s: %d clusters", species, count)

    else:
        logger.warning(
            "No vegetation points were identified. "
            "Skipping clustering, volume estimation and species typing."
        )

    # Add vegetation and cluster attributes to LAS/LAZ.
    extra_dimensions = (
        ("tussock_id", tussock_id, np.uint32),
        ("species_class", species_class, np.uint8),
        ("veg_planarity", veg_planarity, np.float32),
        ("veg_linearity", veg_linearity, np.float32),
        ("veg_sphericity", veg_sphericity, np.float32),
        ("vol_voxel_m3", vol_voxel, np.float32),
        ("vol_convex_hull_m3", vol_hull, np.float32),
        ("vol_alpha_shape_m3", vol_alpha, np.float32),
    )

    for name, values, dtype in extra_dimensions:
        write_dim(las, name, values, dtype)

    # Save cluster-level CSV.
    csv_path = prefix.parent / f"{prefix.name}_tussocks.csv"
    table.reset_index().to_csv(csv_path, index=False)
    logger.info("Tussock table saved: %s", csv_path)

    # Save enriched LAS/LAZ.
    logger.info("Saving enriched point cloud: %s", out_path)
    las.write(str(out_path))
>>>>>>> Stashed changes

    # Optional second epoch: shared-grid raster comparison.
    if compare_with:
        logger.info("Loading comparison epoch: %s", compare_with)

        las_b, coords_b, valid_b = load_las(compare_with)

        ground_result_b = classify_ground_and_vegetation(
            str(compare_with),
            output_laz_path=None,
            planarity_threshold=0.65,
            linearity_threshold=0.30,
            sphericity_threshold=0.15,
        )

        ground_b = np.zeros(len(las_b.points), dtype=bool)
        veg_b = np.zeros(len(las_b.points), dtype=bool)
        ground_b[np.asarray(ground_result_b["ground_idx"], dtype=np.int64)] = True
        veg_b[np.asarray(ground_result_b["veg_idx"], dtype=np.int64)] = True

        ground_b &= valid_b
        veg_b &= valid_b

        compare_epochs(
            las,
            coords,
            ground_for_raster,
            veg_for_raster,
            las_b,
            coords_b,
            ground_b,
            veg_b,
            prefix,
            grid_res,
        )

    logger.info("Pipeline completed successfully.")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

if __name__ == "__main__":
<<<<<<< Updated upstream
    input_file = "data/20261007_north-F001-S001_RD.laz"
    output_file = "/Users/belinaileen/Desktop/2026_001_out.laz"
    main(input_file, output_file)

'''
Identify type of vegetation exists on Zandmotor, focus on the specific geometry to 
narrow down specific vegetation type

Main 2 approaches: Area-based Allocation (ABA) or Individual tree Detection for Biomass Estimation
1. ABA
2. ITD

-> next step: rationale between the different vegetation/forest type suitability which which approach ABA/ITD
https://www.tandfonline.com/doi/pdf/10.1080/17538947.2026.2723552
https://www.sciencedirect.com/science/article/pii/S1470160X25005400
'''
=======
    parser = argparse.ArgumentParser(
        description="Dune LiDAR ABA, vegetation and multi-epoch pipeline."
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="data_drone/20261007_north-F001-S001_RD.laz",
        help="Input LAS/LAZ point cloud.",
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="out.laz",
        help="Output enriched LAS/LAZ file.",
    )

    parser.add_argument(
        "--compare-with",
        default=None,
        help="Optional second LAS/LAZ epoch for DTM/CHM change detection.",
    )

    parser.add_argument(
        "--grid-res",
        type=float,
        default=0.25,
        help="Raster cell size in metres (default: 0.25).",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.5,
        help="Voxel edge length in metres (default: 0.5).",
    )

    parser.add_argument(
        "--no-alpha",
        action="store_true",
        help="Skip alpha-shape volume calculations.",
    )

    args = parser.parse_args()

    main(
        input_las_path=args.input,
        output_las_path=args.output,
        grid_res=args.grid_res,
        voxel_size=args.voxel_size,
        compute_alpha=not args.no_alpha,
        compare_with=args.compare_with,
    )
>>>>>>> Stashed changes
