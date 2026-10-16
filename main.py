from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree, ConvexHull
import CSF
import gc
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist

def initialise_grid_ABA(las):

    # Extract X, Y, and Z coordinates
    x = las.x
    y = las.y
    z = las.z

    points = np.column_stack((x, y))
    values = z

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


if __name__ == "__main__":
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