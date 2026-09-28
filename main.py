from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree, ConvexHull
import CSF
import gc
import scipy.interpolate

# need pip install CSF
'''
Identify type of vegetation exists on Zandmotor, focus on the specific geometry to 
narrow down specific vegetation type
Most vegetation type: sand dunes, european marram grass. types of methods
    - above-ground biomass (AGB)

Do we want individual tree crown approach (ITC) or area-based
1. Delineate individual tree crowns using watershed segmentation
2. Calc. predictory variables for all individual trees
    - height (highest point of tree), crown (horizontal space tree top cover), crown volume, shape/tecture (pointy, flat, dense, sparse)
3. Training data collection
4. RF rgeression model to estimate biomass
link: https://www.neonscience.org/resources/learning-hub/tutorials/calc-biomass-py
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

def predictory_variables(coords, veg_idx, ground_idx, radius=0.5, min_neighbors=8):
    '''
    returns:
    - tree height 
    - point density
        ConvexHull calculation. xy min-max addition and division with 2 is not applied
        as zandmotor vegetation seem to have sparse, not very evenly spread PC?
    - shape/texture (pointy, flat, dense, sparse) 
        compute_geometric_features -> handles this?
    - crown volume

    - CHM
    ? difficult to obtain tho for the formulas
    '''
    veg_kdtree = cKDTree(coords[veg_idx])
    ground_kdtree = cKDTree(coords[ground_idx])

    veg_pts = coords[veg_idx]
    ground_pts = coords[ground_idx]
    n_veg = len(veg_pts)

    tree_height = np.zeros(n_veg, dtype=np.float32)
    crown_diameter = np.zeros(n_veg, dtype=np.float32)
    crown_density = np.zeros(n_veg, dtype=np.float32)
    convexhull_volume = np.zeros(n_veg, dtype=np.float32)

    for i, pt in enumerate(veg_pts):
        veg_neighbor_idx = veg_kdtree.query_ball_point(pt, r=radius)
        
        if len(veg_neighbor_idx) >= min_neighbors:
            cluster_veg = veg_pts[veg_neighbor_idx]

            # Local Height relative to nearby ground points
            ground_neighbor_idx = ground_kdtree.query_ball_point(pt, r=radius * 2.0)
            if len(ground_neighbor_idx) > 0:
                local_ground_z = np.mean(ground_pts[ground_neighbor_idx, 2])
            else:
                local_ground_z = pt[2] # Fallback to current z if no ground points nearby
                
            tree_height[i] = np.max(cluster_veg[:, 2]) - local_ground_z

            # Spread across x & y dimensions
            cluster_xy = cluster_veg[:, :2]
            if len(np.unique(cluster_xy, axis=0)) >= 3:
                try:
                    hull_2d = ConvexHull(cluster_xy)
                    convexhull_area = hull_2d.volume  # In 2D, volume represents area
                    if convexhull_area > 0:
                        crown_diameter[i] = 2.0 * np.sqrt(convexhull_area / np.pi)
                        crown_density[i] = len(cluster_veg) / convexhull_area
                except Exception:
                    pass  # Coplanar or collinear subset safe-guard

            # Crown Volume
            if len(np.unique(cluster_veg, axis=0)) >= 4:
                try:
                    hull_3d = ConvexHull(cluster_veg)
                    convexhull_volume[i] = hull_3d.volume
                except Exception:
                    pass

    return tree_height, crown_diameter, crown_density, convexhull_volume

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
    input_file = "/Users/belinaileen/Desktop/zandmotor-biomass-pointclouds/20230927_72220_72420_451650_451850.laz"
    output_file = "/Users/belinaileen/Desktop/zandmotor-biomass-pointclouds/out.laz"
    main(input_file, output_file)