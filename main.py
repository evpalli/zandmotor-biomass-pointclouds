from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree, ConvexHull
import CSF
import scipy.interpolate

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

def predictotry_variables(coords, veg_idx, ground_idx):
    '''
    returns:
    - tree height 
    - crown diameter and density
        ConvexHull calculation. xy min-max addition and division with 2 is not applied
        as zandmotor vegetation seem to have sparse, not very evenly spread PC?
    - shape/texture (pointy, flat, dense, sparse) 
        compute_geometric_features -> handles this?
    - diameter at breast height
    - crown volume

    - CHM
    ? difficult to obtain tho for the formulas
    '''
    # 1. vegetation height
    veg_pts = coords[veg_idx]
    ground_pts = coords[ground_idx]

    # assess if most vegetation on zandmotor has flat DTM or bumpy DTM
    # if bumpy DTM take average, if flat maybe minimum is better?
    local_ground_z = ground_pts[:, 2].mean() # Average ground elevation under/near the tree
    tree_height = veg_pts[:, 2].max() - local_ground_z

    # 2. Crown Diameter (Spread across x & y dimensions) and density
    veg_xy = veg_pts[:, :2]
    convexhull = ConvexHull(veg_xy)
    convexhull_area = convexhull.volume
    crown_diameter = 2 * np.sqrt(convexhull_area / np.pi)
    
    total_points = len(veg_pts)
    crown_density = total_points / convexhull_area

    # 4. Crown volume (2d->3d convex hull?)
    veg_xyz = veg_pts[:, :3]
    convexhull_3d = ConvexHull(veg_xyz)
    convexhull_volume = convexhull_3d.volume

def CHM(coords, veg_idx, grid_resolution=0.1):

    # 1. Classify ground/non-ground points
    ground_idx, non_ground_idx = run_csf_ground_filter(coords)

    ground_pts = coords[ground_idx]
    veg_pts = coords[veg_idx]

    # 2. Grid extent
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    grid_x, grid_y = np.mgrid[
        x_min:x_max:grid_resolution,
        y_min:y_max:grid_resolution
    ]

    # 3. Interpolate ground points to create DTM
    dtm = scipy.interpolate.griddata(
        ground_pts[:, :2],
        ground_pts[:, 2],
        (grid_x, grid_y),nmethod='linear') 

    # 4. Interpolate vegetation points to create DSM
    dsm = scipy.interpolate.griddata(
        veg_pts[:, :2],
        veg_pts[:, 2],
        (grid_x, grid_y), method='linear')

    # 5. Fill missing values
    dtm = np.nan_to_num(dtm,nan=np.nanmin(ground_pts[:, 2]))

    dsm = np.nan_to_num(dsm,nan=dtm)

    # 6. Calculate CHM
    chm = dsm - dtm

    # Remove negative values
    chm[chm < 0] = 0

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

    # 5. Add custom scalar fields to the single file for CloudCompare
    print("Adding diagnostic scalar fields...")
    las.add_extra_dim(laspy.ExtraBytesParams(name="stage_csf", type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name="stage_refined", type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name="planarity", type=np.float32))

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

    # Ensure output directory exists and write
    Path(output_laz).parent.mkdir(parents=True, exist_ok=True)
    las.write(output_laz)
    print(f"Classified multi-stage file saved to: {output_laz}")


if __name__ == "__main__":
    input_file = "data/clips_drone/20240501_72220_72420_451650_451850.laz"
    output_file = "data/output/zandmotor_classified_csf_eigenvalues_03.laz"
    main(input_file, output_file)