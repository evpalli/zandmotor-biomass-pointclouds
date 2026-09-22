from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree
import CSF


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