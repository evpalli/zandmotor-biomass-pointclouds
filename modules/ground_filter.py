"""
MODULE: ground_filter.py
PURPOSE: Two-stage bare dune ground separation and terrain classification.
ASSIGNMENT QUESTION: A4 (Ground Filtering Methodology)
"""

from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree
import CSF

def csf(coords: np.ndarray, resolution: float = 0.6, rigidness: int = 2, threshold: float = 0.12):
    """
    Stage 1: CSF Macro-Ground Filtering (Zhang et al., 2016).
    Inverts point cloud and simulates an elastic cloth to separate undulating dune
    terrain from above-ground objects.
    """
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

    return np.array(ground_indices, dtype=np.int64), np.array(non_ground_indices, dtype=np.int64)


def eigenvalues(coords: np.ndarray, candidate_idx: np.ndarray, radius: float = 0.25, min_neighbors: int = 8):
    """
    Stage 2: Covariance Eigenvalue Geometric Features (Kucak, 2022).
    Computes 3D structural shape metrics: Planarity, Linearity, Sphericity.
    """
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


def classify_ground_and_vegetation(las: laspy.LasData, planarity_thresh: float = 0.65,
                                   linearity_thresh: float = 0.30, sphericity_thresh: float = 0.15):
    """
    Executes complete 2-stage workflow, assigns ASPRS standard classes:
      - Class 2: Ground
      - Class 4: Medium/Low Vegetation
    Attaches diagnostic scalar fields ('stage_csf', 'stage_refined', 'planarity') for CloudCompare.
    """
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    total_points = len(coords)

    # 1. Macro separation via CSF
    ground_idx, candidate_veg_idx = csf(coords)

    # 2. Micro-refinement via geometric eigenvalues
    planarity, linearity, sphericity = eigenvalues(coords, candidate_veg_idx)

    # 3. Filter remaining planar sand residuals
    is_true_veg = (planarity < planarity_thresh) & ((linearity > linearity_thresh) | (sphericity > sphericity_thresh))
    final_veg_idx = candidate_veg_idx[is_true_veg]
    residual_sand_idx = candidate_veg_idx[~is_true_veg]
    final_ground_idx = np.concatenate((ground_idx, residual_sand_idx))

    # 4. Standard ASPRS Classification
    las.classification[:] = 0
    las.classification[final_ground_idx] = 2
    las.classification[final_veg_idx] = 4

    # 5. Diagnostic scalar fields for CloudCompare inspection
    las.add_extra_dim(laspy.ExtraBytesParams(name="stage_csf", type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name="stage_refined", type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name="planarity", type=np.float32))

    stage1 = np.zeros(total_points, dtype=np.uint8)
    stage1[candidate_veg_idx] = 1
    las.stage_csf = stage1

    stage2 = np.zeros(total_points, dtype=np.uint8)
    stage2[final_veg_idx] = 1
    las.stage_refined = stage2

    full_planarity = np.zeros(total_points, dtype=np.float32)
    full_planarity[candidate_veg_idx] = planarity
    las.planarity = full_planarity

    return final_ground_idx, final_veg_idx