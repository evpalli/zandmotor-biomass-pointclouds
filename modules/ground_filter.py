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

def csf(
    coords: np.ndarray,
    resolution: float = 0.6,
    rigidness: int = 2,
    threshold: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverts the point cloud and drapes a virtual cloth to separate broad dune topography
    from above-ground candidate vegetation.
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


def eigenvalues(
    coords: np.ndarray,
    candidate_idx: np.ndarray,
    radius: float = 0.20,
    min_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts 3D covariance eigenvalues (lambda_1 >= lambda_2 >= lambda_3) within a search radius
    to derive structural descriptors: Planarity, Linearity, and Sphericity.
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


def classify_ground_and_vegetation(
    input_laz_path: str,
    output_laz_path: str | None = None,
    planarity_threshold: float = 0.65,
    linearity_threshold: float = 0.30,
    sphericity_threshold: float = 0.15,
) -> dict:
    """Executes the two-stage classification workflow and exports a single diagnostic

    .laz containing ASPRS classifications and CloudCompare scalar fields.
    """
    las = laspy.read(input_laz_path)
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    total_points = len(coords)

    # Stage 1: Macro-ground filtering (CSF)
    ground_idx, candidate_veg_idx = csf(coords)

    # Stage 2: Micro-refinement via geometric eigenvalues
    planarity, linearity, sphericity = eigenvalues(
        coords, candidate_veg_idx
    )

    # Sand residual decision rule
    is_true_veg = (planarity < planarity_threshold) & (
        (linearity > linearity_threshold) | (sphericity > sphericity_threshold)
    )
    final_veg_idx = candidate_veg_idx[is_true_veg]
    residual_sand_idx = candidate_veg_idx[~is_true_veg]
    final_ground_idx = np.concatenate((ground_idx, residual_sand_idx))

    # Calculate statistics
    veg_ratio = (
        (len(final_veg_idx) / total_points) * 100.0 if total_points > 0 else 0.0
    )

    # Save output LAZ with extra diagnostic scalar fields
    if output_laz_path is not None:
        Path(output_laz_path).parent.mkdir(parents=True, exist_ok=True)

        # Create header using las.header.version (fixes AttributeError)
        header = laspy.LasHeader(
            point_format=las.point_format, version=las.header.version
        )
        header.offsets = las.header.offsets
        header.scales = las.header.scales

        # Add diagnostic scalar fields for CloudCompare inspection
        header.add_extra_dim(
            laspy.ExtraBytesParams(name="stage_csf", type=np.uint8)
        )
        header.add_extra_dim(
            laspy.ExtraBytesParams(name="stage_refined", type=np.uint8)
        )
        header.add_extra_dim(
            laspy.ExtraBytesParams(name="planarity", type=np.float32)
        )

        out_las = laspy.LasData(header)
        out_las.x = las.x
        out_las.y = las.y
        out_las.z = las.z

        # Preserve RGB if present
        if hasattr(las, "red"):
            out_las.red = las.red
            out_las.green = las.green
            out_las.blue = las.blue

        # ASPRS Standard Classification: 2 = Ground, 4 = Medium/Low Vegetation
        out_las.classification[:] = 0
        out_las.classification[final_ground_idx] = 2
        out_las.classification[final_veg_idx] = 4

        # Populate scalar fields
        stage1 = np.zeros(total_points, dtype=np.uint8)
        stage1[candidate_veg_idx] = 1
        out_las.stage_csf = stage1

        stage2 = np.zeros(total_points, dtype=np.uint8)
        stage2[final_veg_idx] = 1
        out_las.stage_refined = stage2

        full_planarity = np.zeros(total_points, dtype=np.float32)
        full_planarity[candidate_veg_idx] = planarity
        out_las.planarity = full_planarity

        out_las.write(output_laz_path)

    return {
        "coords": coords,
        "ground_idx": final_ground_idx,
        "veg_idx": final_veg_idx,
        "veg_percentage": veg_ratio,
        "total_points": total_points,
        "las": las,
    }