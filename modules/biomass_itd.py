"""
MODULE: biomass_itd.py
PURPOSE: Individual Tussock Detection (ITD) and 3D biomass/volume comparison.
ASSIGNMENT QUESTION: Research Question (Biomass Estimation vs. Dune Slope)
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull, QhullError
# from sklearn.cluster import DBSCAN
import numpy as np
import alphashape
from collections import deque

def voxel_3d(las, voxel_size=0.5):
    """
    Initializes 3D voxel grid, separates occupied vs empty space,
    and runs 16-connectivity BFS to extract connected components.
    """
    x = las.x
    y = las.y
    z = las.z

    points = np.column_stack((x, y, z))
    
    # Assign points to grid/voxel units
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Find unique occupied voxels 
    unique_voxels = np.unique(voxel_indices, axis=0)
    voxel_set = {tuple(v) for v in unique_voxels}
    
    # 16-connectivity (6 face neighbors + 10 edge neighbors)
    face_neighbors = [
        (1, 0, 0), (-1, 0, 0), 
        (0, 1, 0), (0, -1, 0), 
        (0, 0, 1), (0, 0, -1)]
    edge_neighbors = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]
    connectivity_16 = face_neighbors + edge_neighbors 

    # Global Connected Components Tracking
    global_visited = set()
    components = {}
    component_id = 0
    veg_id_volume = {}
    cluster_points_dict = {}

    for voxel in voxel_set:
        if voxel not in global_visited:
            component_id += 1
            # Run BFS starting from this unvisited seed voxel
            cluster = bfs(connectivity_16, voxel, voxel_set, global_visited)
            
            # Map all voxels found in the component cluster to their ID
            for v in cluster:
                components[v] = component_id
            
            # Occupied voxel volume for this cluster
            veg_id_volume[component_id] = (len(cluster) * (voxel_size ** 3))
            # Calculate discrete voxel volume for this specific cluster
            cluster_occupied = len(cluster)
            volume_veg = cluster_occupied * (voxel_size ** 3)
            veg_id_volume[component_id] = volume_veg
    # add point to trace back the points/indexing for volume computations
    point_cluster_ids = np.array([components[tuple(v)] for v in voxel_indices], dtype=np.int32)
    total_volume = len(voxel_set) * (voxel_size ** 3)

    return voxel_set, voxel_indices, components, component_id, veg_id_volume, total_volume, points, point_cluster_ids

def bfs(neighbour_offsets, root, voxel_set, global_visited):
    queue = deque([root])
    seen = {root}
    global_visited.add(root)

    while queue:
        cn = queue.popleft()
        
        # Check all 16 directions by adding offsets to the current coordinate
        for dx, dy, dz in neighbour_offsets:
            nn = (cn[0] + dx, cn[1] + dy, cn[2] + dz)
            
            # If neighbor exists in our voxel data and hasn't been claimed yet
            if nn in voxel_set and nn not in seen:
                seen.add(nn)
                global_visited.add(nn)
                queue.append(nn)
    return seen

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
    
def segment_tussocks():
    """
    groups discrete pioneer grass clumps using DBSCAN spatial clustering.
    returns cluster labels? (-1 indicates unclustered noise points).
    """
    pass


def convex_hull_vol(cluster_coords):
    """Method 1: Solid bounding polyhedral volume via 3D Convex Hull."""
    # at least 4 to avoid QHull raise error
    if len(cluster_coords) < 4:
        return 0.0
    hull = ConvexHull(cluster_coords)
    convexhull_vol = float(hull.volume)
    return convexhull_vol
    
def alpha_shape_vol(cluster_coords, alpha=0.5):
    """
    Calculate 3D alpha-shape volume.
    Return NaN when a valid 3D volume cannot be calculated.
    """

    cluster_coords = np.asarray(cluster_coords, dtype=float)

    # Remove invalid and duplicate points
    cluster_coords = cluster_coords[np.all(np.isfinite(cluster_coords), axis=1)]
    cluster_coords = np.unique(cluster_coords, axis=0)

    # A 3D shape needs at least four unique points
    if len(cluster_coords) < 4:
        return np.nan

    # Check whether the points span three dimensions
    centered = cluster_coords - cluster_coords.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 3:
        return np.nan

    try:
        shape = alphashape.alphashape(cluster_coords, alpha)

        # Check that a valid 3D mesh was generated
        if not hasattr(shape, "is_watertight"):
            return np.nan

        if not shape.is_watertight:
            return np.nan

        volume = float(shape.volume)

        if not np.isfinite(volume) or volume < 0:
            return np.nan

        return volume

    except (QhullError, ValueError, IndexError, ZeroDivisionError):
        return np.nan

def extract_tussock_dataframe():
    """
    Compiles a per-tussock summary table containing centroid coordinates,
    height relative to local ground, the three competing volume estimates, and dune slope.
    """
    pass