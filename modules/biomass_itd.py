"""
MODULE: biomass_itd.py
PURPOSE: Individual Tussock Detection (ITD) and 3D biomass/volume comparison.
ASSIGNMENT QUESTION: Research Question (Biomass Estimation vs. Dune Slope)
"""

# import numpy as np
# import pandas as pd
# from scipy.spatial import cKDTree, ConvexHull
# from sklearn.cluster import DBSCAN
import numpy as np
import alphashape
from scipy.spatial import ConvexHull
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
            
            # Calculate discrete voxel volume for this specific cluster
            cluster_occupied = len(cluster)
            volume_veg = cluster_occupied * (voxel_size ** 3)
            veg_id_volume[component_id] = volume_veg

    n_occupied = len(global_visited)
    total_volume = n_occupied * (voxel_size ** 3)

    return voxel_set, voxel_indices, components, component_id, veg_id_volume, total_volume

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
    convex_hull = ConvexHull(cluster_coords)
    convex_hull_vol = convex_hull.volume
    return convex_hull_vol

def voxel_vol(cluster_voxels, voxel_size=0.5):
    """
    Method 2: Discrete 3D voxel grid volume (V = N_occupied * r^3).
    Captures internal void space and porosity of Ammophila arenaria.
    """
    voxel_vol = len(cluster_voxels) * (voxel_size ** 3)
    return voxel_vol
    
def alpha_shape_vol(cluster_coords, alpha=0.5):
    """
    Method 3: Concave boundary mesh volume (Jang & Kang, 2024).
    Uses alphashape library
    """
    alphashape = alphashape.alphashape(cluster_coords, alpha)
    alphashape_volume = alphashape.volume
    return alphashape_volume

def extract_tussock_dataframe():
    """
    Compiles a per-tussock summary table containing centroid coordinates,
    height relative to local ground, the three competing volume estimates, and dune slope.
    """
    pass
