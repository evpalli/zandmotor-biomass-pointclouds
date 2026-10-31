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

def voxel_3d(las, connectivity, voxel_size=0.5):
    """
    Initializes a 3D voxel grid and extracts connected components.

    Parameters
    ----------
    las : object
        Object containing x, y, z coordinates.
    voxel_size : float
        Side length of each voxel.
    connectivity : int
        Connectivity criterion: 6, 18, or 26.

        6  = face neighbors only
        18 = face and edge neighbors
        26 = face, edge, and corner neighbors

    Returns
    -------
    voxel_set, voxel_indices, components, component_id,
    veg_id_volume, total_volume, points, point_cluster_ids
    """

    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")

    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than 0")

    x = las.x
    y = las.y
    z = las.z

    points = np.column_stack((x, y, z))

    # Assign points to voxel grid coordinates
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Find unique occupied voxels
    unique_voxels = np.unique(voxel_indices, axis=0)
    voxel_set = {tuple(v) for v in unique_voxels}

    # Face neighbors: 6-connectivity
    face_neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    # Edge neighbors: 12 additional neighbors
    edge_neighbors = [
        (1, 1, 0), (1, -1, 0),
        (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1),
        (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1),
        (0, -1, 1), (0, -1, -1)
    ]

    # Corner neighbors: 8 additional neighbors
    corner_neighbors = [
        (dx, dy, dz)
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (-1, 1)
    ]

    # Select neighborhood based on connectivity
    if connectivity == 6:
        neighbors = face_neighbors
    elif connectivity == 18:
        neighbors = face_neighbors + edge_neighbors
    else:  # connectivity == 26
        neighbors = face_neighbors + edge_neighbors + corner_neighbors

    # Connected component tracking
    global_visited = set()
    components = {}
    component_id = 0
    veg_id_volume = {}

    for voxel in voxel_set:
        if voxel not in global_visited:
            component_id += 1

            # Find all voxels connected to this seed
            cluster = bfs(
                neighbors,
                voxel,
                voxel_set,
                global_visited
            )

            # Assign component IDs
            for v in cluster:
                components[v] = component_id

            # Calculate occupied voxel volume for this component
            veg_id_volume[component_id] = (
                len(cluster) * (voxel_size ** 3)
            )

    # Map each original point to its component ID
    point_cluster_ids = np.array(
        [components[tuple(v)] for v in voxel_indices],
        dtype=np.int32
    )

    # Total occupied voxel volume
    total_volume = len(voxel_set) * (voxel_size ** 3)

    return (
        voxel_set,
        voxel_indices,
        components,
        component_id,
        veg_id_volume,
        total_volume,
        points,
        point_cluster_ids
    )

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

def compute_geometric_features(cluster_coords):
    """Refinement computing eigenvalue descriptors (Kucak, 2022)."""
    coords = np.asarray(cluster_coords, dtype=float)

    # Require at least 3 points
    if len(coords) < 3:
        return np.nan, np.nan, np.nan

    # Calculate covariance matrix using all points in the cluster
    cov = np.cov(coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov) # Calculate eigenvalues
    eigenvalues = np.sort(eigenvalues)[::-1] # Sort from largest to smallest
    l1, l2, l3 = eigenvalues
    if l1 <= 0: # Avoid division by zero
        return np.nan, np.nan, np.nan

    # Calculate geometric features
    planarity = (l2 - l3) / l1
    linearity = (l1 - l2) / l1
    sphericity = l3 / l1

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
        return 1001

    # Check whether the points span three dimensions
    centered = cluster_coords - cluster_coords.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 3:
        return 1002

    try:
        shape = alphashape.alphashape(cluster_coords, alpha)

        # Check that a valid 3D mesh was generated
        if not hasattr(shape, "is_watertight"):
            return 1003

        if not shape.is_watertight:
            return 1003

        volume = float(shape.volume)

        if not np.isfinite(volume) or volume < 0:
            return 1004

        return volume

    except (QhullError, ValueError, IndexError, ZeroDivisionError):
        return 1005

def extract_tussock_dataframe():
    """
    Compiles a per-tussock summary table containing centroid coordinates,
    height relative to local ground, the three competing volume estimates, and dune slope.
    """
    pass

def export_voxels_to_obj(voxel_set, voxel_size=0.5, output_path="voxels.obj"):
    """
    Export occupied 3D voxels as cube geometry in an OBJ file.

    Parameters
    ----------
    voxel_set : set
        Set of occupied voxel coordinates (i, j, k) returned by voxel_3d().
    voxel_size : float
        Side length of each voxel.
    output_path : str
        Path for the output OBJ file.
    """
    # Eight vertices of a unit cube
    cube_vertices = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]

    # Six faces, using 1-based vertex indices
    cube_faces = [
        (1, 4, 3, 2),  # Bottom
        (5, 6, 7, 8),  # Top
        (1, 2, 6, 5),  # Front
        (2, 3, 7, 6),  # Right
        (3, 4, 8, 7),  # Back
        (4, 1, 5, 8),  # Left
    ]

    with open(output_path, "w") as obj:
        obj.write("# Occupied voxel geometry\n")

        vertex_offset = 0

        for voxel in sorted(voxel_set):
            i, j, k = voxel

            # Convert voxel indices to physical coordinates
            x0 = i * voxel_size
            y0 = j * voxel_size
            z0 = k * voxel_size

            # Write the eight vertices of this voxel
            for vx, vy, vz in cube_vertices:
                x = x0 + vx * voxel_size
                y = y0 + vy * voxel_size
                z = z0 + vz * voxel_size

                obj.write(f"v {x} {y} {z}\n")

            # Write the six cube faces
            for face in cube_faces:
                indices = [vertex_offset + idx for idx in face]
                obj.write("f " + " ".join(map(str, indices)) + "\n")

            vertex_offset += 8

    print(f"Exported {len(voxel_set)} voxels to {output_path}")