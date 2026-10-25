"""
main.py - full pipeline: ground filtering -> rasters -> vegetation clusters ->
spectral typing -> volumes -> outputs.

Outputs (all next to the output LAZ, sharing its name):
  out.laz                 point cloud with every per-point result as an extra dimension
  out_tussocks.csv        one row per vegetation cluster (position, height, volumes, species, slope)
  out_rasters.npz         DTM, CHM and slope rasters + grid info (always written)
  out_rasters.tif         same rasters as a 3-band GeoTIFF (only if `rasterio` is installed)
"""

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import rasterio
import laspy
import numpy as np
import pandas as pd

from modules.biomass_itd import (
    alpha_shape_vol,
    compute_geometric_features,
    convex_hull_vol,
    voxel_3d
)
from modules.ground_filter import classify_ground_and_vegetation
from modules.spectral_typing import classify_pioneer_species, compute_vari
from modules.terrain_aba import rasters, sample_raster

# Labels come from spectral_typing.py.
SPECIES_CODES = {"Dry Drift / Senescent Matter": 1,
                "Ammophila arenaria (Marram Grass)": 2,
                "Elytrigia juncea (Sand Couch)": 3,
                "Unclassified (missing RGB or too few points)": 4}


# helpers
def has_rgb(las: laspy.LasData) -> bool:
    return {"red", "green", "blue"} <= set(las.point_format.dimension_names)

def write_dim(las: laspy.LasData, name: str, values: np.ndarray, dtype):
    """Adds (or overwrites) an extra dimension on the LAS object."""
    if name not in las.point_format.dimension_names:  # add new empty column
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    setattr(las, name, np.asarray(values).astype(dtype)) # add value on the newly added column

def save_rasters(prefix: Path, dtm, chm, slope_deg, bounds, las):
    """Saves the 2D rasters. LAZ cannot hold rasters -> GeoTIFF."""

    crs = las.header.parse_crs()
    crs_wkt = crs.to_wkt() if crs is not None else None

    res = bounds["grid_res"]
    top = bounds["y_min"] + bounds["nrows"] * res
    tif_path = prefix.parent / f"{prefix.name}_rasters.tif"
    with rasterio.open(
        tif_path, "w", driver="GTiff", height=bounds["nrows"], width=bounds["ncols"], count=3,
        dtype="float32", crs=crs_wkt,nodata=np.nan) as dst:
        # Our arrays start at the southern edge; GeoTIFF starts at the northern edge -> flip
        for band, (arr, desc) in enumerate(((dtm, "DTM"), (chm, "CHM"), (slope_deg, "slope_deg")), 1):
            dst.write(np.flipud(arr).astype(np.float32), band)
            dst.set_band_description(band, desc)
    print(f" GeoTIFF saved to {tif_path}")

# pipeline start 
def main(input_las_path: str, output_las_path: str, grid_res: float = 0.10,
         voxel_size: float = 0.5, compute_alpha: bool = True):
    out_path = Path(output_las_path)
    prefix = out_path.with_suffix("")  

    las = laspy.read(input_las_path)
    coords = np.column_stack((las.x, las.y, las.z))
    n_pts = len(coords)

    # 1. Ground filtering (CSF)
    print(" start ground_filter.py")
    final_ground_idx, final_veg_idx = classify_ground_and_vegetation(las, planarity_thresh=0.65, linearity_thresh=0.30, sphericity_thresh=0.15)

    #  2. Rasters -> CHM/DTM/DSM/Slope calculation on 2D grid -> already differentiation between veg and ground idx
    print("start DTM / CHM / slope rasters (terrain_aba.py)")
    dtm, chm, slope_deg, bounds = rasters(coords, final_ground_idx, final_veg_idx, grid_res=grid_res)
    print("saving rasters (terrain_aba.py)")
    save_rasters(prefix, dtm, chm, slope_deg, bounds, las)
    print("calculate height above ground (terrain_aba.py)")
    ground_z = sample_raster(dtm, bounds, coords[:, 0], coords[:, 1])
    height_above_ground = (coords[:, 2] - ground_z).astype(np.float32)
    slope_pt = sample_raster(slope_deg, bounds, coords[:, 0], coords[:, 1]).astype(np.float32)
    print("write DIM (terrain_aba.py)")
    write_dim(las, "height_above_ground", height_above_ground, np.float32)
    write_dim(las, "slope_deg", slope_pt, np.float32)

    # 3. VARI Spectral Typing for greenness inspection
    print("start spectral_typing.py")
    vari = compute_vari(np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue))
    write_dim(las, "vari", vari, np.float32)

    print("start biomass_itd.py")
    tussock_id = np.full(n_pts, 0, np.uint32)
    species_class = np.full(n_pts, 0, np.uint8)
    veg_planarity = np.full(n_pts, np.nan, np.float32)
    veg_linearity = np.full(n_pts, np.nan, np.float32)
    veg_sphericity = np.full(n_pts, np.nan, np.float32)
    vol_voxel = np.full(n_pts, np.nan, np.float32)
    vol_hull = np.full(n_pts, np.nan, np.float32)
    vol_alpha = np.full(n_pts, np.nan, np.float32)
    table = pd.DataFrame()

    veg_coords = coords[final_veg_idx]

    # 4a. Voxel connected components init. (voxel_3d only needs .x .y .z)
    veg_xyz = SimpleNamespace(x=veg_coords[:, 0], y=veg_coords[:, 1], z=veg_coords[:, 2])
    voxel_set, voxel_indices, components, component_id, veg_id_volume, total_volume, points, point_cluster_ids = voxel_3d(veg_xyz, voxel_size=voxel_size)

    # 4b. 3D voxel indexing -> cluster ID 
    # -> 2 uses: clusted ID to identify voxels with vegetation, and biomass estimation via voxels
    uniq, inv = np.unique(voxel_indices, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    uniq_ids = np.fromiter((components[tuple(u)] for u in uniq), dtype=np.int64, count=len(uniq))
    veg_cid = uniq_ids[inv]

    # 5. planarity, sphericity, and linearity initialisation using veg. pts
    planarity, linearity, sphericity = compute_geometric_features(veg_coords, np.arange(len(veg_coords)), radius=0.20, min_neighbors=5)
    valid = (planarity + linearity + sphericity) > 0.5
    planarity = np.where(valid, planarity, np.nan)
    linearity = np.where(valid, linearity, np.nan)
    sphericity = np.where(valid, sphericity, np.nan)

    # 6a. initialise dataframa with columns
    pts = pd.DataFrame({
        "cluster_id": veg_cid,
        "x": veg_coords[:, 0], "y": veg_coords[:, 1], "z": veg_coords[:, 2],
        "height_above_ground": height_above_ground[final_veg_idx], "ground_z": ground_z[final_veg_idx],
        "slope_deg": slope_pt[final_veg_idx],
        "planarity": planarity, "linearity": linearity, "sphericity": sphericity,
        "vari": vari[final_veg_idx] if vari is not None else np.nan})
    # 6b. initialise table of points grouped by cluster 
    # (cluster represent group of voxel representing vegetation)
    table = pts.groupby("cluster_id").agg(
        point_count=("x", "size"),
        centroid_x=("x", "mean"), centroid_y=("y", "mean"), centroid_z=("z", "mean"),
        ground_z=("ground_z", "mean"),
        height_above_ground_max=("height_above_ground", "max"), height_above_ground_mean=("height_above_ground", "mean"),
        slope_deg_mean=("slope_deg", "mean"), slope_deg_median=("slope_deg", "median"), # i don't know to use median/mean here 
        planarity_median=("planarity", "median"),
        linearity_median=("linearity", "median"),
        sphericity_median=("sphericity", "median"),
        vari_median=("vari", "median"))

    # 7. Volumes: voxel (Method 1), convex hull (Method 2), alpha shape (Method 3)
    '''
    calculate alpha shape and convex hull via existing clusters
    '''
    voxels_by_cluster = defaultdict(list)
    hull_vols, alpha_vols = {}, {}
    for v, cid in components.items():
        voxels_by_cluster[cid].append(v)
    # table["volume_voxel_m3"] = [voxel_vol(voxels_by_cluster[c], voxel_size) for c in table.index]
    table["volume_voxel_m3"] = veg_id_volume
    table["volume_convex_hull_m3"] = [convex_hull_vol(points[point_cluster_ids == c])for c in table.index]
    table["volume_alpha_shape_m3"] = [alpha_shape_vol(points[point_cluster_ids == c]) for c in table.index]

    # 8. Spectral typing: Classify pioneer species
    labels = []
    for _, r in table.iterrows():
        if not np.isnan([r.height_max_m, r.linearity_median, r.vari_median]).any():
            labels.append(classify_pioneer_species(r.height_max_m, r.linearity_median, r.vari_median))
        else:
            labels.append("Unclassified (missing RGB or too few points)")
    table["species"] = labels
    table["species_code"] = table["species"].map(SPECIES_CODES).astype(np.uint8)
    for name, count in table["species"].value_counts().items():
        print(f"{name}: {count}")

    # 9.  save back each cluster to individual points
    def spread(col):
        return table[col].reindex(veg_cid).to_numpy()

    tussock_id[final_veg_idx] = veg_cid
    species_class[final_veg_idx] = spread("species_code")
    veg_planarity[final_veg_idx] = planarity
    veg_linearity[final_veg_idx] = linearity
    veg_sphericity[final_veg_idx] = sphericity
    vol_voxel[final_veg_idx] = spread("volume_voxel_m3")
    vol_hull[final_veg_idx] = spread("volume_convex_hull_m3")
    vol_alpha[final_veg_idx] = spread("volume_alpha_shape_m3")

    for name, arr, dt in (
        ("tussock_id", tussock_id, np.uint32), ("species_class", species_class, np.uint8),
        ("veg_planarity", veg_planarity, np.float32), ("veg_linearity", veg_linearity, np.float32),
        ("veg_sphericity", veg_sphericity, np.float32),
        ("vol_voxel_m3", vol_voxel, np.float32), ("vol_convex_hull_m3", vol_hull, np.float32),
        ("vol_alpha_shape_m3", vol_alpha, np.float32),
    ):
        write_dim(las, name, arr, dt)

    # Save 
    csv_path = prefix.parent / f"{prefix.name}_tussocks.csv"
    table.reset_index().to_csv(csv_path, index=False)
    print(f" Tussock table saved to: {csv_path}")
    print(f" Saving point cloud to: {out_path}")
    las.write(str(out_path))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dune LiDAR pipeline")
    ap.add_argument("input", nargs="?", default="data_drone/20261007_north-F001-S001_RD.laz")
    ap.add_argument("output", nargs="?", default="out.laz")
    ap.add_argument("--grid-res", type=float, default=0.25, help="raster cell size in metres")
    ap.add_argument("--voxel-size", type=float, default=0.5, help="voxel edge length in metres")
    ap.add_argument("--no-alpha", action="store_true", help="skip alpha-shape volumes (slow on big clusters)")
    a = ap.parse_args()
    main(a.input, a.output, a.grid_res, a.voxel_size, compute_alpha=not a.no_alpha)