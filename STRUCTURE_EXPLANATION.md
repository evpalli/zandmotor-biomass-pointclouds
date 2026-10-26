# Pipeline Architecture & Assignment Mapping (TU Delft AESM5230C)

This document explains the organization of the codebase, detailing what each module does, why it is structured this way, and which graded assignment question (or Research Question) it satisfies.

---

## 1. Directory Structure

```text
biomass/
├── modules/                   # Reusable mathematical & processing scripts
│   ├── __init__.py            # Package initialization
│   ├── ground_filter.py       # Macro/micro ground separation (Q A4)
│   ├── terrain_aba.py         # 2.5D Raster DTM/CHM, slope, and multi-epoch differences (Q A1, A3, RQ)
│   ├── biomass_itd.py         # Tussock clustering & 3-method volume calculation (Research Question)
│   └── spectral_typing.py     # VARI greenness & species structural discrimination (Q A2)
├── notebooks/                 # Execution, visualisation, and report figure generation
│   ├── 01_explore_and_typing.ipynb   # Answers A1 & A2
│   ├── 02_spatiotemporal_A3.ipynb    # Answers A3
│   └── 03_biomass_vs_slope_RQ.ipynb  # Answers the Biomass vs. Slope Research Question
├── data/                      # Local datasets (Git-ignored)
│   ├── clips_drone/           # Raw flightline clips (.laz)
│   ├── raw/                   # Working subsets / uncompressed laz
│   └── output/                # Classified .laz, rasters, and summary CSV tables
├── STRUCTURE_EXPLANATION.md   # This documentation
├── requirements.txt           # Virtual environment libraries
└── .gitignore                 # Excludes heavy LiDAR binaries from GitHub

```
#### Pipeline Summary

1. **Ground filtering:** Separate ground and vegetation points using geometric features.
2. **Terrain modelling:** Generate DTM, CHM, and slope rasters.
3. **Terrain attribute calculation:** Calculate height above ground and point-level slope.
4. **Spectral analysis:** Calculate VARI from RGB values.
5. **Voxelisation:** Convert vegetation points into 3D voxels.
6. **Individual vegetation detection:** Identify connected voxel components and assign cluster IDs.
7. **Geometric feature extraction:** Calculate planarity, linearity, and sphericity.
8. **Cluster aggregation:** Calculate cluster-level spatial and attribute statistics.
9. **Volume estimation:** Estimate vegetation volume using voxels, convex hulls, and alpha shapes.
10. **Species classification:** Assign pioneer species labels using height, linearity, and VARI.
11. **Attribute propagation:** Transfer cluster attributes back to individual vegetation points.
12. **Export:** Save the enriched point cloud and cluster-level CSV table.

**Overall objective:** Transform a raw dune LiDAR point cloud into an enriched 3D vegetation dataset with individual vegetation clusters, terrain-relative height, geometric and spectral characteristics, estimated volumes, and pioneer species classifications.