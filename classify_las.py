
import laspy
import numpy as np
import subprocess
import struct

input_file = "/Users/belinaileen/Desktop/test_set.laz"
output_file = "/Users/belinaileen/Desktop/test_classified.laz"

# Read LAS
las = laspy.read(input_file)

x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)
z = np.asarray(las.z, dtype=np.float64)

npoints = len(x)
print(f"Loaded {npoints:,} points")

# Calculate bounding box automatically
bbox = (
    x.min(), y.min(), z.min(),
    x.max(), y.max(), z.max()
)

# Write binary coordinates
with open("coordinates.bin", "wb") as f:
    f.write(struct.pack("I", npoints))
    x.tofile(f)
    y.tofile(f)
    z.tofile(f)

# Write binary bounding box in PTD::Bbox field order
with open("bbox.bin", "wb") as f:
    f.write(struct.pack("6d", *bbox))

# Run C++ PTD
subprocess.run([
    "./PTD/build/ptd_run",
    "coordinates.bin",
    "bbox.bin",
    "ground_indices.txt"
], check=True)

# Read ground indices
ground = np.atleast_1d(
    np.loadtxt("ground_indices.txt", dtype=np.uint32)
)

if ground.size == 0:
    raise RuntimeError("No ground points were returned.")

if np.any(ground >= npoints):
    raise RuntimeError("Invalid ground point indices.")

# Assign LAS classifications
las.classification[:] = 1
las.classification[ground] = 2

# Save final LAS
las.write(output_file)

print("Classification completed.")
print(f"Ground points: {len(ground):,}")
print(f"Non-ground points: {npoints - len(ground):,}")
print(f"Output saved: {output_file}")
