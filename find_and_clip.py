"""Find which LAZ files cover an area, cut that area out and (optionally) thin it so the result is small.

Usage (PowerShell):
  python find_and_clip.py <folder with .laz files> <output folder> [xmin ymin xmax ymax] [--density 250]

Examples:
  python find_and_clip.py data\\20250327 data\\out_0327 72290 451410 72510 451640 --density 250
  python find_and_clip.py data\\20261007 data\\out            (default area = the field box)

--density N   randomly thins the cut-out to about N points per m2 (250 is what the analysis uses anyway),
              which makes the files 2-4 times smaller. Leave it out to keep every point.
All coordinates are RD New (metres).
"""
import sys
from pathlib import Path

import laspy
import numpy as np

args = [a for a in sys.argv[1:]]
density = None
if "--density" in args:
    i = args.index("--density"); density = float(args[i + 1]); del args[i:i + 2]
if len(args) < 2:
    raise SystemExit(__doc__)
src, out = Path(args[0]), Path(args[1])
BOX = tuple(float(v) for v in args[2:6]) if len(args) >= 6 else (72450, 451890, 72650, 452090)
x0, y0, x1, y1 = BOX            # note the order: xmin ymin xmax ymax
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".laz", ".las"))
print(f"Looking in: {src.resolve()}  (exists: {src.exists()})")
print(f"Area: x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}   |   found {len(files)} laz/las files")
if not files:
    print("No files found - check the folder path (run:  dir <folder>)")

for f in files:
    with laspy.open(f) as r:
        h = r.header
        hit = not (h.maxs[0] < x0 or h.mins[0] > x1 or h.maxs[1] < y0 or h.mins[1] > y1)
        print(f"{f.name}: x {h.mins[0]:.0f}-{h.maxs[0]:.0f}  y {h.mins[1]:.0f}-{h.maxs[1]:.0f}  ->", "OVERLAPS" if hit else "no")
        if not hit:
            continue
        keep = 1.0
        if density:
            keep = min(1.0, density / (h.point_count / max((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1]), 1.0)))
        parts = []
        for pts in r.chunk_iterator(5_000_000):
            m = (pts.x >= x0) & (pts.x <= x1) & (pts.y >= y0) & (pts.y <= y1)
            if keep < 1.0:
                m &= rng.random(len(m)) < keep
            if m.any():
                parts.append(pts[m].array)
        if parts:
            new = laspy.LasData(h)
            new.points = laspy.ScaleAwarePointRecord(np.concatenate(parts), h.point_format, h.scales, h.offsets)
            new.write(out / f"{f.stem}_clip.laz")
            print(f"   saved {sum(len(p) for p in parts):,} points")
