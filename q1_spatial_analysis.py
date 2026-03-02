import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import numpy as np

# Loading NCR Boundary
ncr = gpd.read_file(r"Delhi-Airshed-LandUse-AI-Audit\delhi_ncr_region.geojson")

print("Original CRS:", ncr.crs)

# Reprojecting to UTM
ncr_utm = ncr.to_crs(epsg=32644)

print("Reprojected CRS:", ncr_utm.crs)

# Creating 60x60 km Grid
# Get bounding box (in meters)
minx, miny, maxx, maxy = ncr_utm.total_bounds

print("Bounding Box (meters):")
print(minx, miny, maxx, maxy)

grid_size = 60000  # 60 km in meters

grid_cells = []

for x in np.arange(minx, maxx, grid_size):
    for y in np.arange(miny, maxy, grid_size):
        grid_cells.append(box(x, y, x + grid_size, y + grid_size))

grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:32644")

print("Total grid cells created:", len(grid))

# Keeping only grid cells that intersect NCR
grid = grid[grid.intersects(ncr_utm.union_all())]

print("Grid cells after intersection:", len(grid))

# ploting grgid overlay

fig, ax = plt.subplots(figsize=(8,8))

ncr_utm.plot(ax=ax, edgecolor='black', facecolor='none')
grid.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1)

plt.title("60x60 km Grid Overlay on Delhi-NCR")

plt.tight_layout()
plt.savefig("grid_overlay.png", dpi=300)
plt.close()

print("Grid overlay image saved as grid_overlay.png")