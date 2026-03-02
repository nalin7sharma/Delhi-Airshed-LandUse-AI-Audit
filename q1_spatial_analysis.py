import geopandas as gpd
import os
import pandas as pd
from shapely.geometry import Point

# ---------------------------
# LOAD NCR FIRST
# ---------------------------
ncr = gpd.read_file("Delhi-Airshed-LandUse-AI-Audit\\delhi_ncr_region.geojson")

# ---------------------------
# IMAGE FILTERING
# ---------------------------
image_folder = "Delhi-Airshed-LandUse-AI-Audit\\rgb"
image_files = os.listdir(image_folder)

print("Total images before filtering:", len(image_files))

# Compute union ONCE
ncr_polygon = ncr.union_all()

filtered_images = []

for img in image_files:
    
    name = img.replace(".png", "")
    
    try:
        lat, lon = name.split("_")
        lat = float(lat)
        lon = float(lon)
    except:
        continue

    point = Point(lon, lat)

    if point.within(ncr_polygon):
        filtered_images.append([img, lat, lon])

print("Filtered images count:", len(filtered_images))

df_filtered = pd.DataFrame(filtered_images, columns=["filename", "latitude", "longitude"])
df_filtered.to_csv("filtered_images.csv", index=False)

print("filtered_images.csv created successfully!")