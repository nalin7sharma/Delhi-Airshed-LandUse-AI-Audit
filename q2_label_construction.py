import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rasterio.warp import transform

# Load filtered image list
df = pd.read_csv("filtered_images.csv")

print("Total filtered images:", len(df))


# Open WorldCover raster
raster_path = "worldcover_bbox_delhi_ncr_2021.tif"
src = rasterio.open(raster_path)

labels = []


# Extract 128x128 patch
for index, row in df.iterrows():

    lat = row["latitude"]
    lon = row["longitude"]

    # Convert lat/lon (EPSG:4326) to raster CRS
    x, y = transform("EPSG:4326", src.crs, [lon], [lat])
    x = x[0]
    y = y[0]

    # Get pixel location
    col, row_pixel = src.index(x, y)

    # Create 128x128 window centered at pixel
    window = rasterio.windows.Window(
        col_off=col - 64,
        row_off=row_pixel - 64,
        width=128,
        height=128
    )

    patch = src.read(1, window=window)

    # Flatten and compute dominant class
    values = patch.flatten()
    mode = np.bincount(values).argmax()

    labels.append(mode)

# Add ESA code column
df["esa_code"] = labels


# Map ESA codes to simple labels

def map_label(code):
    if code == 50:
        return "Built-up"
    elif code == 40:
        return "Cropland"
    elif code == 80:
        return "Water"
    elif code in [10, 20, 30]:
        return "Vegetation"
    else:
        return "Others"

df["label"] = df["esa_code"].apply(map_label)

# Save labelled dataset
df.to_csv("labelled_images.csv", index=False)

print("Labels assigned successfully!")


# 60/40 Train-Test Split
train, test = train_test_split(
    df,
    test_size=0.4,
    random_state=42
)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("Train size:", len(train))
print("Test size:", len(test))


# Visualize class distribution
df["label"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

print("Class distribution saved!")