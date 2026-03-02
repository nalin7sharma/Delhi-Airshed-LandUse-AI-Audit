# Delhi Airshed Land-Use Classification

AI-based land-use classification over Delhi-NCR using Sentinel-2 imagery and ESA WorldCover 2021.

## Pipeline
- Spatial filtering with 60×60 km grid (EPSG:32644)
- 128×128 raster patch label extraction (mode-based)
- 60/40 train-test split
- ResNet18 fine-tuning

## Results
Accuracy: 81.13%  
F1 Score: 0.76

## Tech Stack
Python, GeoPandas, Rasterio, PyTorch, Scikit-learn