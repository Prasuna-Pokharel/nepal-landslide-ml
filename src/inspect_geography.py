
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

POINT_PATH = (
    ROOT / "data" / "raw" /
    "DataFile2 - ASMInventory_PointAttributes.txt"
)

POLYGON_PATH = (
    ROOT / "data" / "raw" / "landslide_polygons" /
    "DataFile1 - 12838ASMInventory.shp"
)

# Read the point inventory.
points_df = pd.read_csv(POINT_PATH, sep="\t", skiprows=1)

points = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(
        points_df["POINT_X"],
        points_df["POINT_Y"],
    ),
    crs="EPSG:32645",
)

# Read the landslide polygons.
polygons = gpd.read_file(POLYGON_PATH)

print("\n=== POINT INVENTORY ===")
print("Records:", len(points))
print("CRS:", points.crs)
print("Bounds:", points.total_bounds)

print("\n=== LANDSLIDE POLYGONS ===")
print("Records:", len(polygons))
print("CRS:", polygons.crs)
print("Bounds:", polygons.total_bounds)

# Convert bounds to latitude and longitude for easier interpretation.
points_wgs84 = points.to_crs("EPSG:4326")
print("\n=== POINT COVERAGE (longitude, latitude) ===")
print("Bounds:", points_wgs84.total_bounds)

# Check whether the two datasets use the same CRS.
print("\nCRS matches:", points.crs == polygons.crs)
