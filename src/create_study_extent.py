
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Approximate extent from the published BGS inventory metadata.
# This is NOT an official study-area boundary.
min_lon, min_lat = 84.30, 26.74
max_lon, max_lat = 88.19, 28.72

extent = gpd.GeoDataFrame(
    {
        "name": ["Provisional central-eastern Nepal extent"],
        "boundary_type": ["Approximate metadata bounding box"],
        "official_boundary": [False],
        "source": ["BGS landslide inventory metadata"],
    },
    geometry=[box(min_lon, min_lat, max_lon, max_lat)],
    crs="EPSG:4326",
)

output_path = OUTPUT_DIR / "provisional_study_extent.geojson"
extent.to_file(output_path, driver="GeoJSON")

print("Saved provisional extent:", output_path)
print("CRS:", extent.crs)
print("Official boundary:", extent.loc[0, "official_boundary"])
