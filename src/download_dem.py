
from pathlib import Path

import requests
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "dem_tiles"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Approximate study extent; not an official study boundary.
WEST, SOUTH, EAST, NORTH = 84.30, 26.74, 88.19, 28.72

BASE_URL = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
tile_paths = []

# Download every 1-degree tile intersecting the extent.
for lat in range(int(SOUTH), int(NORTH) + 1):
    for lon in range(int(WEST), int(EAST) + 1):
        name = (
            f"Copernicus_DSM_COG_30_N{lat:02d}_00_"
            f"E{lon:03d}_00_DEM"
        )
        url = f"{BASE_URL}/{name}/{name}.tif"
        path = RAW_DIR / f"{name}.tif"

        if not path.exists():
            print(f"Downloading {name}...")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        else:
            print(f"Already downloaded: {name}")

        # Verify the downloaded file is a readable raster.
        with rasterio.open(path) as src:
            print(f"  CRS: {src.crs}, size: {src.width} x {src.height}")

        tile_paths.append(path)

print(f"\nDownloaded/verified {len(tile_paths)} tiles.")

# Merge tiles into a single raster.
sources = [rasterio.open(path) for path in tile_paths]

try:
    mosaic, transform = merge(sources)
    profile = sources[0].profile.copy()
finally:
    for src in sources:
        src.close()

# Clip the mosaic to the provisional geographic extent.
window = from_bounds(
    WEST, SOUTH, EAST, NORTH,
    transform,
).round_offsets().round_lengths()

row_start = max(0, int(window.row_off))
col_start = max(0, int(window.col_off))
row_end = min(mosaic.shape[1], row_start + int(window.height))
col_end = min(mosaic.shape[2], col_start + int(window.width))

clipped = mosaic[:, row_start:row_end, col_start:col_end]
clipped_transform = rasterio.windows.transform(
    rasterio.windows.Window(
        col_start, row_start,
        col_end - col_start, row_end - row_start,
    ),
    transform,
)

profile.update(
    height=clipped.shape[1],
    width=clipped.shape[2],
    transform=clipped_transform,
    compress="deflate",
)

output_path = PROCESSED_DIR / "copernicus_dem_glo90.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(clipped)

print("\nSaved DEM:", output_path)
print("Raster size:", clipped.shape[2], "columns x", clipped.shape[1], "rows")
print("CRS:", profile["crs"])
print("Resolution:", profile["transform"].a, "degrees per pixel")
print("Source: Copernicus DEM GLO-90")
print("Note: clipped to an approximate extent, not an official boundary.")
