
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "copernicus_dem_glo90.tif"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECTED_DEM = OUTPUT_DIR / "copernicus_dem_utm45n.tif"
SLOPE_PATH = OUTPUT_DIR / "slope_degrees.tif"
ASPECT_PATH = OUTPUT_DIR / "aspect_degrees.tif"

TARGET_CRS = "EPSG:32645"
TARGET_RESOLUTION = 90.0
NODATA = -9999.0


def main():
    # 1. Reproject the DEM into metres.
    with rasterio.open(INPUT) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RESOLUTION,
        )

        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            count=1,
            nodata=NODATA,
            compress="deflate",
        )

        with rasterio.open(PROJECTED_DEM, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )

    print("Saved projected DEM:", PROJECTED_DEM)

    # 2. Calculate terrain gradients from the projected DEM.
    with rasterio.open(PROJECTED_DEM) as src:
        elevation = src.read(1).astype("float64")
        valid = np.isfinite(elevation) & (elevation != NODATA)

        # Avoid calculating gradients through missing cells.
        filled = elevation.copy()
        filled[~valid] = np.nan

        pixel_x = abs(src.transform.a)
        pixel_y = abs(src.transform.e)

        # Central differences estimate elevation change per metre.
        dz_dy, dz_dx = np.gradient(filled, pixel_y, pixel_x)

        slope = np.degrees(
            np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        )

        # Compass aspect: 0=N, 90=E, 180=S, 270=W.
        aspect = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360) % 360

        # Flat terrain has no meaningful compass aspect.
        flat = np.sqrt(dz_dx**2 + dz_dy**2) < 1e-6
        aspect[flat] = np.nan

        # Do not save gradients next to missing elevation cells.
        valid_gradient = valid & np.isfinite(slope) & np.isfinite(aspect)
        slope[~valid_gradient] = NODATA
        aspect[~valid_gradient] = NODATA

        output_profile = src.profile.copy()
        output_profile.update(dtype="float32", nodata=NODATA)

        with rasterio.open(SLOPE_PATH, "w", **output_profile) as dst:
            dst.write(slope.astype("float32"), 1)

        with rasterio.open(ASPECT_PATH, "w", **output_profile) as dst:
            dst.write(aspect.astype("float32"), 1)

    print("Saved slope:", SLOPE_PATH)
    print("Saved aspect:", ASPECT_PATH)

    for path in [PROJECTED_DEM, SLOPE_PATH, ASPECT_PATH]:
        with rasterio.open(path) as src:
            values = src.read(1, masked=True)
            print(
                f"{path.name}: CRS={src.crs}, "
                f"size={src.width}x{src.height}, "
                f"range={values.min():.2f} to {values.max():.2f}"
            )


if __name__ == "__main__":
    main()
