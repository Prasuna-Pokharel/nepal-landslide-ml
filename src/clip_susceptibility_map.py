
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask


ROOT = Path(__file__).resolve().parents[1]

RASTER_PATH = (
    ROOT / "data" / "processed"
    / "preliminary_susceptibility.tif"
)

EXTENT_PATH = (
    ROOT / "data" / "processed"
    / "provisional_study_extent.geojson"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed"
    / "preliminary_susceptibility_clipped.tif"
)


def main():
    print("Loading provisional study extent...")

    extent = gpd.read_file(EXTENT_PATH)

    if extent.empty:
        raise ValueError("The study extent file is empty.")

    with rasterio.open(RASTER_PATH) as src:
        # Transform the extent into the raster's CRS.
        if extent.crs is None:
            raise ValueError(
                "The study extent has no CRS information."
            )

        extent = extent.to_crs(src.crs)

        geometries = [
            geometry.__geo_interface__
            for geometry in extent.geometry
            if geometry is not None and not geometry.is_empty
        ]

        if not geometries:
            raise ValueError(
                "No valid geometries found in the study extent."
            )

        print("Clipping susceptibility raster...")

        clipped, transform = mask(
            src,
            geometries,
            crop=True,
            filled=False,
        )

        # Preserve existing NoData pixels and mask outside the extent.
        band = clipped[0]
        nodata = src.nodata if src.nodata is not None else -9999.0
        output_array = band.filled(nodata)

        profile = src.profile.copy()
        profile.update(
            height=output_array.shape[0],
            width=output_array.shape[1],
            transform=transform,
            nodata=nodata,
            compress="deflate",
        )

        with rasterio.open(
            OUTPUT_PATH,
            "w",
            **profile,
        ) as dst:
            dst.write(output_array, 1)

    print("\nClipped raster saved:")
    print(OUTPUT_PATH)
    print(
        "Reminder: this uses a provisional study extent, "
        "not an official Nepal administrative boundary."
    )


if __name__ == "__main__":
    main()
