
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "susceptibility_model.joblib"
PROCESSED_DIR = ROOT / "data" / "processed"

DEM_PATH = PROCESSED_DIR / "copernicus_dem_utm45n.tif"
SLOPE_PATH = PROCESSED_DIR / "slope_degrees.tif"
ASPECT_PATH = PROCESSED_DIR / "aspect_degrees.tif"

OUTPUT_PATH = (
    PROCESSED_DIR / "preliminary_susceptibility.tif"
)

FEATURE_NAMES = [
    "DEM_Elevation",
    "DEM_Slope",
    "DEM_Aspect",
]

BLOCK_ROWS = 256


def main():
    print("Loading trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    raster_paths = [
        DEM_PATH,
        SLOPE_PATH,
        ASPECT_PATH,
    ]

    for path in raster_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required terrain raster not found: {path}"
            )

    print("Opening terrain rasters...")

    with (
        rasterio.open(DEM_PATH) as dem,
        rasterio.open(SLOPE_PATH) as slope,
        rasterio.open(ASPECT_PATH) as aspect,
    ):
        # All rasters must have matching grids.
        reference = (dem.width, dem.height, dem.transform, dem.crs)

        for name, src in [
            ("slope", slope),
            ("aspect", aspect),
        ]:
            current = (
                src.width,
                src.height,
                src.transform,
                src.crs,
            )

            if current != reference:
                raise ValueError(
                    f"{name} raster does not match the DEM grid. "
                    "Reproject/resample it to the DEM grid first."
                )

        # Output uses the DEM grid and a floating-point probability.
        profile = dem.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
            tiled=True,
        )

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_blocks = (
            dem.height + BLOCK_ROWS - 1
        ) // BLOCK_ROWS

        print("Predicting susceptibility by raster block...")

        with rasterio.open(
            OUTPUT_PATH,
            "w",
            **profile,
        ) as output:
            for block_number, row_start in enumerate(
                range(0, dem.height, BLOCK_ROWS),
                start=1,
            ):
                height = min(
                    BLOCK_ROWS,
                    dem.height - row_start,
                )

                window = Window(
                    col_off=0,
                    row_off=row_start,
                    width=dem.width,
                    height=height,
                )

                elevation = dem.read(
                    1,
                    window=window,
                    masked=True,
                )

                slope_values = slope.read(
                    1,
                    window=window,
                    masked=True,
                )

                aspect_values = aspect.read(
                    1,
                    window=window,
                    masked=True,
                )

                # Combine raster masks and invalid values.
                elevation_data = np.asarray(
                    elevation.filled(np.nan),
                    dtype=np.float32,
                )

                slope_data = np.asarray(
                    slope_values.filled(np.nan),
                    dtype=np.float32,
                )

                aspect_data = np.asarray(
                    aspect_values.filled(np.nan),
                    dtype=np.float32,
                )

                valid = (
                    np.isfinite(elevation_data)
                    & np.isfinite(slope_data)
                    & np.isfinite(aspect_data)
                )

                result = np.full(
                    elevation_data.shape,
                    -9999.0,
                    dtype=np.float32,
                )

                if valid.any():
                    # Keep the exact feature order used in training.
                    X_block = np.column_stack(
                        [
                            elevation_data[valid],
                            slope_data[valid],
                            aspect_data[valid],
                        ]
                    )

                    probabilities = model.predict_proba(
                        X_block
                    )[:, 1]

                    result[valid] = probabilities.astype(
                        np.float32
                    )

                output.write(
                    result,
                    1,
                    window=window,
                )

                print(
                    f"Processed block "
                    f"{block_number}/{total_blocks}"
                )

    print("\nMap raster created successfully:")
    print(OUTPUT_PATH)
    print(
        "\nValues range from 0 to 1 and represent model "
        "scores, not calibrated real-world landslide probabilities."
    )


if __name__ == "__main__":
    main()
