
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]

POINTS_PATH = (
    ROOT / "data" / "raw"
    / "DataFile2 - ASMInventory_PointAttributes.txt"
)

DEM_PATH = ROOT / "data" / "processed" / "copernicus_dem_utm45n.tif"
SLOPE_PATH = ROOT / "data" / "processed" / "slope_degrees.tif"
ASPECT_PATH = ROOT / "data" / "processed" / "aspect_degrees.tif"

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "point_terrain_validation.csv"
)


def sample_raster(path, coordinates):
    """Sample one raster at projected point coordinates."""

    with rasterio.open(path) as src:
        samples = src.sample(coordinates, masked=True)

        values = np.array(
            [sample[0] for sample in samples],
            dtype=float,
        )

        values[~np.isfinite(values)] = np.nan
        return values


def main():
    # Load the landslide inventory.
    df = pd.read_csv(
        POINTS_PATH,
        sep="\t",
        skiprows=1,
    )

    # Get the projected coordinates of the inventory points.
    coordinates = list(zip(df["POINT_X"], df["POINT_Y"]))

    # Extract terrain values from the rasters.
    df["DEM_Elevation"] = sample_raster(DEM_PATH, coordinates)
    df["DEM_Slope"] = sample_raster(SLOPE_PATH, coordinates)
    df["DEM_Aspect"] = sample_raster(ASPECT_PATH, coordinates)

    # Check terrain coverage.
    print("\n=== TERRAIN COVERAGE ===")

    for column in ["DEM_Elevation", "DEM_Slope", "DEM_Aspect"]:
        print(
            f"{column}: "
            f"{df[column].notna().sum():,}/{len(df):,} valid"
        )

    # Compare original inventory values with DEM-derived values.
    comparisons = [
        ("Elevation", "DEM_Elevation"),
        ("Slope", "DEM_Slope"),
    ]

    print("\n=== COMPARISON WITH INVENTORY ===")

    for original, derived in comparisons:
        actual_all = pd.to_numeric(
            df[original],
            errors="coerce",
        )

        # Treat -9999 as a missing-data marker.
        actual_all = actual_all.mask(actual_all == -9999)

        valid = actual_all.notna() & df[derived].notna()

        actual = actual_all.loc[valid]
        predicted = df.loc[valid, derived]

        if len(actual) == 0:
            print(f"{original}: no comparable values")
            continue

        correlation = actual.corr(predicted)
        mae = mean_absolute_error(actual, predicted)

        print(f"\n{original} vs {derived}")
        print(f"Comparable points: {len(actual):,}")
        print(f"Mean absolute difference: {mae:.2f}")
        print(f"Pearson correlation: {correlation:.3f}")

    # Inspect the slope value ranges, excluding -9999.
    print("\n=== SLOPE VALUE RANGES ===")

    for column in ["Slope", "DEM_Slope"]:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if column == "Slope":
            values = values.mask(values == -9999)

        print(f"\n{column}")
        print(values.describe())

    # Display the first 10 slope comparisons.
    print("\n=== FIRST 10 SLOPE COMPARISONS ===")

    print(
        df[["Slope", "DEM_Slope"]]
        .head(10)
        .to_string(index=False)
    )

    # Save sampled terrain values.
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved sampled values to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
