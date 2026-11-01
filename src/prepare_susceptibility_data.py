
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

POINTS_PATH = (
    ROOT
    / "data"
    / "raw"
    / "DataFile2 - ASMInventory_PointAttributes.txt"
)

DEM_PATH = (
    ROOT / "data" / "processed" / "copernicus_dem_utm45n.tif"
)

SLOPE_PATH = (
    ROOT / "data" / "processed" / "slope_degrees.tif"
)

ASPECT_PATH = (
    ROOT / "data" / "processed" / "aspect_degrees.tif"
)

OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "susceptibility_positive_samples.csv"


# --------------------------------------------------
# 2. Sample raster values at landslide coordinates
# --------------------------------------------------

def sample_raster(path, coordinates):
    """Extract raster values at projected point coordinates."""

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")

        samples = list(src.sample(coordinates, masked=True))
        values = []

        for sample in samples:
            value = sample[0]

            if np.ma.is_masked(value):
                values.append(np.nan)
            else:
                value = float(value)

                if not np.isfinite(value):
                    values.append(np.nan)
                else:
                    values.append(value)

        return np.asarray(values, dtype=float)


# --------------------------------------------------
# 3. Prepare the inventory and terrain dataset
# --------------------------------------------------

def main():

    print("Loading Nepal landslide inventory...")

    if not POINTS_PATH.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {POINTS_PATH}"
        )

    df = pd.read_csv(
        POINTS_PATH,
        sep="\t",
        skiprows=1,
    )

    print(f"Inventory records loaded: {len(df):,}")

    # Confirm that the coordinate columns exist.
    required_columns = ["POINT_X", "POINT_Y"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required coordinate column is missing: {column}"
            )

    # Convert coordinates to numeric values.
    df["POINT_X"] = pd.to_numeric(
        df["POINT_X"],
        errors="coerce",
    )

    df["POINT_Y"] = pd.to_numeric(
        df["POINT_Y"],
        errors="coerce",
    )

    valid_coordinates = (
        np.isfinite(df["POINT_X"])
        & np.isfinite(df["POINT_Y"])
    )

    if not valid_coordinates.all():
        raise ValueError(
            f"{(~valid_coordinates).sum()} records have invalid "
            "coordinates. Resolve these before sampling."
        )

    coordinates = list(
        zip(df["POINT_X"], df["POINT_Y"])
    )

    # The inventory coordinates are expected to use EPSG:32645.
    # Check this against each raster before sampling.
    for raster_path in [DEM_PATH, SLOPE_PATH, ASPECT_PATH]:
        if not raster_path.exists():
            raise FileNotFoundError(
                f"Required raster not found: {raster_path}"
            )

        with rasterio.open(raster_path) as src:
            if src.crs is None or src.crs.to_epsg() != 32645:
                raise ValueError(
                    f"Expected EPSG:32645 for {raster_path.name}; "
                    f"found {src.crs}. Reproject before sampling."
                )

    print("Sampling elevation, slope, and aspect...")

    df["DEM_Elevation"] = sample_raster(
        DEM_PATH,
        coordinates,
    )

    df["DEM_Slope"] = sample_raster(
        SLOPE_PATH,
        coordinates,
    )

    df["DEM_Aspect"] = sample_raster(
        ASPECT_PATH,
        coordinates,
    )

    # --------------------------------------------------
    # 4. Clean invalid measurements
    # --------------------------------------------------

    # -9999 is a missing-data marker, not a real measurement.
    if "Slope" in df.columns:
        df["Slope"] = pd.to_numeric(
            df["Slope"],
            errors="coerce",
        )

        df.loc[df["Slope"] == -9999, "Slope"] = np.nan

    if "Elevation" in df.columns:
        df["Elevation"] = pd.to_numeric(
            df["Elevation"],
            errors="coerce",
        )

    if "Aspect" in df.columns:
        df["Aspect"] = pd.to_numeric(
            df["Aspect"],
            errors="coerce",
        )

        # Aspect values should be between 0 and 360 degrees.
        df.loc[
            (df["Aspect"] < 0) | (df["Aspect"] > 360),
            "Aspect",
        ] = np.nan

    # Keep the recorded inventory points separate from background
    # samples. These records represent known landslide locations.
    df["sample_type"] = "landslide_inventory"

    # --------------------------------------------------
    # 5. Validate sampled terrain values
    # --------------------------------------------------

    terrain_columns = [
        "DEM_Elevation",
        "DEM_Slope",
        "DEM_Aspect",
    ]

    print("\n=== TERRAIN DATA QUALITY ===")

    for column in terrain_columns:
        valid_count = df[column].notna().sum()
        missing_count = df[column].isna().sum()

        print(
            f"{column}: {valid_count:,} valid, "
            f"{missing_count:,} missing"
        )

    print("\n=== DERIVED TERRAIN SUMMARY ===")
    print(df[terrain_columns].describe().round(2))

    # Do not silently discard records with missing raster values.
    # Save all records so missingness can be investigated later.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nRecords saved: {len(df):,}")
    print(f"Output file: {OUTPUT_PATH}")

    print(
        "\nPreparation complete. "
        "This file contains recorded landslide locations only; "
        "it is not yet a complete susceptibility training dataset."
    )


if __name__ == "__main__":
    main()
