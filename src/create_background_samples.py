
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

EXTENT_PATH = (
    ROOT / "data" / "processed"
    / "provisional_study_extent.geojson"
)

POLYGONS_PATH = (
    ROOT / "data" / "raw" / "landslide_polygons"
    / "DataFile1 - 12838ASMInventory.shp"
)

POSITIVE_PATH = (
    ROOT / "data" / "processed"
    / "susceptibility_positive_samples.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed"
    / "susceptibility_training_data.csv"
)

DEM_PATH = ROOT / "data" / "processed" / "copernicus_dem_utm45n.tif"
SLOPE_PATH = ROOT / "data" / "processed" / "slope_degrees.tif"
ASPECT_PATH = ROOT / "data" / "processed" / "aspect_degrees.tif"

CRS = "EPSG:32645"

# One approximate DEM pixel as an initial exclusion distance.
# This is a methodological choice, not a scientifically validated buffer.
EXCLUSION_BUFFER_METERS = 90

RANDOM_SEED = 42
MAX_ATTEMPTS = 30


# --------------------------------------------------
# 2. Extract terrain raster values
# --------------------------------------------------

def sample_raster(path, coordinates):
    """Sample a raster at projected coordinates."""

    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 32645:
            raise ValueError(
                f"{path.name} must use EPSG:32645; found {src.crs}"
            )

        result = []

        for sample in src.sample(coordinates, masked=True):
            value = sample[0]

            if np.ma.is_masked(value):
                result.append(np.nan)
            else:
                value = float(value)
                result.append(
                    value if np.isfinite(value) else np.nan
                )

        return np.asarray(result, dtype=float)


# --------------------------------------------------
# 3. Create candidate background locations
# --------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading study extent and landslide polygons...")

    for path in [EXTENT_PATH, POLYGONS_PATH, POSITIVE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    extent = gpd.read_file(EXTENT_PATH)
    polygons = gpd.read_file(POLYGONS_PATH)
    positives = pd.read_csv(POSITIVE_PATH)

    if extent.empty or polygons.empty or positives.empty:
        raise ValueError("One or more input datasets are empty.")

    # GeoJSON normally uses WGS84 when no CRS is specified.
    if extent.crs is None:
        extent = extent.set_crs("EPSG:4326")

    if polygons.crs is None:
        raise ValueError("The landslide polygon file has no CRS.")

    extent = extent.to_crs(CRS)
    polygons = polygons.to_crs(CRS)

    # Combine the study-area geometries.
    study_area = extent.geometry.union_all()

    # Exclude known landslide polygons and nearby locations.
    # This reduces overlap with mapped landslide footprints.
    valid_polygons = polygons.geometry[
        polygons.geometry.notna()
        & ~polygons.geometry.is_empty
    ]

    landslide_areas = valid_polygons.union_all()

    exclusion_area = landslide_areas.buffer(
        EXCLUSION_BUFFER_METERS
    )

    print(f"Study extent CRS: {extent.crs}")
    print(f"Landslide polygon CRS: {polygons.crs}")
    print(
        f"Exclusion buffer: {EXCLUSION_BUFFER_METERS} metres"
    )

    # The existing positive dataset contains the inventory points.
    required_positive_columns = [
        "POINT_X",
        "POINT_Y",
        "DEM_Elevation",
        "DEM_Slope",
        "DEM_Aspect",
    ]

    missing_columns = [
        column
        for column in required_positive_columns
        if column not in positives.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Positive dataset is missing columns: {missing_columns}"
        )

    positives = positives[required_positive_columns].copy()

    positives = positives.rename(
        columns={
            "POINT_X": "x",
            "POINT_Y": "y",
        }
    )

    positives["label"] = 1
    positives["sample_type"] = "landslide_inventory"
    positives["label_note"] = (
        "Recorded landslide location in the inventory"
    )

    # Ensure positive samples have valid terrain measurements.
    terrain_columns = [
        "DEM_Elevation",
        "DEM_Slope",
        "DEM_Aspect",
    ]

    positives = positives.dropna(
        subset=["x", "y"] + terrain_columns
    )

    target_count = len(positives)

    if target_count == 0:
        raise ValueError("No valid positive samples remain.")

    print(f"Usable landslide samples: {target_count:,}")

    # --------------------------------------------------
    # 4. Randomly sample background candidates
    # --------------------------------------------------

    print("Sampling candidate background locations...")

    min_x, min_y, max_x, max_y = study_area.bounds

    accepted_points = []
    attempts = 0
    batch_size = 20000

    while (
        len(accepted_points) < target_count
        and attempts < MAX_ATTEMPTS
    ):
        attempts += 1

        xs = rng.uniform(min_x, max_x, batch_size)
        ys = rng.uniform(min_y, max_y, batch_size)

        candidates = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(xs, ys)],
            crs=CRS,
        )

        # Keep candidates inside the provisional study extent.
        inside_extent = candidates.within(study_area)

        # Remove points within the landslide polygons/buffer.
        outside_exclusion = ~candidates.intersects(exclusion_area)

        accepted = candidates[
            inside_extent & outside_exclusion
        ]

        accepted_points.extend(
            list(accepted.iloc[:target_count - len(accepted)])
        )

        print(
            f"Attempt {attempts}: "
            f"{len(accepted_points):,}/{target_count:,} "
            "background candidates collected"
        )

    if len(accepted_points) < target_count:
        raise RuntimeError(
            f"Only {len(accepted_points):,} background candidates "
            f"were found, but {target_count:,} are needed. "
            "Do not reduce the exclusion buffer automatically. "
            "Review the extent and sampling settings."
        )

    # --------------------------------------------------
    # 5. Extract background terrain values
    # --------------------------------------------------

    background_coordinates = [
        (point.x, point.y)
        for point in accepted_points
    ]

    background = pd.DataFrame({
        "x": [point.x for point in accepted_points],
        "y": [point.y for point in accepted_points],
    })

    print("Extracting terrain values for background points...")

    background["DEM_Elevation"] = sample_raster(
        DEM_PATH,
        background_coordinates,
    )

    background["DEM_Slope"] = sample_raster(
        SLOPE_PATH,
        background_coordinates,
    )

    background["DEM_Aspect"] = sample_raster(
        ASPECT_PATH,
        background_coordinates,
    )

    # Remove candidates with missing raster values.
    background = background.dropna(
        subset=terrain_columns
    )

    if background.empty:
        raise RuntimeError(
            "No background points have complete terrain values."
        )

    background["label"] = 0
    background["sample_type"] = "background"
    background["label_note"] = (
        "Unrecorded background location; "
        "not confirmed landslide-free"
    )

    # --------------------------------------------------
    # 6. Combine and save the modeling dataset
    # --------------------------------------------------

    columns = [
        "x",
        "y",
        "DEM_Elevation",
        "DEM_Slope",
        "DEM_Aspect",
        "label",
        "sample_type",
        "label_note",
    ]

    training_data = pd.concat(
        [
            positives[columns],
            background[columns],
        ],
        ignore_index=True,
    )

    training_data = training_data.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== DATASET SUMMARY ===")
    print(f"Total samples: {len(training_data):,}")
    print(training_data["sample_type"].value_counts())
    print("\nLabel counts:")
    print(training_data["label"].value_counts().sort_index())

    print(f"\nSaved dataset to: {OUTPUT_PATH}")

    print(
        "\nIMPORTANT: label 0 represents background sampling, "
        "not verified landslide absence. The study extent is "
        "provisional, and this dataset is not yet a validated "
        "susceptibility model."
    )


if __name__ == "__main__":
    main()
