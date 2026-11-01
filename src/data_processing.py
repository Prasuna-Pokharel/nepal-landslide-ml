
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed_environmental_data.csv"


def load_data() -> pd.DataFrame:
    """Load the processed environmental dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The processed dataset is empty.")

    return df


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """Return basic data-quality checks for the dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_features": int(
            len(df.select_dtypes(include="number").columns)
        ),
        "categorical_features": int(
            len(df.select_dtypes(exclude="number").columns)
        ),
    }
