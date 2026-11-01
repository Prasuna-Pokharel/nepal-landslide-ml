
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "DataFile2 - ASMInventory_PointAttributes.txt"

df = pd.read_csv(DATA_PATH, sep="\t", skiprows=1)

print("Dataset shape:", df.shape)
print("\nColumns:")
for column in df.columns:
    print(f"- {column}")

print("\nPossible grouping columns:")
for column in df.columns:
    name = column.lower()
    if any(term in name for term in ["year", "event", "id", "date", "group"]):
        print(f"\n{column}")
        print("Unique values:", df[column].nunique(dropna=True))
        print(df[column].value_counts(dropna=False).head(8).to_string())
