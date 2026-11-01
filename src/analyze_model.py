
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "reactivation_model.joblib"
OUTPUT_PATH = ROOT / "models" / "feature_importance.csv"

model = joblib.load(MODEL_PATH)

# Get feature names after preprocessing, including encoded categories.
preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importance = classifier.feature_importances_

results = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
}).sort_values("importance", ascending=False)

print("\nTop 15 model features:")
print(results.head(15).to_string(index=False))

results.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved feature rankings to: {OUTPUT_PATH}")
