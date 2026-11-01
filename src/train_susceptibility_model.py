
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "susceptibility_training_data.csv"
)

MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "susceptibility_model.joblib"

METRICS_PATH = (
    MODEL_DIR / "susceptibility_metrics.json"
)

FEATURES = [
    "DEM_Elevation",
    "DEM_Slope",
    "DEM_Aspect",
]

TARGET = "label"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

def main():

    print("Loading susceptibility training dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET, "sample_type"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Use only the three terrain predictors for this prototype.
    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    # Invalid labels must not be used for training.
    valid_labels = y.isin([0, 1])
    X = X.loc[valid_labels].copy()
    y = y.loc[valid_labels].astype(int)

    # Make sure predictors are numeric.
    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Replace impossible/non-finite measurements with missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove rows with no usable predictor information.
    usable_rows = X.notna().any(axis=1)
    X = X.loc[usable_rows].copy()
    y = y.loc[usable_rows].copy()

    if y.nunique() != 2:
        raise ValueError(
            "Training requires both label classes: 0 and 1."
        )

    print(f"Samples available: {len(X):,}")
    print("\nClass counts:")
    print(y.value_counts().sort_index())

    print("\nPredictors:")
    for feature in FEATURES:
        print(f"- {feature}")

    # --------------------------------------------------
    # 3. Split data into training and test sets
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    # --------------------------------------------------
    # 4. Build and train the Random Forest
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest model...")

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 5. Evaluate on the held-out test set
    # --------------------------------------------------

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "Background",
            "Recorded landslide",
        ],
        output_dict=True,
        zero_division=0,
    )

    print("\n=== PRELIMINARY MODEL RESULTS ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC: {auc:.3f}")

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Background",
                "Recorded landslide",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 6. Save the model and metrics
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    metrics = {
        "model": "RandomForestClassifier",
        "features": FEATURES,
        "target": TARGET,
        "sample_count": int(len(X)),
        "training_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "split_method": "stratified random split",
        "accuracy": float(accuracy),
        "roc_auc": float(auc),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "interpretation": (
            "Preliminary prototype. Label 0 represents sampled "
            "background locations, not verified landslide absence. "
            "The study extent is provisional. Random-split metrics "
            "do not establish spatial or temporal generalization."
        ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    print(
        "\nTraining complete. These results are preliminary "
        "and must not be interpreted as validated susceptibility."
    )


if __name__ == "__main__":
    main()
