from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "DataFile2 - ASMInventory_PointAttributes.txt"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "Elevation", "Slope", "Aspect", "PlanCurv", "ProfCurv",
    "ExToppo", "LocalRelief", "DistChan", "DistRoad",
    "NearChanKsn", "NearChanSSP", "30Yr_AvPcP",
    "1Yr_TotPcP", "1Yr_PkMnthPcP",
]
CATEGORICAL_FEATURES = ["Landuse_Co", "Geology", "Tectonic_Unit"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Reactivate"


def main():
    df = pd.read_csv(DATA_PATH, sep="\t", skiprows=1)

    # Standardize inconsistent target labels.
    df[TARGET] = df[TARGET].astype("string").str.strip().str.upper()
    df = df[df[TARGET].isin(["Y", "N"])].copy()

    # Use the environmental features only.
    X = df[FEATURES].copy()
    y = df[TARGET].map({"N": 0, "Y": 1})

    # Aspect -1 is an invalid direction value in this dataset.
    X.loc[X["Aspect"] == -1, "Aspect"] = float("nan")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "task": "reactivation classification among recorded landslides",
        "rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_class": "Y (reactivated)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels": ["N", "Y"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    print("\n=== Reactivation Model: Initial Baseline ===")
    print(f"Records used: {len(df):,}")
    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")
    print("\nTarget distribution:")
    print(df[TARGET].value_counts().to_string())
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["N (not reactivated)", "Y (reactivated)"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted; order N, Y):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"\nAccuracy: {metrics['accuracy']:.3f}")
    print(f"F1 score: {metrics['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    joblib.dump(model, MODEL_DIR / "reactivation_model.joblib")
    with open(MODEL_DIR / "reactivation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model: models/reactivation_model.joblib")
    print("Saved metrics: models/reactivation_metrics.json")
    print("\nIMPORTANT: This is an initial random-split baseline, not a")
    print("validated landslide susceptibility or real-world risk model.")


if __name__ == "__main__":
    main()
