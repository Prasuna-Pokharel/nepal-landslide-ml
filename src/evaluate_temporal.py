
from pathlib import Path

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "DataFile2 - ASMInventory_PointAttributes.txt"

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

    df[TARGET] = df[TARGET].astype("string").str.strip().str.upper()
    df = df[df[TARGET].isin(["Y", "N"])].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).sort_values("Year")

    X = df[FEATURES].copy()
    y = df[TARGET].map({"N": 0, "Y": 1})

    X.loc[X["Aspect"] == -1, "Aspect"] = float("nan")

    # Train on earlier years, test on the latest 20% of records by year.
    cutoff = df["Year"].quantile(0.80)
    train_mask = df["Year"] < cutoff
    test_mask = df["Year"] >= cutoff

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    print(f"Training years: {df.loc[train_mask, 'Year'].min():.0f}–"
          f"{df.loc[train_mask, 'Year'].max():.0f}")
    print(f"Test years: {df.loc[test_mask, 'Year'].min():.0f}–"
          f"{df.loc[test_mask, 'Year'].max():.0f}")
    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
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

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nTemporal evaluation report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["N (not reactivated)", "Y (reactivated)"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted; N, Y):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Reactivation F1: {f1_score(y_test, predictions, zero_division=0):.3f}")

    if y_test.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    else:
        print("ROC-AUC unavailable: the test set contains only one class.")


if __name__ == "__main__":
    main()
