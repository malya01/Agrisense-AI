"""
stress_detection.py
--------------------
Module 2: Early Crop Stress Detection.

Two approaches implemented (as recommended in the plan):
  A. Rule-based flagging  -> fast, explainable, good MVP / early-warning trigger
  B. ML classifier (Random Forest) -> learns non-obvious interactions between
     NDVI trend, soil moisture, and weather.

Both are evaluated against `stress_level_true` (0=healthy,1=mild,2=severe),
which in a real deployment would come from either (a) agronomist field
labels, or (b) historical yield-loss-derived labels.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def rule_based_stress(row):
    """
    Simple, explainable thresholds an agronomist could sanity-check:
      - NDVI dropping fast (>15% over 2 weeks) AND
      - soil moisture below 30% AND
      - rainfall in the recent window is low
    -> flag severity based on how many conditions are triggered.
    """
    score = 0
    if pd.notna(row["NDVI_pct_change_2wk"]) and row["NDVI_pct_change_2wk"] < -0.15:
        score += 1
    if row["soil_moisture_pct"] < 30:
        score += 1
    if row["rainfall_mm"] < 3:
        score += 1
    if row["temp_C"] > 30:
        score += 1

    if score >= 3:
        return 2  # severe
    elif score >= 1:
        return 1  # mild
    return 0  # healthy


def run_rule_based(weekly_df):
    weekly_df = weekly_df.copy()
    weekly_df["stress_rule_based"] = weekly_df.apply(rule_based_stress, axis=1)
    return weekly_df


def run_ml_classifier(weekly_df):
    """
    Train a Random Forest to predict per-week stress level from
    NDVI/NDRE + weather + IoT features. Ground truth here (for
    demonstration) is derived from the same underlying plot stress
    profile used to simulate the data -- in production this would be
    replaced by field-verified labels.
    """
    weekly_df = weekly_df.copy()
    weekly_df["NDVI_pct_change_2wk"] = weekly_df["NDVI_pct_change_2wk"].fillna(0)

    # attach ground truth per plot (season-level -> broadcast to weeks)
    yield_df = pd.read_csv("../data/yield_data.csv")[["plot_id", "stress_level_true"]]
    weekly_df = weekly_df.merge(yield_df, on="plot_id")

    feature_cols = [
        "NDVI", "NDRE", "NDVI_pct_change_2wk", "temp_C", "rainfall_mm",
        "humidity_pct", "solar_rad_MJ_m2", "soil_moisture_pct", "soil_temp_C",
        "soil_pH", "organic_carbon_pct", "nitrogen_pct", "clay_pct",
    ]
    X = weekly_df[feature_cols]
    y = weekly_df["stress_level_true"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Mild", "Severe"])
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(clf, "../models/stress_classifier_rf.pkl")

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return clf, acc, report, cm, importances


if __name__ == "__main__":
    weekly = pd.read_csv("../data/weekly_features.csv")

    print("=== A. Rule-Based Stress Detection ===")
    rb = run_rule_based(weekly)
    rb.to_csv("../data/weekly_with_rule_based_stress.csv", index=False)
    print(rb["stress_rule_based"].value_counts().sort_index())

    print("\n=== B. ML-Based Stress Detection (Random Forest) ===")
    clf, acc, report, cm, importances = run_ml_classifier(weekly)
    print(f"Test Accuracy: {acc:.3f}\n")
    print(report)
    print("Confusion Matrix:\n", cm)
    print("\nTop feature importances:\n", importances.head(6))
