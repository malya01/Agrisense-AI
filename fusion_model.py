"""
fusion_model.py
----------------
This is what makes the system genuinely MULTIMODAL: it takes the CNN's
image-based stress prediction and adds it as an EXTRA FEATURE alongside
the satellite/weather/soil/IoT features, then retrains the stress
classifier with this richer feature set.

Simple idea: two "opinions" (image-based + sensor-based) combined are
more reliable than either alone -- same principle as a doctor combining
an X-ray reading with blood test results, not just using one.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from image_branch import generate_synthetic_images, CLASS_NAMES
import joblib


def get_image_branch_scores_per_plot(weekly_df):
    """
    In production: for each plot+week, you'd run its REAL field photo
    through the trained CNN to get a stress probability score.

    Here: we simulate a per-plot-week "image stress score" that's
    correlated with the plot's true stress level (same idea as a real
    CNN's output would be), using the trained CNN on a synthetic image
    that matches that plot's stress class -- demonstrating the fusion
    pipeline end-to-end without needing real field photos yet.
    """
    model = tf.keras.models.load_model("../models/image_branch_cnn.keras")

    # generate one representative image per stress class, run through CNN
    # to get realistic prediction probabilities for use as a feature
    X, y = generate_synthetic_images()
    preds = model.predict(X, verbose=0)  # shape (N, 3) probabilities

    # average predicted "severity score" (0=healthy .. 2=severe) per true class
    severity_by_class = {}
    for class_idx in range(3):
        mask = y == class_idx
        # severity score = weighted average of class index by predicted probability
        weighted_severity = (preds[mask] * np.array([0, 1, 2])).sum(axis=1)
        severity_by_class[class_idx] = weighted_severity.mean()

    weekly_df = weekly_df.copy()
    weekly_df["image_stress_score"] = weekly_df["stress_level_true"].map(severity_by_class)
    # add realistic per-observation noise so it's not a perfect copy
    weekly_df["image_stress_score"] += np.random.normal(0, 0.15, len(weekly_df))
    weekly_df["image_stress_score"] = weekly_df["image_stress_score"].clip(0, 2)
    return weekly_df


def run_fused_classifier():
    weekly = pd.read_csv("../data/weekly_features.csv")
    weekly["NDVI_pct_change_2wk"] = weekly["NDVI_pct_change_2wk"].fillna(0)

    yield_df = pd.read_csv("../data/yield_data.csv")[["plot_id", "stress_level_true"]]
    weekly = weekly.merge(yield_df, on="plot_id")

    weekly = get_image_branch_scores_per_plot(weekly)

    feature_cols = [
        "NDVI", "NDRE", "NDVI_pct_change_2wk", "temp_C", "rainfall_mm",
        "humidity_pct", "solar_rad_MJ_m2", "soil_moisture_pct", "soil_temp_C",
        "soil_pH", "organic_carbon_pct", "nitrogen_pct", "clay_pct",
        "image_stress_score",  # <-- the new fused feature from the image branch
    ]
    X = weekly[feature_cols]
    y = weekly["stress_level_true"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Mild", "Severe"])
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    joblib.dump(clf, "../models/fused_stress_classifier.pkl")
    return acc, report, importances


if __name__ == "__main__":
    acc, report, importances = run_fused_classifier()
    print(f"=== FUSED Multimodal Stress Classifier (Sensors + Image Branch) ===\n")
    print(f"Test Accuracy: {acc:.3f}\n")
    print(report)
    print("Feature importances (note where image_stress_score ranks):")
    print(importances)
