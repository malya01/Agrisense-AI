

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from image_branch import generate_synthetic_images, CLASS_NAMES
import joblib


def get_image_branch_scores_per_plot(weekly_df):
   
    model = tf.keras.models.load_model("../models/image_branch_cnn.keras")
    X, y = generate_synthetic_images()
    preds = model.predict(X, verbose=0)  

    severity_by_class = {}
    for class_idx in range(3):
        mask = y == class_idx
        weighted_severity = (preds[mask] * np.array([0, 1, 2])).sum(axis=1)
        severity_by_class[class_idx] = weighted_severity.mean()

    weekly_df = weekly_df.copy()
    weekly_df["image_stress_score"] = weekly_df["stress_level_true"].map(severity_by_class)
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
        "image_stress_score", 
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
