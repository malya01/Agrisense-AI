

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


FEATURE_COLS = [
    "NDVI_peak", "NDVI_mean", "NDRE_peak", "NDRE_mean",
    "temp_mean", "rainfall_total", "humidity_mean", "solar_mean",
    "soil_moisture_mean", "soil_moisture_min",
    "soil_pH", "organic_carbon_pct", "nitrogen_pct", "clay_pct",
]
TARGET_COL = "yield_tonnes_per_ha"


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def run():
    df = pd.read_csv("../data/season_features.csv")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    results = {}

    rf = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["RandomForest"] = evaluate(y_test, rf_pred)

    xgb = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = evaluate(y_test, xgb_pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results["RandomForest"]["CV_R2_mean"] = cross_val_score(rf, X, y, cv=kf, scoring="r2").mean()
    results["XGBoost"]["CV_R2_mean"] = cross_val_score(xgb, X, y, cv=kf, scoring="r2").mean()

    best_name = max(results, key=lambda k: results[k]["CV_R2_mean"])
    best_model = rf if best_name == "RandomForest" else xgb
    joblib.dump(best_model, "../models/yield_model_best.pkl")

    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return results, best_name, importances, (y_test, rf_pred, xgb_pred)


if __name__ == "__main__":
    results, best_name, importances, preds = run()

    print("=== Yield Prediction: Model Comparison ===\n")
    for name, metrics in results.items():
        print(f"{name}:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        print()

    print(f"Best model (by 5-fold CV R2): {best_name}\n")
    print("Top feature importances (best model):")
    print(importances.head(8))
