
import pandas as pd
import json
from advisory_system import generate_advisory
from stress_detection import run_ml_classifier
from yield_prediction import run as run_yield_model


def export():
    weekly = pd.read_csv("../data/weekly_with_rule_based_stress.csv")
    yield_df = pd.read_csv("../data/yield_data.csv")
    season = pd.read_csv("../data/season_features.csv")
    merged = weekly.merge(yield_df[["plot_id", "stress_level_true"]], on="plot_id")
    ndvi_traj = {}
    for pid, g in merged.sort_values("week").groupby("plot_id"):
        ndvi_traj[pid] = {
            "stress": int(g["stress_level_true"].iloc[0]),
            "weeks": g["week"].tolist(),
            "ndvi": [round(v, 4) for v in g["NDVI"].tolist()],
            "ndre": [round(v, 4) for v in g["NDRE"].tolist()],
        }

    rb_counts = {str(k): int(v) for k, v in weekly["stress_rule_based"].value_counts().sort_index().items()}

    weekly_feat = pd.read_csv("../data/weekly_features.csv")
    clf, acc, report, cm, importances = run_ml_classifier(weekly_feat)
    from sklearn.metrics import precision_recall_fscore_support
    weekly_feat2 = weekly_feat.copy()
    weekly_feat2["NDVI_pct_change_2wk"] = weekly_feat2["NDVI_pct_change_2wk"].fillna(0)
    yt = yield_df[["plot_id", "stress_level_true"]]
    weekly_feat2 = weekly_feat2.merge(yt, on="plot_id")
    feature_cols = ["NDVI", "NDRE", "NDVI_pct_change_2wk", "temp_C", "rainfall_mm",
                     "humidity_pct", "solar_rad_MJ_m2", "soil_moisture_pct", "soil_temp_C",
                     "soil_pH", "organic_carbon_pct", "nitrogen_pct", "clay_pct"]
    from sklearn.model_selection import train_test_split
    X = weekly_feat2[feature_cols]
    y = weekly_feat2["stress_level_true"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    y_pred = clf.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2])
    names = ["Healthy", "Mild", "Severe"]

    ml_report = {
        "accuracy": round(float(acc), 4),
        "precision": {n: round(float(v), 3) for n, v in zip(names, p)},
        "recall": {n: round(float(v), 3) for n, v in zip(names, r)},
        "f1": {n: round(float(v), 3) for n, v in zip(names, f1)},
        "confusion_matrix": cm.tolist(),
        "importances": {k: round(float(v), 4) for k, v in importances.head(6).items()},
    }
    yield_results, best_name, yield_importances, _ = run_yield_model()
    yield_results_out = {
        name: {k: round(float(v), 4) for k, v in metrics.items()}
        for name, metrics in yield_results.items()
    }
    yield_importances_out = {k: round(float(v), 4) for k, v in yield_importances.head(10).items()}

    scatter = season[["plot_id", "NDVI_mean", "yield_tonnes_per_ha", "soil_moisture_mean",
                       "stress_level_true"]].round(4).to_dict(orient="records")

    latest = weekly.sort_values("week").groupby("plot_id").tail(1).copy()
    latest["advisory"] = latest.apply(generate_advisory, axis=1)
    adv_records = latest[["plot_id", "week", "NDVI", "NDRE", "soil_moisture_pct", "temp_C",
                           "rainfall_mm", "soil_pH", "nitrogen_pct", "stress_rule_based",
                           "advisory"]].round(3).to_dict(orient="records")

    out = {
        "ndvi_traj": ndvi_traj,
        "rule_based_counts": rb_counts,
        "ml_report": ml_report,
        "yield_results": yield_results_out,
        "yield_importances": yield_importances_out,
        "scatter": scatter,
        "advisories": adv_records,
    }

    with open("../outputs/dashboard_data.json", "w") as f:
        json.dump(out, f)
    print(f"Wrote ../outputs/dashboard_data.json  ({len(ndvi_traj)} plots, {len(adv_records)} advisories)")


if __name__ == "__main__":
    export()
