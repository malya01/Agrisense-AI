"""
advisory_system.py
-------------------
Module 4: Smart Farmer Advisory System.
Takes the outputs of Modules 2 & 3 (stress level, predicted yield) plus
raw soil/weather/IoT readings, and produces plain-language recommendations.
Deliberately rule-based (a decision layer, not another ML model) so every
recommendation is explainable to a farmer/agronomist -- important for trust
and adoption in real deployments.
"""

import pandas as pd


def generate_advisory(row):
    tips = []

    # --- Irrigation ---
    if row["soil_moisture_pct"] < 25:
        tips.append("URGENT: Soil moisture critically low (<25%). Irrigate within 24-48 hours.")
    elif row["soil_moisture_pct"] < 35 and row["rainfall_mm"] < 5:
        tips.append("Soil moisture trending low with little recent rainfall. Plan irrigation in the next 3-5 days.")

    # --- Fertilizer / nutrients ---
    if row.get("nitrogen_pct", 0.05) < 0.03:
        tips.append("Soil nitrogen is below optimal range. Consider a top-dress nitrogen application.")
    if row.get("soil_pH", 6.5) < 6.0:
        tips.append("Soil pH is acidic (<6.0). Consider lime application to improve nutrient uptake.")
    elif row.get("soil_pH", 6.5) > 7.8:
        tips.append("Soil pH is alkaline (>7.8). Monitor for micronutrient (Fe, Zn) deficiency.")

    # --- Crop health / stress response ---
    stress = row.get("stress_level_predicted", row.get("stress_rule_based", 0))
    if stress == 2:
        tips.append("SEVERE stress detected from NDVI/NDRE trend. Recommend field inspection within 48 hours "
                     "to rule out pest/disease alongside water stress.")
    elif stress == 1:
        tips.append("Mild stress signals detected. Increase monitoring frequency (satellite + sensor) over next 2 weeks.")

    # --- Heat stress ---
    if row.get("temp_C", 25) > 32:
        tips.append("High temperature stress risk. If flowering/grain-fill stage, consider light irrigation to "
                     "reduce canopy temperature.")

    if not tips:
        tips.append("Conditions look normal. Continue standard monitoring schedule.")

    return " | ".join(tips)


def run_for_latest_week(weekly_stress_df):
    """Generate advisories for the most recent week per plot."""
    latest = weekly_stress_df.sort_values("week").groupby("plot_id").tail(1).copy()
    latest["advisory"] = latest.apply(generate_advisory, axis=1)
    return latest[["plot_id", "week", "NDVI", "soil_moisture_pct", "stress_rule_based", "advisory"]]


if __name__ == "__main__":
    weekly = pd.read_csv("../data/weekly_with_rule_based_stress.csv")
    advisories = run_for_latest_week(weekly)
    advisories.to_csv("../outputs/latest_advisories.csv", index=False)

    print("Sample advisories (5 plots):\n")
    for _, r in advisories.head(5).iterrows():
        print(f"[{r.plot_id}] Week {r.week} | NDVI={r.NDVI:.2f} | SoilMoist={r.soil_moisture_pct:.1f}% "
              f"| Stress={r.stress_rule_based}")
        print(f"   -> {r.advisory}\n")
