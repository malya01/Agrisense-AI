"""
generate_data.py
-----------------
Simulates the 5 real-world datasets for the project, using the SAME schema,
units, and value ranges that the real APIs/datasets would return.

WHY SIMULATED: This sandbox has no network access to Earth Engine, NASA POWER,
SoilGrids, or Kaggle. In production, replace each generator function below
with the real API call marked "REPLACE WITH REAL API".

Scope chosen (per the plan): Wheat crop, Rabi season, 40 field plots across
a district, one growing season with 12 weekly timesteps (~84 days, typical
wheat cycle).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PLOTS = 40
N_WEEKS = 12
PLOT_IDS = [f"PLOT_{i:03d}" for i in range(1, N_PLOTS + 1)]

# Each plot gets a fixed lat/long inside a sample district bounding box
# (e.g., a wheat belt district in Madhya Pradesh: ~22.7N, 75.8E)
plot_coords = {
    pid: (22.5 + np.random.uniform(-0.3, 0.3), 75.6 + np.random.uniform(-0.3, 0.3))
    for pid in PLOT_IDS
}

# Each plot is randomly assigned a "true stress profile" that drives all
# downstream signals consistently (so the dataset is internally coherent,
# not just random noise) -> 0 = healthy, 1 = mild stress, 2 = severe stress
plot_stress_profile = {
    pid: np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]) for pid in PLOT_IDS
}


def generate_sentinel2_ndvi_ndre():
    """
    REPLACE WITH REAL API: Google Earth Engine Python API pulling
    COPERNICUS/S2_SR_HARMONIZED, computing:
        NDVI = (B8 - B4) / (B8 + B4)
        NDRE = (B8 - B5) / (B8 + B5)
    averaged over each plot polygon, once per week.
    """
    rows = []
    for pid in PLOT_IDS:
        stress = plot_stress_profile[pid]
        # Healthy wheat NDVI arcs from ~0.3 (sowing) to ~0.85 (peak) back to ~0.4 (senescence)
        base_curve = 0.3 + 0.55 * np.sin(np.linspace(0.1, np.pi - 0.1, N_WEEKS))
        # stress lowers peak and causes earlier decline
        penalty = stress * 0.12
        ndvi = base_curve - penalty + np.random.normal(0, 0.02, N_WEEKS)
        ndvi = np.clip(ndvi, 0.05, 0.92)
        ndre = ndvi * np.random.uniform(0.55, 0.7) + np.random.normal(0, 0.015, N_WEEKS)
        ndre = np.clip(ndre, 0.03, 0.6)
        for week in range(N_WEEKS):
            rows.append({
                "plot_id": pid, "week": week + 1,
                "lat": plot_coords[pid][0], "lon": plot_coords[pid][1],
                "NDVI": round(ndvi[week], 4), "NDRE": round(ndre[week], 4),
            })
    return pd.DataFrame(rows)


def generate_nasa_power_weather():
    """
    REPLACE WITH REAL API: NASA POWER API
    https://power.larc.nasa.gov/api/temporal/daily/point
    Parameters: T2M (temp), PRECTOTCORR (rainfall), RH2M (humidity), ALLSKY_SFC_SW_DWN (solar radiation)
    Aggregate daily -> weekly per plot coordinate.
    """
    rows = []
    for pid in PLOT_IDS:
        stress = plot_stress_profile[pid]
        # stressed plots get a simulated dry/hot spell mid-season
        base_temp = 22 + 6 * np.sin(np.linspace(0, np.pi, N_WEEKS)) + np.random.normal(0, 1, N_WEEKS)
        base_rain = np.random.gamma(2, 3, N_WEEKS)
        if stress >= 1:
            dry_weeks = slice(4, 8)
            base_rain[dry_weeks] *= 0.25
            base_temp[dry_weeks] += stress * 2.5
        humidity = np.clip(70 - (base_temp - 22) * 1.5 + np.random.normal(0, 4, N_WEEKS), 25, 95)
        solar = np.clip(18 + np.random.normal(0, 2, N_WEEKS), 10, 28)
        for week in range(N_WEEKS):
            rows.append({
                "plot_id": pid, "week": week + 1,
                "temp_C": round(base_temp[week], 2),
                "rainfall_mm": round(max(base_rain[week], 0), 2),
                "humidity_pct": round(humidity[week], 2),
                "solar_rad_MJ_m2": round(solar[week], 2),
            })
    return pd.DataFrame(rows)


def generate_soilgrids():
    """
    REPLACE WITH REAL API: ISRIC SoilGrids REST API
    https://rest.isric.org/soilgrids/v2.0/properties/query?lon=..&lat=..
    Properties: phh2o, soc (organic carbon), nitrogen, cec, sand/silt/clay.
    Soil properties are static for a season, so one row per plot.
    """
    rows = []
    for pid in PLOT_IDS:
        stress = plot_stress_profile[pid]
        ph = np.random.normal(6.8, 0.4) - (0.15 if stress == 2 else 0)
        organic_carbon = np.clip(np.random.normal(0.55, 0.15) - stress * 0.05, 0.1, 1.2)
        nitrogen = np.clip(np.random.normal(0.045, 0.01) - stress * 0.004, 0.01, 0.08)
        clay_pct = np.random.uniform(15, 40)
        rows.append({
            "plot_id": pid, "soil_pH": round(ph, 2),
            "organic_carbon_pct": round(organic_carbon, 3),
            "nitrogen_pct": round(nitrogen, 4),
            "clay_pct": round(clay_pct, 1),
        })
    return pd.DataFrame(rows)


def generate_iot_sensors():
    """
    REPLACE WITH REAL SOURCE: Kaggle/UCI "Smart Agriculture IoT" dataset,
    OR live sensor feed (soil moisture probes, DHT22 temp/humidity nodes)
    ingested via MQTT -> a time-series DB. Weekly-averaged here.
    """
    rows = []
    for pid in PLOT_IDS:
        stress = plot_stress_profile[pid]
        base_moisture = 45 - stress * 8 + np.random.normal(0, 3, N_WEEKS)
        if stress >= 1:
            base_moisture[4:8] -= stress * 6  # dry spell matches weather module
        soil_moisture = np.clip(base_moisture, 5, 65)
        soil_temp = np.clip(20 + np.random.normal(0, 2, N_WEEKS), 12, 32)
        for week in range(N_WEEKS):
            rows.append({
                "plot_id": pid, "week": week + 1,
                "soil_moisture_pct": round(soil_moisture[week], 2),
                "soil_temp_C": round(soil_temp[week], 2),
            })
    return pd.DataFrame(rows)


def generate_yield_data(ndvi_df, weather_df, soil_df, iot_df):
    """
    REPLACE WITH REAL SOURCE: India Crop Yield Dataset (data.gov.in / Kaggle),
    typically district/season-wise actual reported yield (tonnes/hectare).
    Here we derive a synthetic-but-physically-plausible yield from the other
    modules so the ML pipeline has genuine signal to learn (not pure noise).
    """
    rows = []
    for pid in PLOT_IDS:
        stress = plot_stress_profile[pid]
        peak_ndvi = ndvi_df[ndvi_df.plot_id == pid].NDVI.max()
        avg_rain = weather_df[weather_df.plot_id == pid].rainfall_mm.mean()
        avg_moisture = iot_df[iot_df.plot_id == pid].soil_moisture_pct.mean()
        oc = soil_df[soil_df.plot_id == pid].organic_carbon_pct.values[0]

        base_yield = 3.0  # tonnes/hectare, typical wheat baseline
        yield_t_ha = (
            base_yield
            + 2.2 * (peak_ndvi - 0.5)
            + 0.015 * (avg_rain - 8)
            + 0.02 * (avg_moisture - 35)
            + 0.8 * (oc - 0.5)
            - stress * 0.35
            + np.random.normal(0, 0.15)
        )
        rows.append({
            "plot_id": pid,
            "stress_level_true": stress,  # kept for validation, not used as a feature
            "yield_tonnes_per_ha": round(max(yield_t_ha, 0.3), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ndvi = generate_sentinel2_ndvi_ndre()
    weather = generate_nasa_power_weather()
    soil = generate_soilgrids()
    iot = generate_iot_sensors()
    yield_df = generate_yield_data(ndvi, weather, soil, iot)

    ndvi.to_csv("../data/sentinel2_ndvi_ndre.csv", index=False)
    weather.to_csv("../data/nasa_power_weather.csv", index=False)
    soil.to_csv("../data/soilgrids_properties.csv", index=False)
    iot.to_csv("../data/iot_sensors.csv", index=False)
    yield_df.to_csv("../data/yield_data.csv", index=False)

    print("Generated datasets:")
    print(f"  Sentinel-2 NDVI/NDRE : {ndvi.shape}")
    print(f"  NASA POWER weather   : {weather.shape}")
    print(f"  SoilGrids properties : {soil.shape}")
    print(f"  IoT sensors          : {iot.shape}")
    print(f"  Yield data           : {yield_df.shape}")
