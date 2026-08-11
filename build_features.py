

import pandas as pd
import numpy as np


def build():
    ndvi = pd.read_csv("../data/sentinel2_ndvi_ndre.csv")
    weather = pd.read_csv("../data/nasa_power_weather.csv")
    soil = pd.read_csv("../data/soilgrids_properties.csv")
    iot = pd.read_csv("../data/iot_sensors.csv")
    yield_df = pd.read_csv("../data/yield_data.csv")

  
    weekly = ndvi.merge(weather, on=["plot_id", "week"]).merge(iot, on=["plot_id", "week"])
    weekly = weekly.merge(soil, on="plot_id") 
    weekly = weekly.sort_values(["plot_id", "week"]).reset_index(drop=True)

    weekly["NDVI_pct_change_2wk"] = weekly.groupby("plot_id")["NDVI"].pct_change(2)
    weekly["rainfall_7wk_cumsum"] = weekly.groupby("plot_id")["rainfall_mm"].cumsum()

    weekly.to_csv("../data/weekly_features.csv", index=False)
    season = weekly.groupby("plot_id").agg(
        NDVI_peak=("NDVI", "max"),
        NDVI_mean=("NDVI", "mean"),
        NDRE_peak=("NDRE", "max"),
        NDRE_mean=("NDRE", "mean"),
        temp_mean=("temp_C", "mean"),
        rainfall_total=("rainfall_mm", "sum"),
        humidity_mean=("humidity_pct", "mean"),
        solar_mean=("solar_rad_MJ_m2", "mean"),
        soil_moisture_mean=("soil_moisture_pct", "mean"),
        soil_moisture_min=("soil_moisture_pct", "min"),
        soil_pH=("soil_pH", "first"),
        organic_carbon_pct=("organic_carbon_pct", "first"),
        nitrogen_pct=("nitrogen_pct", "first"),
        clay_pct=("clay_pct", "first"),
    ).reset_index()

    season = season.merge(yield_df, on="plot_id")
    season.to_csv("../data/season_features.csv", index=False)

    print(f"weekly_features.csv : {weekly.shape}")
    print(f"season_features.csv : {season.shape}")
    return weekly, season


if __name__ == "__main__":
    build()
