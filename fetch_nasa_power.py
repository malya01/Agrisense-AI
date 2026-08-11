

import requests
import pandas as pd
import time


def fetch_weather_for_plot(plot_id, lat, lon, start_date, end_date):
   
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()  # will raise an error if the request failed
    data = response.json()["properties"]["parameter"]
    dates = list(data["T2M"].keys())
    rows = []
    for date in dates:
        rows.append({
            "plot_id": plot_id,
            "date": date,
            "temp_C": data["T2M"][date],
            "rainfall_mm": data["PRECTOTCORR"][date],
            "humidity_pct": data["RH2M"][date],
            "solar_rad_MJ_m2": data["ALLSKY_SFC_SW_DWN"][date],
        })
    return pd.DataFrame(rows)


def fetch_weather_for_all_plots(plots_csv, start_date, end_date, out_path):
    """
    plots_csv: a CSV you provide with columns: plot_id, lat, lon
    (This is the ONE thing you must create yourself — your field GPS coordinates)
    """
    plots = pd.read_csv(plots_csv)
    all_rows = []

    for _, row in plots.iterrows():
        print(f"Fetching weather for {row['plot_id']} ...")
        df = fetch_weather_for_plot(row["plot_id"], row["lat"], row["lon"], start_date, end_date)
        all_rows.append(df)
        time.sleep(0.5) 
    result = pd.concat(all_rows, ignore_index=True)

    result["date"] = pd.to_datetime(result["date"], format="%Y%m%d")
    result["week"] = result.groupby("plot_id")["date"].transform(
        lambda d: ((d - d.min()).dt.days // 7) + 1
    )
    weekly = result.groupby(["plot_id", "week"]).agg(
        temp_C=("temp_C", "mean"),
        rainfall_mm=("rainfall_mm", "sum"),
        humidity_pct=("humidity_pct", "mean"),
        solar_rad_MJ_m2=("solar_rad_MJ_m2", "mean"),
    ).reset_index()

    weekly.to_csv(out_path, index=False)
    print(f"\nSaved real weather data to {out_path}")
    return weekly


if __name__ == "__main__":
   
    fetch_weather_for_all_plots(
        plots_csv="../data/plots.csv",
        start_date="20250101",
        end_date="20250325",
        out_path="../data/nasa_power_weather.csv",
    )
