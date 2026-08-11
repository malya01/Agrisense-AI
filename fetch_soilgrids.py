"""
fetch_soilgrids.py
--------------------
Fetches REAL soil data from ISRIC SoilGrids (free, no signup/API key needed).
Replaces the simulated generate_soilgrids() function.

Usage:
    python3 fetch_soilgrids.py
"""

import requests
import pandas as pd
import time


def fetch_soil_for_plot(plot_id, lat, lon):
    """
    Queries SoilGrids for pH, organic carbon, nitrogen, and clay %
    at the standard 0-5cm topsoil depth (most relevant for crop roots/early growth).
    """
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["phh2o", "soc", "nitrogen", "clay"],
        "depth": "0-5cm",
        "value": "mean",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Response structure: properties -> layers -> [ {name, depths: [{values: {mean: X}}]} ]
    layers = {layer["name"]: layer for layer in data["properties"]["layers"]}

    def get_value(prop_name, divisor=1):
        try:
            raw = layers[prop_name]["depths"][0]["values"]["mean"]
            return raw / divisor if raw is not None else None
        except (KeyError, IndexError, TypeError):
            return None

    return {
        "plot_id": plot_id,
        # SoilGrids returns pH*10, organic carbon in dg/kg, nitrogen in cg/kg -- divisors convert to normal units
        "soil_pH": get_value("phh2o", divisor=10),
        "organic_carbon_pct": get_value("soc", divisor=100),   # dg/kg -> %
        "nitrogen_pct": get_value("nitrogen", divisor=1000),   # cg/kg -> %
        "clay_pct": get_value("clay", divisor=10),              # g/kg -> %
    }


def fetch_soil_for_all_plots(plots_csv, out_path):
    plots = pd.read_csv(plots_csv)
    rows = []

    for _, row in plots.iterrows():
        print(f"Fetching soil data for {row['plot_id']} ...")
        rows.append(fetch_soil_for_plot(row["plot_id"], row["lat"], row["lon"]))
        time.sleep(0.5)  # be polite to the free API

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"\nSaved real soil data to {out_path}")
    return result


if __name__ == "__main__":
    # Uses the SAME data/plots.csv you created for the weather script
    fetch_soil_for_all_plots(
        plots_csv="../data/plots.csv",
        out_path="../data/soilgrids_properties.csv",
    )
