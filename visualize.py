import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110

weekly = pd.read_csv("../data/weekly_with_rule_based_stress.csv")
season = pd.read_csv("../data/season_features.csv")
yield_df = pd.read_csv("../data/yield_data.csv")

# ---------- 1. NDVI trajectories colored by true stress level ----------
fig, ax = plt.subplots(figsize=(8, 5))
colors = {0: "#2e8b57", 1: "#e6a817", 2: "#c0392b"}
labels = {0: "Healthy", 1: "Mild stress", 2: "Severe stress"}
plotted_labels = set()
merged = weekly.merge(yield_df[["plot_id", "stress_level_true"]], on="plot_id")
for pid, g in merged.groupby("plot_id"):
    s = g["stress_level_true"].iloc[0]
    lbl = labels[s] if s not in plotted_labels else None
    plotted_labels.add(s)
    ax.plot(g["week"], g["NDVI"], color=colors[s], alpha=0.6, linewidth=1.3, label=lbl)
ax.set_xlabel("Week of growing season")
ax.set_ylabel("NDVI")
ax.set_title("Simulated Sentinel-2 NDVI Trajectories by Stress Class (40 plots)")
ax.legend()
fig.tight_layout()
fig.savefig("../outputs/1_ndvi_trajectories.png")
plt.close(fig)

# ---------- 2. Yield vs NDVI_mean scatter ----------
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(season["NDVI_mean"], season["yield_tonnes_per_ha"],
                 c=season["soil_moisture_mean"], cmap="YlGn", s=70, edgecolor="k")
ax.set_xlabel("Season-mean NDVI")
ax.set_ylabel("Yield (tonnes/ha)")
ax.set_title("Yield vs NDVI (color = mean soil moisture %)")
plt.colorbar(sc, label="Soil moisture (%)")
fig.tight_layout()
fig.savefig("../outputs/2_yield_vs_ndvi.png")
plt.close(fig)

# ---------- 3. Stress class distribution: rule-based vs ML-derived ground truth ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
rule_counts = weekly["stress_rule_based"].value_counts().sort_index()
ax.bar(["Healthy", "Mild", "Severe"], rule_counts.values, color=["#2e8b57", "#e6a817", "#c0392b"])
ax.set_title("Rule-Based Stress Flags Across All Plot-Weeks")
ax.set_ylabel("Count of plot-weeks")
fig.tight_layout()
fig.savefig("../outputs/3_stress_distribution.png")
plt.close(fig)

# ---------- 4. Model comparison bar chart ----------
from yield_prediction import run as run_yield
results, best_name, importances, _ = run_yield()

fig, ax = plt.subplots(figsize=(7, 5))
models = list(results.keys())
r2_scores = [results[m]["CV_R2_mean"] for m in models]
mae_scores = [results[m]["MAE"] for m in models]
x = np.arange(len(models))
ax2 = ax.twinx()
ax.bar(x - 0.15, r2_scores, width=0.3, label="5-fold CV R2", color="#4c72b0")
ax2.bar(x + 0.15, mae_scores, width=0.3, label="MAE (tonnes/ha)", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("CV R2 (higher better)")
ax2.set_ylabel("MAE (lower better)")
ax.set_title(f"Yield Model Comparison (Best: {best_name})")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, loc="upper center")
fig.tight_layout()
fig.savefig("../outputs/4_model_comparison.png")
plt.close(fig)

# ---------- 5. Feature importance for best yield model ----------
fig, ax = plt.subplots(figsize=(7, 5))
importances.head(8).sort_values().plot(kind="barh", ax=ax, color="#55a868")
ax.set_title(f"Top Feature Importances - {best_name} (Yield Model)")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig("../outputs/5_feature_importance.png")
plt.close(fig)

print("Saved 5 plots to ../outputs/")
