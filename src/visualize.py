"""
Plots the weekly heart rate series with imputed points clearly marked
— not blended invisibly into the real data.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/weekly_heart_rate.csv", parse_dates=["week_ending"])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["week_ending"], df["peak_heart_rate"], color="#4c72b0", linewidth=1.5, zorder=1)

real = df[~df["is_imputed"]]
imputed = df[df["is_imputed"]]
ax.scatter(real["week_ending"], real["peak_heart_rate"], color="#4c72b0", s=15, zorder=2, label="Real reading")
ax.scatter(imputed["week_ending"], imputed["peak_heart_rate"], color="red", s=25, zorder=3, label="Imputed (no workout recorded)")

ax.set_title("Weekly peak heart rate (real Apple Watch data, 2017-2019) — imputed weeks marked")
ax.set_ylabel("Peak HR (bpm)")
ax.legend()
plt.tight_layout()
plt.savefig("reports/weekly_hr_series.png", dpi=100)
print("Saved reports/weekly_hr_series.png")
