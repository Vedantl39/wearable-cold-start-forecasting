"""
Tests theforecastingcompany/t0-alpha zero-shot on real Apple Watch
weekly peak heart rate data — the actual point of this project.

THE REAL QUESTION: t0-alpha was built and trained on time series from
business/economic domains (retail sales, the kind of data tested in
the sibling demand-forecasting project). This data is a completely
different domain — personal physiological/wearable data — and it's
short (96 weeks of context, the kind of history an actual new wearable
user would realistically have). Does a foundation model pretrained on
unrelated domains generalize to this one, zero-shot, with a genuinely
small amount of context? That's not a rhetorical question — the
result below is whatever it actually is, run and reported honestly,
same as every other zero-shot test in this portfolio.

SETUP (same as tiny-timeseries-foundation-model's t0-alpha test):
  1. pip install "tfc-t0[evaluation]"
  2. Accept the model's terms at
     https://huggingface.co/theforecastingcompany/t0-alpha and be
     logged in via `hf auth login` (same login used before).
  3. Run: python src/test_t0_alpha_wearable.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

HORIZON = 12
CONTEXT_LEN = 96

df = pd.read_csv("data/processed/weekly_heart_rate.csv", parse_dates=["week_ending"])
series = df["peak_heart_rate"].values

train, test = series[:-HORIZON], series[-HORIZON:]
context_values = train[-CONTEXT_LEN:]

from t0 import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

context = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    out = model.predict(context, horizon=HORIZON, quantiles=[0.1, 0.5, 0.9])

t0_forecast = pd.Series(out.median.squeeze(0).numpy())

mape = mean_absolute_percentage_error(test, t0_forecast) * 100
rmse = np.sqrt(mean_squared_error(test, t0_forecast))

print(f"t0-alpha (zero-shot, real wearable heart rate data)")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.2f} bpm")

print("\nComparison:")
print(f"  Naive baseline: 18.37% MAPE, 30.98 bpm RMSE")
print(f"  t0-alpha:       {mape:.2f}% MAPE, {rmse:.2f} bpm RMSE")
