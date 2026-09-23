"""
A naive baseline forecast, for honest comparison against the zero-shot
t0-alpha result — the same "does the sophisticated approach actually
beat doing nothing clever" discipline used throughout this portfolio.

With only 96 weeks of context (108 total minus a 12-week holdout),
this is exactly the cold-start scenario: not enough personal history
to fit a real classical model (SARIMA/Prophet need more data than
this to estimate seasonal/trend parameters reliably), so the fair
classical comparison is the simplest defensible baseline — last
observed value repeated forward — not a sophisticated model straining
against too little data.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

HORIZON = 12  # weeks held out
CONTEXT_LEN = 96  # weeks of context — matches the same convention used in tiny-timeseries-foundation-model, for a directly comparable setup

df = pd.read_csv("data/processed/weekly_heart_rate.csv", parse_dates=["week_ending"])
series = df["peak_heart_rate"].values

train, test = series[:-HORIZON], series[-HORIZON:]
context = train[-CONTEXT_LEN:]

# Naive baseline: repeat the last observed value for the whole horizon
naive_forecast = np.full(HORIZON, context[-1])

mape = mean_absolute_percentage_error(test, naive_forecast) * 100
rmse = np.sqrt(mean_squared_error(test, naive_forecast))

print(f"Naive baseline (last value repeated)")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.2f} bpm")
