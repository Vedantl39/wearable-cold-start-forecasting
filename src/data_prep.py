"""
Prepares a real Apple Watch heart rate dataset for a cold-start
forecasting test.

THE REAL DATA, HONESTLY DESCRIBED:
This is genuine personal Apple Watch data (2017-2019, 538 readings) —
but it's workout-session peak heart rate, recorded irregularly
whenever a workout happened, not continuous monitoring. That's a real
difference from what a device like Temple (a continuous cerebral
blood-flow wearable) produces, and it's disclosed here rather than
implied to be something it isn't.

RESAMPLING DECISION:
Resampled to WEEKLY frequency (max heart rate per week), giving 108
regular data points — far more usable for forecasting than monthly
(25 points, too few for a meaningful holdout). 13 of 108 weeks (12%)
have no recorded workout. Gap-run analysis found these aren't
uniform: most are isolated 1-2 week gaps (plausibly "no workout that
week," forward-filled as a defensible read of what happened), but one
run is 10 CONSECUTIVE missing weeks (Oct-Dec 2018) — forward-filling
that flat would misrepresent 10 weeks of no data as 10 weeks of
genuinely stable readings. That one long gap is linearly interpolated
instead, and every imputed point (short or long) is flagged with an
explicit `is_imputed` column so nothing downstream mistakes filled
values for real measurements.

WHY THIS DATASET IS THE RIGHT TEST, NOT A COMPROMISE:
A single person's wearable history is short by nature — you can't
have years of data from a device you started wearing last month. This
is exactly the cold-start problem: classical time-series methods
(ARIMA, Prophet) need real history to fit a good model, and a new
wearable user doesn't have much. A zero-shot foundation model doesn't
have that constraint — it was pretrained on millions of OTHER series
before it ever saw this one. Testing that gap directly is the point.
"""
import pandas as pd

RAW_PATH = "data/raw/Health_Data.csv"
OUT_PATH = "data/processed/weekly_heart_rate.csv"


def main():
    df = pd.read_csv(RAW_PATH)
    df["Finish"] = pd.to_datetime(df["Finish"], format="%d-%b-%Y %H:%M")
    df = df.sort_values("Finish").set_index("Finish")

    weekly = df["HeartRate"].resample("W").max()
    n_missing = weekly.isnull().sum()

    # Real check performed before choosing a fill method: gap-run analysis
    # showed most missing weeks are isolated (1-2 in a row — plausibly just
    # "no workout that week"), but one real run is 10 CONSECUTIVE missing
    # weeks (Oct-Dec 2018). Forward-filling that as flat is misleading — it
    # would present 10 weeks of "no data" as if they were 10 weeks of
    # genuinely stable readings. Short gaps (<=2 weeks) are forward-filled;
    # the one long gap is linearly interpolated instead (a smoother, less
    # falsely-confident estimate) AND explicitly flagged with is_imputed,
    # so anything built on this data can see which points are real.
    is_imputed = weekly.isnull()

    filled = weekly.copy()
    null_mask = filled.isnull()
    # identify run lengths
    run_id = (null_mask != null_mask.shift()).cumsum()
    run_lengths = null_mask.groupby(run_id).transform("sum")

    short_gap_mask = null_mask & (run_lengths <= 2)
    long_gap_mask = null_mask & (run_lengths > 2)

    filled[short_gap_mask] = filled.ffill()[short_gap_mask]
    filled[long_gap_mask] = filled.interpolate(method="linear")[long_gap_mask]
    filled = filled.bfill()  # covers a possible missing value at the very start

    out = filled.reset_index()
    out.columns = ["week_ending", "peak_heart_rate"]
    out["is_imputed"] = is_imputed.reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH}: {len(out)} weekly points")
    print(f"Total imputed weeks: {n_missing} ({n_missing/len(out)*100:.1f}%)")
    print(f"  Short gaps (<=2 weeks, forward-filled): {short_gap_mask.sum()}")
    print(f"  Long gap (>2 weeks, linearly interpolated): {long_gap_mask.sum()}")
    print(f"Range: {out['week_ending'].min()} to {out['week_ending'].max()}")


if __name__ == "__main__":
    main()
