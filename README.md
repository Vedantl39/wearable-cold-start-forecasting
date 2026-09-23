# Wearable Health Forecasting: The Cold-Start Problem

Testing whether a zero-shot time-series foundation model (t0-alpha, from [The Forecasting Company](https://theforecastingcompany.com)) can forecast something it almost certainly never saw in training — real personal wearable heart-rate data — with only a short, genuinely realistic amount of history.

## Why this project

Deepinder Goyal (Zomato/Eternal co-founder) is building [Temple](https://en.wikipedia.org/wiki/Deepinder_Goyal), a wearable that continuously tracks cerebral blood flow, motivated by what he calls the "Gravity Ageing Hypothesis." Temple isn't publicly available, so this project doesn't use real Temple data — it uses real, public Apple Watch heart-rate data instead, and asks the question a device like Temple would actually face on day one with a new user: **you don't have years of history to build a model from. What forecasts a wearable signal well when there's almost nothing to learn from yet?**

That's the cold-start problem, and it's exactly the scenario zero-shot foundation models like t0-alpha are built for — pretrained on millions of *other* series before they ever see yours, so a new user's short history isn't a blocker the way it is for a classical per-series model.

## The data — real, and honestly limited

Real Apple Watch data, 2017-2019, 538 individual heart-rate readings. Two things disclosed upfront rather than glossed over:

- **This is workout-session peak heart rate, not continuous monitoring.** A device like Temple tracks continuously; this data is recorded only when a workout happens. Real, but a different signal shape than what Temple itself would produce.
- **Resampled to weekly (108 points)**, since monthly resampling left only 25 points — too few for any meaningful holdout. 13 of 108 weeks (12%) had no recorded workout. Gap-run analysis found these weren't uniform: most were isolated 1-2 week gaps (forward-filled, a defensible read of "no workout that week"), but one real gap ran 10 consecutive weeks (Oct-Dec 2018) — forward-filling that flat would have misrepresented 10 weeks of no data as 10 weeks of genuine stability. That one gap is linearly interpolated instead, and **every imputed point is flagged explicitly** (`is_imputed` column) — see `reports/weekly_hr_series.png`, where imputed weeks are marked in red, not blended invisibly into the real series.

## Setup

- **Context:** 96 weeks (the same convention used in `tiny-timeseries-foundation-model`, for a directly comparable setup)
- **Holdout:** the final 12 weeks
- **Naive baseline:** last observed value repeated forward — the fair classical comparison here, since 96 weeks isn't enough history to fit a real SARIMA/Prophet model reliably. That's not a workaround, it's the actual point: this is what "not enough data for a classical model" looks like in practice.

## Result

| Model | MAPE | RMSE |
|---|---|---|
| Naive baseline (last value repeated) | 18.37% | 30.98 bpm |
| t0-alpha (zero-shot) | *(run `src/test_t0_alpha_wearable.py` locally — requires Hugging Face access, see below)* | |

This project's t0-alpha result wasn't run inside the environment this was built in (no access to Hugging Face's gated model download there — same limitation as the original demand-forecasting t0-alpha test). The script is written, verified for correct syntax, and ready to run.

## Repository structure

```
wearable-cold-start-forecasting/
├── data/
│   ├── raw/Health_Data.csv              # real Apple Watch data
│   └── processed/weekly_heart_rate.csv  # cleaned, resampled, imputation-flagged
├── src/
│   ├── data_prep.py                     # resampling + gap-run-aware imputation
│   ├── visualize.py                     # plots real vs. imputed points, clearly marked
│   ├── naive_baseline.py                # classical comparison baseline
│   └── test_t0_alpha_wearable.py        # zero-shot t0-alpha test (run locally)
├── reports/
│   └── weekly_hr_series.png             # real vs. imputed points clearly marked
└── requirements.txt
```

## Running this project

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/visualize.py
python src/naive_baseline.py

# For the t0-alpha result (requires HF login, see tiny-timeseries-foundation-model's README for the same setup steps):
pip install "tfc-t0[evaluation]"
python src/test_t0_alpha_wearable.py
```

## Data source

Real Apple Watch HealthKit export, via [leehaesung/My_Heart_Rate_For_Time_Series_Analysis](https://github.com/leehaesung/My_Heart_Rate_For_Time_Series_Analysis) (public GitHub repo).

## Author

Vedant Limaye
