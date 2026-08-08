"""
Superstore Sales - Time-Series Forecasting.

This module aggregates the Superstore sales data to a monthly time series and
forecasts future sales/profit using a simple, dependency-light approach
(exponential smoothing and a linear trend model). It compares the forecast
against a hold-out set and reports the error.

Run:
    python superstore_forecast.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superstore_analysis import clean_data, feature_engineering, load_data

warnings.filterwarnings("ignore")

OUTPUT_DIR = "img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868"]


def build_monthly_series(df, metric="Sales"):
    """Aggregate to a monthly time series indexed by YearMonth."""
    monthly = (
        df.groupby("YearMonth")[metric]
        .sum()
        .reset_index()
        .rename(columns={metric: "value"})
    )
    monthly["date"] = pd.to_datetime(monthly["YearMonth"])
    monthly = monthly.set_index("date")["value"].sort_index()
    return monthly


def simple_exp_smoothing(series, alpha=0.3, steps=12):
    """Forecast using simple exponential smoothing."""
    last = series.iloc[0]
    for value in series.iloc[1:]:
        last = alpha * value + (1 - alpha) * last
    forecast = [last] * steps
    return forecast


def linear_trend_forecast(series, steps=12):
    """Fit a linear trend and forecast beyond the data."""
    x = np.arange(len(series))
    # Least squares fit
    coeffs = np.polyfit(x, series.values, 1)
    future_x = np.arange(len(series), len(series) + steps)
    return np.polyval(coeffs, future_x)


def forecast_series(series, steps=12):
    """Return a combined forecast (average of two simple models)."""
    es = simple_exp_smoothing(series, steps=steps)
    lt = linear_trend_forecast(series, steps=steps)
    return (np.array(es) + np.array(lt)) / 2


def evaluate_forecast(series, steps=6):
    """Hold-out evaluation: forecast last `steps` and compute MAPE."""
    train = series.iloc[:-steps]
    test = series.iloc[-steps:]

    fc = forecast_series(train, steps=steps)
    fc = np.array(fc)

    # Avoid division by zero
    mask = test.values != 0
    mape = np.mean(np.abs((test.values[mask] - fc[mask]) / test.values[mask])) * 100
    rmse = np.sqrt(np.mean((test.values - fc) ** 2))
    return mape, rmse, fc, test


def plot_forecast(series, forecast, metric, steps=12):
    """Plot historical series plus forecast."""
    last_date = series.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=steps, freq="MS"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, marker="o", label="Historical", color=PALETTE[0])
    plt.plot(future_index, forecast, marker="s", linestyle="--", label="Forecast", color=PALETTE[1])
    plt.axvline(last_date, color="gray", linestyle=":", alpha=0.7)
    plt.title(f"{metric} Forecast (next {steps} months)")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"forecast_{metric.lower()}.png"))
    plt.show()


def main():
    print("Loading & preparing data...")
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    for metric in ["Sales", "Profit"]:
        series = build_monthly_series(df, metric)
        print(f"\n{'='*60}")
        print(f"{metric.upper()} FORECAST")
        print(f"{'='*60}")
        print(f"Historical months: {len(series)}")
        print(f"Last 6 months of data:")
        print(series.tail(6).to_string())

        # Hold-out evaluation
        mape, rmse, fc, test = evaluate_forecast(series, steps=6)
        print(f"\nHold-out evaluation (last 6 months):")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: ${rmse:,.2f}")

        # Future forecast (12 months ahead)
        steps = 12
        forecast = forecast_series(series, steps=steps)
        print(f"\nForecast for next {steps} months:")
        future_index = pd.date_range(
            start=series.index[-1] + pd.DateOffset(months=1), periods=steps, freq="MS"
        )
        for dt, val in zip(future_index, forecast):
            print(f"  {dt.strftime('%Y-%m')}: ${val:,.2f}")

        plot_forecast(series, forecast, metric, steps=steps)

    print("\nForecasting complete. Charts saved to 'img/'.")


if __name__ == "__main__":
    main()
