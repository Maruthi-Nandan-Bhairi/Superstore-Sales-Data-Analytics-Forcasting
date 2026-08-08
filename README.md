# Superstore Sales — Data Analytics & Forecasting

A self-contained data analytics & data science project that performs end-to-end
analysis of Superstore sales data: **data generation → cleaning → feature
engineering → exploratory data analysis (EDA) → time-series forecasting**.

## Project structure
```
superstore-sales-project/
├── data/
│   ├── generate_sample_data.py   # generates a sample Superstore dataset
│   └── superstore.csv            # generated dataset (7,500+ rows)
├── superstore_analysis.py        # cleaning, feature engineering, EDA + charts
├── superstore_forecast.py        # monthly sales & profit time-series forecasting
├── img/                          # generated charts
├── requirements.txt
└── TODO.md
```

## How to run
```bash
# 1. (Optional) (Re)generate the sample dataset
python data/generate_sample_data.py

# 2. Run exploratory analysis (prints stats, creates charts in img/)
python superstore_analysis.py

# 3. Run forecasting (prints forecasts + error metrics, creates charts)
python superstore_forecast.py
```

## What it does
- **Data cleaning**: drops missing/duplicate rows, coerces numeric types.
- **Feature engineering**: adds year, month, quarter, year-month, ship time, profit margin.
- **EDA visualizations**:
  - Sales & profit by category
  - Monthly sales/profit trend
  - Sales by region
  - Profit by sub-category
  - Quantity vs sales scatter (colored by category)
  - Profit margin distribution boxplot
- **Forecasting**: aggregates sales/profit to a monthly time series, fits simple
  exponential smoothing and linear trend models (averaged), evaluates on a 6-month
  hold-out (MAPE / RMSE), and produces a 12-month outlook.

## Dependencies
`numpy`, `pandas`, `matplotlib`, `scipy` (see `requirements.txt`).

