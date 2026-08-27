"""Interactive Streamlit dashboard for the Superstore sales project."""

from pathlib import Path

import pandas as pd
import streamlit as st

from superstore_analysis import clean_data, feature_engineering
from superstore_forecast import build_monthly_series, forecast_series


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "superstore.csv"
IMAGE_DIR = PROJECT_DIR / "img"

st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: #f4f7fb;
        border-left: 4px solid #2463eb;
        padding: 14px 16px;
        border-radius: 6px;
    }
    [data-testid="stImage"] img { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dashboard_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Upload data/superstore.csv to the same GitHub repository "
            "as app.py, then reboot the Streamlit app."
        )
    data = pd.read_csv(DATA_PATH)
    data["Order Date"] = pd.to_datetime(data["Order Date"])
    data["Ship Date"] = pd.to_datetime(data["Ship Date"])
    return feature_engineering(clean_data(data))


def money(value):
    return f"${value:,.0f}"


st.title("Superstore Sales Dashboard")
st.caption("Explore sales performance, profitability, and the 12-month forecast.")

try:
    df = load_dashboard_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Filters")
    categories = st.multiselect(
        "Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique())
    )
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    date_range = st.date_input(
        "Order date range",
        value=(df["Order Date"].min().date(), df["Order Date"].max().date()),
        min_value=df["Order Date"].min().date(),
        max_value=df["Order Date"].max().date(),
    )

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date = end_date = pd.to_datetime(date_range[0])

filtered = df[
    df["Category"].isin(categories)
    & df["Region"].isin(regions)
    & df["Order Date"].between(start_date, end_date)
]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

sales, profit = filtered["Sales"].sum(), filtered["Profit"].sum()
profit_margin = profit / sales * 100 if sales else 0

metric_columns = st.columns(4)
metric_columns[0].metric("Total Sales", money(sales))
metric_columns[1].metric("Total Profit", money(profit))
metric_columns[2].metric("Profit Margin", f"{profit_margin:.1f}%")
metric_columns[3].metric("Orders", f"{filtered['Order ID'].nunique():,}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Sales and profit by category")
    category_summary = filtered.groupby("Category")[["Sales", "Profit"]].sum().sort_values("Sales")
    st.bar_chart(category_summary)
with right:
    st.subheader("Monthly performance")
    monthly_summary = filtered.set_index("Order Date").resample("MS")[["Sales", "Profit"]].sum()
    st.line_chart(monthly_summary)

left, right = st.columns(2)
with left:
    st.subheader("Sales by region")
    st.bar_chart(filtered.groupby("Region")["Sales"].sum().sort_values())
with right:
    st.subheader("Profit by sub-category")
    st.bar_chart(filtered.groupby("Sub-Category")["Profit"].sum().sort_values())

st.subheader("Forecast")
forecast_tabs = st.tabs(["Sales forecast", "Profit forecast"])
for tab, metric in zip(forecast_tabs, ["Sales", "Profit"]):
    with tab:
        series = build_monthly_series(filtered, metric)
        forecast = forecast_series(series, steps=12)
        future_dates = pd.date_range(
            start=series.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS"
        )
        history_frame = series.rename("Historical").to_frame()
        forecast_frame = pd.Series(forecast, index=future_dates, name="Forecast").to_frame()
        st.line_chart(pd.concat([history_frame, forecast_frame], axis=1))
        st.caption("Forecast combines simple exponential smoothing and a linear trend model.")

st.divider()
st.subheader("Generated analysis images")
st.caption("These are the charts produced by the command-line analysis pipeline.")
image_specs = [
    ("sales_by_category.png", "Sales and profit by category"),
    ("monthly_trend.png", "Monthly sales and profit trend"),
    ("sales_by_region.png", "Sales by region"),
    ("profit_by_subcategory.png", "Profit by sub-category"),
    ("quantity_vs_sales.png", "Quantity versus sales"),
    ("profit_margin_boxplot.png", "Profit margin distribution"),
    ("forecast_sales.png", "Sales forecast"),
    ("forecast_profit.png", "Profit forecast"),
]
for row_start in range(0, len(image_specs), 2):
    columns = st.columns(2)
    for column, (filename, caption) in zip(columns, image_specs[row_start : row_start + 2]):
        image_path = IMAGE_DIR / filename
        with column:
            if image_path.is_file():
                st.image(str(image_path), caption=caption, use_container_width=True)
            else:
                st.info(f"Image not found: {image_path}")