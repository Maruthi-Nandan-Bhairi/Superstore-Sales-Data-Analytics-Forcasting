"""
Superstore Sales - Data Cleaning, Feature Engineering & Exploratory Analysis.

This module loads the generated Superstore dataset, performs data cleaning,
creates derived features, and produces a set of EDA visualizations and summary
statistics.

Run:
    python superstore_analysis.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force matplotlib to use a non-interactive backend when needed
plt.close("all")

DATA_PATH = os.path.join("data", "superstore.csv")
OUTPUT_DIR = "img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seaborn-like color palette
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]


def load_data(path=DATA_PATH):
    """Load raw data and fix types."""
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df


def clean_data(df):
    """Perform data cleaning steps."""
    df = df.copy()
    # Drop rows with missing essential values
    df = df.dropna(subset=["Sales", "Profit", "Order Date"])
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Ensure numeric columns are numeric
    for col in ["Sales", "Profit", "Quantity", "Discount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Sales", "Quantity", "Profit"])
    return df


def feature_engineering(df):
    """Create derived features useful for analysis and forecasting."""
    df = df.copy()
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Order Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Ship Time (days)"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, 0)
    # Order-month ordering for categorical plots
    order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    df["Order Month Name"] = pd.Categorical(df["Order Month Name"], categories=order, ordered=True)
    return df


def summary_statistics(df):
    """Print high-level descriptive statistics."""
    print("=" * 60)
    print("SUPERSTORE SALES - DATA OVERVIEW")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDate range: {df['Order Date'].min().date()} -> {df['Order Date'].max().date()}")
    print(f"\nTotal Sales: ${df['Sales'].sum():,.2f}")
    print(f"Total Profit: ${df['Profit'].sum():,.2f}")
    print(f"Overall Profit Margin: {df['Profit'].sum() / df['Sales'].sum() * 100:.2f}%")
    print(f"\nSales by Category:")
    print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False).to_string())
    print(f"\nProfit by Category:")
    print(df.groupby("Category")["Profit"].sum().sort_values(ascending=False).to_string())
    print(f"\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string() or "None")


def plot_top_categories(df):
    """Bar chart of sales & profit by category."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["Sales", "Profit"], ["Sales", "Profit"]):
        data = df.groupby("Category")[metric].sum().sort_values(ascending=False)
        ax.bar(data.index, data.values, color=PALETTE[: len(data)])
        ax.set_title(f"{title} by Category")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sales_by_category.png"))
    plt.show()


def plot_monthly_trend(df):
    """Monthly sales & profit trend."""
    monthly = df.groupby("YearMonth")[["Sales", "Profit"]].sum().reset_index()
    monthly["YearMonth"] = pd.to_datetime(monthly["YearMonth"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["YearMonth"], monthly["Sales"], marker="o", label="Sales", color=PALETTE[0])
    ax.plot(monthly["YearMonth"], monthly["Profit"], marker="s", label="Profit", color=PALETTE[1])
    ax.set_title("Monthly Sales & Profit Trend")
    ax.set_ylabel("Amount ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_trend.png"))
    plt.show()


def plot_sales_by_region(df):
    """Bar chart of sales by region."""
    region = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(region.index, region.values, color=PALETTE[: len(region)])
    ax.set_title("Total Sales by Region")
    ax.set_ylabel("Sales ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sales_by_region.png"))
    plt.show()


def plot_profit_by_subcategory(df):
    """Horizontal bar chart of profit by sub-category."""
    sub = df.groupby("Sub-Category")["Profit"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(sub.index, sub.values, color=PALETTE[0])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Profit by Sub-Category")
    ax.set_xlabel("Profit ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "profit_by_subcategory.png"))
    plt.show()


def plot_sales_quantity_scatter(df):
    """Scatter plot of quantity vs sales coloured by category."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = df["Category"].unique()
    for i, cat in enumerate(cats):
        sub = df[df["Category"] == cat]
        ax.scatter(sub["Quantity"], sub["Sales"], alpha=0.4, s=20, label=cat, color=PALETTE[i % len(PALETTE)])
    ax.set_title("Quantity vs Sales by Category")
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "quantity_vs_sales.png"))
    plt.show()


def plot_profit_margin_boxplot(df):
    """Boxplot of profit margin by category."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[df["Category"] == c]["Profit Margin"].values for c in df["Category"].unique()]
    ax.boxplot(data, tick_labels=df["Category"].unique(), patch_artist=True,
               boxprops=dict(facecolor=PALETTE[0], alpha=0.6))
    ax.set_title("Profit Margin Distribution by Category")
    ax.set_ylabel("Profit Margin (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "profit_margin_boxplot.png"))
    plt.show()


def main():
    print("Loading data...")
    df = load_data()
    print("Cleaning data...")
    df = clean_data(df)
    print("Engineering features...")
    df = feature_engineering(df)

    summary_statistics(df)

    print("\nGenerating visualizations...")
    plot_top_categories(df)
    plot_monthly_trend(df)
    plot_sales_by_region(df)
    plot_profit_by_subcategory(df)
    plot_sales_quantity_scatter(df)
    plot_profit_margin_boxplot(df)

    print(f"\nAll charts saved to '{OUTPUT_DIR}/'")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
