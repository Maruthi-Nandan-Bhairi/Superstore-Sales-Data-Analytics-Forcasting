"""
Generate a realistic sample Superstore sales dataset.

This script produces a CSV file with orders that include product category,
sub-category, region, segment, sales, quantity, profit, discount, and order / ship
dates. It is designed to run fully offline so the project is self-contained.

Output: data/superstore.csv
"""

import os
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "superstore.csv")

N_ORDERS = 5000
START_DATE = pd.Timestamp("2014-01-01")
END_DATE = pd.Timestamp("2017-12-31")

CATEGORIES = {
    "Technology": ["Accessories", "Copiers", "Machines", "Phones"],
    "Furniture": ["Bookcases", "Chairs", "Furnishings", "Tables"],
    "Office Supplies": [
        "Appliances",
        "Art",
        "Binders",
        "Envelopes",
        "Fasteners",
        "Labels",
        "Paper",
        "Storage",
        "Supplies",
    ],
}

SEGMENTS = ["Consumer", "Corporate", "Home Office"]
REGIONS = ["West", "East", "Central", "South"]

# Approximate base price ranges per sub-category (for realistic sales values)
SUB_PRICE = {
    "Accessories": (10, 400),
    "Copiers": (200, 2000),
    "Machines": (150, 1500),
    "Phones": (50, 900),
    "Bookcases": (80, 700),
    "Chairs": (60, 800),
    "Furnishings": (20, 500),
    "Tables": (200, 1500),
    "Appliances": (20, 800),
    "Art": (10, 300),
    "Binders": (5, 100),
    "Envelopes": (3, 50),
    "Fasteners": (2, 30),
    "Labels": (2, 40),
    "Paper": (10, 200),
    "Storage": (20, 400),
    "Supplies": (5, 100),
}


def _gen_category():
    return random.choices(list(CATEGORIES.keys()), weights=[0.3, 0.3, 0.4])[0]


def _gen_date(start, end):
    delta = (end - start).days
    return start + pd.Timedelta(days=random.randint(0, delta))


def _build_row(order_id, order_date):
    category = _gen_category()
    sub_category = random.choice(CATEGORIES[category])
    segment = random.choice(SEGMENTS)
    region = random.choice(REGIONS)

    low, high = SUB_PRICE[sub_category]
    # Base price influenced by category and some noise
    unit_price = random.uniform(low, high)
    quantity = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.25, 0.15, 0.12, 0.08])[0]

    discount = random.choices(
        [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        weights=[0.6, 0.1, 0.1, 0.08, 0.05, 0.04, 0.03],
    )[0]

    sales = round(unit_price * quantity, 2)
    # Profit tends to be positive but can be negative at high discounts
    profit_rate = random.uniform(0.05, 0.35) - discount * random.uniform(0.5, 1.2)
    profit = round(sales * profit_rate, 2)

    ship_date = order_date + pd.Timedelta(days=random.randint(0, 6))

    return {
        "Order ID": f"US-{order_id:05d}",
        "Order Date": order_date.strftime("%Y-%m-%d"),
        "Ship Date": ship_date.strftime("%Y-%m-%d"),
        "Segment": segment,
        "Region": region,
        "Category": category,
        "Sub-Category": sub_category,
        "Sales": sales,
        "Quantity": quantity,
        "Discount": discount,
        "Profit": profit,
    }


def main():
    orders = []
    for i in range(1, N_ORDERS + 1):
        # Group several line items into a single order
        n_items = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        order_date = _gen_date(START_DATE, END_DATE)
        for j in range(n_items):
            orders.append(_build_row(i, order_date))

    df = pd.DataFrame(orders)
    df = df.sort_values("Order Date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(df):,} rows -> {OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()
