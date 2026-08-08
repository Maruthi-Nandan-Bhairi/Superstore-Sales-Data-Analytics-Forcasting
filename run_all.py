"""
Run the complete Superstore Sales pipeline end-to-end:
  1. Generate sample data
  2. Exploratory analysis & visualizations
  3. Time-series forecasting

Run:
    python run_all.py
"""

import subprocess
import sys

STEPS = [
    ("Generating sample data", ["python", "data/generate_sample_data.py"]),
    ("Running exploratory analysis", ["python", "superstore_analysis.py"]),
    ("Running sales/profit forecasting", ["python", "superstore_forecast.py"]),
]


def main():
    for title, cmd in STEPS:
        print(f"\n{'#'*60}")
        print(f"# {title}")
        print(f"{'#'*60}")
        code = subprocess.run(cmd, cwd=".").returncode
        if code != 0:
            print(f"\nERROR: Step failed with exit code {code}: {title}")
            sys.exit(code)

    print("\nAll steps completed successfully!")


if __name__ == "__main__":
    main()
