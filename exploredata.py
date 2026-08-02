"""
Initial exploratory data analysis — load NTPC raw data and plot
closing price trend over the full 5-year period.

Outputs: plots/ntpc_closing_price.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Find the main project folder.
PROJECT_FOLDER = Path(__file__).resolve().parent

# Create paths for the input CSV and output chart.
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_raw.csv"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"

# Make sure the plots folder exists.
PLOTS_FOLDER.mkdir(exist_ok=True)


def main():
    """Load raw data, print summary stats, and plot closing price."""
    data = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0, parse_dates=True)
    data.columns = data.columns.get_level_values(0)

    print("Data loaded successfully.")
    print("Number of rows:", len(data))
    print("\nColumn names:")
    print(data.columns.tolist())
    print("\nFirst five rows:")
    print(data.head())
    print("\nMissing values:")
    print(data.isna().sum())

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"])
    plt.title("NTPC Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price in INR")
    plt.grid(True)
    plt.tight_layout()

    output_chart = PLOTS_FOLDER / "ntpc_closing_price.png"
    plt.savefig(output_chart, dpi=150)
    plt.show()

    print("\nChart saved at:", output_chart)


if __name__ == "__main__":
    main()