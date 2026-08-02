"""
Download 5 years of daily OHLCV data for NTPC.NS from Yahoo Finance.

Saves raw data to data/ntpc_raw.csv for downstream EDA and modelling.
"""

from pathlib import Path

import yfinance as yf


# This finds the folder containing this Python file.
PROJECT_FOLDER = Path(__file__).resolve().parent

# This points to the data folder inside the project.
DATA_FOLDER = PROJECT_FOLDER / "data"

# Create the data folder if it does not already exist.
DATA_FOLDER.mkdir(exist_ok=True)


# NTPC's Yahoo Finance ticker.
TICKER = "NTPC.NS"


def main():
    """Download NTPC stock data and save to CSV."""
    print("Downloading NTPC stock data...")

    data = yf.download(
        TICKER,
        start="2019-08-01",
        end="2026-08-01",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            "No data was downloaded. Check your internet connection."
        )

    output_file = DATA_FOLDER / "ntpc_raw.csv"
    data.to_csv(output_file)

    print("Download completed successfully.")
    print("Number of rows downloaded:", len(data))
    print("File saved at:", output_file)
    print("\nFirst five rows:")
    print(data.head())


if __name__ == "__main__":
    main()