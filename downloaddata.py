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

print("Downloading NTPC stock data...")


data = yf.download(
    TICKER,
    start="2019-08-01",
    end="2026-08-01",
    auto_adjust=False,
    progress=False
)


# Check whether Yahoo Finance returned any rows.
if data.empty:
    raise RuntimeError(
        "No data was downloaded. Check your internet connection."
    )


# Decide where the CSV file will be saved.
output_file = DATA_FOLDER / "ntpc_raw.csv"

# Save the downloaded table as a CSV file.
data.to_csv(output_file)


print("Download completed successfully.")
print("Number of rows downloaded:", len(data))
print("File saved at:", output_file)

print("\nFirst five rows:")
print(data.head())