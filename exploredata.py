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


# Load the CSV file.
data = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0, parse_dates=True)

# yfinance may save column names in two levels.
# Keep only the first level: Close, High, Low, Open, Volume.
data.columns = data.columns.get_level_values(0)

print("Data loaded successfully.")
print("Number of rows:", len(data))

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst five rows:")
print(data.head())

print("\nMissing values:")
print(data.isna().sum())


# Create a closing-price chart.
plt.figure(figsize=(12, 6))

plt.plot(data.index, data["Close"])

plt.title("NTPC Closing Price")
plt.xlabel("Date")
plt.ylabel("Price in INR")
plt.grid(True)

# Make the layout fit properly.
plt.tight_layout()

# Save the chart inside the plots folder.
output_chart = PLOTS_FOLDER / "ntpc_closing_price.png"
plt.savefig(output_chart, dpi=150)

# Display the chart.
plt.show()

print("\nChart saved at:", output_chart)