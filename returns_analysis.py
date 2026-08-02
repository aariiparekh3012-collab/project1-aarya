from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_raw.csv"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"

PLOTS_FOLDER.mkdir(exist_ok=True)


# Load the yfinance CSV
data = pd.read_csv(
    DATA_FILE,
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

# Simplify two-level column names
data.columns = data.columns.get_level_values(0)


# Daily return = percentage change in closing price
data["Daily_Return"] = data["Close"].pct_change()


# 30-day rolling annualised volatility
data["Volatility_30D"] = (
    data["Daily_Return"].rolling(30).std() * np.sqrt(252)
)


print("First rows with returns:")
print(data[["Close", "Daily_Return"]].head(10))

print("\nAverage daily return:")
print(data["Daily_Return"].mean())

print("\nAnnualised volatility:")
print(data["Daily_Return"].std() * np.sqrt(252))

plt.figure(figsize=(12, 6))

plt.plot(data.index, data["Daily_Return"])

plt.title("NTPC Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True)
plt.tight_layout()

returns_chart = PLOTS_FOLDER / "ntpc_daily_returns.png"
plt.savefig(returns_chart, dpi=150)
plt.show()

print("Returns chart saved at:", returns_chart)

plt.figure(figsize=(12, 6))

plt.plot(data.index, data["Volatility_30D"])

plt.title("NTPC 30-Day Rolling Annualised Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.tight_layout()

volatility_chart = PLOTS_FOLDER / "ntpc_volatility.png"
plt.savefig(volatility_chart, dpi=150)
plt.show()

print("Volatility chart saved at:", volatility_chart)