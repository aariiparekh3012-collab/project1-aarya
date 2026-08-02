from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_raw.csv"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"
PLOTS_FOLDER.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")

# Load data (handle yfinance two-level columns)
data = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0, parse_dates=True)
data.columns = data.columns.get_level_values(0)

# Daily returns
data["Returns"] = data["Close"].pct_change()

print(f"Loaded {len(data)} rows")


# === PLOT 1: Returns Distribution Histogram ===
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data["Returns"].dropna(), bins=100, kde=True, ax=ax, color="#7B9BC4")
ax.axvline(
    data["Returns"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {data['Returns'].mean():.4f}",
)
ax.set_title("Daily Returns Distribution", fontsize=14)
ax.set_xlabel("Daily Return")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "returns_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved: returns_distribution.png")


# === PLOT 2: Monthly Returns Heatmap ===
monthly = data["Returns"].resample("M").sum().to_frame()
monthly["Year"] = monthly.index.year
monthly["Month"] = monthly.index.month
pivot = monthly.pivot_table(values="Returns", index="Year", columns="Month")

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    pivot,
    cmap="RdYlGn",
    center=0,
    annot=True,
    fmt=".2%",
    ax=ax,
    xticklabels=[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
)
ax.set_title("Monthly Returns Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "monthly_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved: monthly_heatmap.png")


# === PLOT 3: ACF / PACF ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(data["Returns"].dropna(), lags=40, ax=ax1)
ax1.set_title("Autocorrelation (ACF)")
plot_pacf(data["Returns"].dropna(), lags=40, ax=ax2)
ax2.set_title("Partial Autocorrelation (PACF)")
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved: acf_pacf.png")


# === ADF Stationarity Test ===
result_price = adfuller(data["Close"].dropna())
result_returns = adfuller(data["Returns"].dropna())

print("\n--- STATIONARITY TESTS ---")
print(
    f"  Prices   ADF p-value: {result_price[1]:.6f}  "
    f"{'(Non-stationary)' if result_price[1] > 0.05 else '(Stationary)'}"
)
print(
    f"  Returns  ADF p-value: {result_returns[1]:.6f}  "
    f"{'(Stationary)' if result_returns[1] < 0.05 else '(Non-stationary)'}"
)


# === Key Stats for README ===
print("\n--- KEY STATS FOR README ---")
print(f"  Total trading days:    {len(data)}")
print(f"  Mean daily return:     {data['Returns'].mean():.4%}")
print(f"  Annualized return:     {data['Returns'].mean() * 252:.2%}")
print(f"  Annualized volatility: {data['Returns'].std() * np.sqrt(252):.2%}")
print(f"  Skewness:              {data['Returns'].skew():.3f}")
print(f"  Kurtosis:              {data['Returns'].kurtosis():.3f}")

cumulative = (1 + data["Returns"]).cumprod()
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
print(f"  Max drawdown:          {drawdown.min():.2%}")

print("\nDone. Check plots/ for 3 new charts.")
