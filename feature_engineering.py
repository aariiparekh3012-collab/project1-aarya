from pathlib import Path

import numpy as np
import pandas as pd
import ta

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_raw.csv"

# Load raw data (handle yfinance two-level columns)
df = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0, parse_dates=True)
df.columns = df.columns.get_level_values(0)

print(f"Loaded {len(df)} rows of raw data")


# === TREND INDICATORS ===
# Simple Moving Averages — smooth out noise to reveal direction
df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)

# Exponential Moving Averages — same idea but recent days weigh more
df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)

# MACD — difference between fast and slow EMA, shows momentum shifts
df["MACD"] = ta.trend.macd_diff(df["Close"])


# === MOMENTUM INDICATORS ===
# RSI — measures overbought (>70) vs oversold (<30) conditions
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

# Stochastic K — where the close sits relative to the high-low range
df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])


# === VOLATILITY INDICATORS ===
# Bollinger Bands — upper and lower bands around the moving average
df["BB_upper"] = ta.volatility.bollinger_hband(df["Close"])
df["BB_lower"] = ta.volatility.bollinger_lband(df["Close"])

# ATR — average true range, measures daily price movement magnitude
df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])


# === VOLUME INDICATORS ===
# OBV — on-balance volume, confirms trends with volume
df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])


# === CUSTOM FEATURES ===
# Returns over different time horizons
df["Returns_1d"] = df["Close"].pct_change(1)
df["Returns_5d"] = df["Close"].pct_change(5)
df["Returns_20d"] = df["Close"].pct_change(20)

# Volume ratio — today's volume vs 20-day average
df["Vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

# Price range — intraday spread relative to close
df["Price_range"] = (df["High"] - df["Low"]) / df["Close"]


# === TARGET VARIABLE ===
# What we're trying to predict: tomorrow's closing price
df["Target"] = df["Close"].shift(-1)

# Binary direction: 1 = price goes up tomorrow, 0 = down
df["Direction"] = (df["Target"] > df["Close"]).astype(int)


# Drop rows with NaN (first ~200 rows lose SMA_200, last row loses Target)
df = df.dropna()

# Save processed data
output_file = PROJECT_FOLDER / "data" / "ntpc_features.csv"
df.to_csv(output_file)


# === DEFINE FEATURE COLUMNS ===
feature_cols = [
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "MACD", "RSI", "Stoch_K", "BB_upper", "BB_lower",
    "ATR", "OBV", "Returns_1d", "Returns_5d",
    "Returns_20d", "Vol_ratio", "Price_range",
]

print(f"\nFeature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Features: {len(feature_cols)}")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

print("\nFeature columns:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")


# === TRAIN / VAL / TEST SPLIT ===
# Chronological split — no shuffling (would leak future data)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"\n--- DATA SPLIT ---")
print(f"  Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
print(f"  Val:   {val.index[0].date()} to {val.index[-1].date()} ({len(val)} days)")
print(f"  Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")

print(f"\nDone. Saved to {output_file}")
