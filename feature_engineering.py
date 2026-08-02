"""
Engineer 17 technical indicators from raw OHLCV data using the `ta` library.

Creates trend (SMA, EMA, MACD), momentum (RSI, Stochastic K), volatility
(Bollinger Bands, ATR), volume (OBV), and custom features (multi-horizon
returns, volume ratio, price range). Adds next-day close as target variable.

Outputs: data/ntpc_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import ta

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_raw.csv"

FEATURE_COLS = [
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "MACD", "RSI", "Stoch_K", "BB_upper", "BB_lower",
    "ATR", "OBV", "Returns_1d", "Returns_5d",
    "Returns_20d", "Vol_ratio", "Price_range",
]


def main():
    """Compute all 17 technical indicators and save feature CSV."""
    df = pd.read_csv(DATA_FILE, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)

    print(f"Loaded {len(df)} rows of raw data")

    # === TREND INDICATORS ===
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)
    df["MACD"] = ta.trend.macd_diff(df["Close"])

    # === MOMENTUM INDICATORS ===
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])

    # === VOLATILITY INDICATORS ===
    df["BB_upper"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_lower"] = ta.volatility.bollinger_lband(df["Close"])
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

    # === VOLUME INDICATORS ===
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # === CUSTOM FEATURES ===
    df["Returns_1d"] = df["Close"].pct_change(1)
    df["Returns_5d"] = df["Close"].pct_change(5)
    df["Returns_20d"] = df["Close"].pct_change(20)
    df["Vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Price_range"] = (df["High"] - df["Low"]) / df["Close"]

    # === TARGET VARIABLE ===
    df["Target"] = df["Close"].shift(-1)
    df["Direction"] = (df["Target"] > df["Close"]).astype(int)

    df = df.dropna()

    output_file = PROJECT_FOLDER / "data" / "ntpc_features.csv"
    df.to_csv(output_file)

    print(f"\nFeature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    print("\nFeature columns:")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {col}")

    # === TRAIN / VAL / TEST SPLIT ===
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


if __name__ == "__main__":
    main()
