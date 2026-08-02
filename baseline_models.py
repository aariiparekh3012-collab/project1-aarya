"""
Train and evaluate baseline models: Linear Regression and Random Forest.

Chronological train/val/test split (70/15/15), StandardScaler on train only.
Evaluates RMSE, MAE, and directional accuracy on the test set.

Outputs: results/baseline_results.json, plots/feature_importance.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_features.csv"
RESULTS_FOLDER = PROJECT_FOLDER / "results"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"

RESULTS_FOLDER.mkdir(exist_ok=True)
PLOTS_FOLDER.mkdir(exist_ok=True)

FEATURE_COLS = [
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "MACD", "RSI", "Stoch_K", "BB_upper", "BB_lower",
    "ATR", "OBV", "Returns_1d", "Returns_5d",
    "Returns_20d", "Vol_ratio", "Price_range",
]


def evaluate(y_true, y_pred, name, close_prices):
    """Compute RMSE, MAE, and directional accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    actual_dir = np.sign(y_true - close_prices)
    pred_dir = np.sign(y_pred - close_prices)
    dir_acc = np.mean(actual_dir == pred_dir)
    print(f"  {name:20s} | RMSE: {rmse:10.2f} | MAE: {mae:10.2f} | Dir.Acc: {dir_acc:.2%}")
    return {"model": name, "rmse": float(rmse), "mae": float(mae), "dir_acc": float(dir_acc)}


def main():
    """Train and evaluate Linear Regression and Random Forest baselines."""
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_val = scaler.transform(val[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])

    y_train = train["Target"].values
    y_test = test["Target"].values
    close_test = test["Close"].values

    results = []

    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results.append(evaluate(y_test, lr_pred, "Linear Regression", close_test))

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append(evaluate(y_test, rf_pred, "Random Forest", close_test))

    with open(RESULTS_FOLDER / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance plot
    importance = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    importance.plot(kind="barh", ax=ax, color="#C4A265")
    ax.set_title("Random Forest — Feature Importance", fontsize=14)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to {RESULTS_FOLDER / 'baseline_results.json'}")
    print(f"Feature importance plot saved to {PLOTS_FOLDER / 'feature_importance.png'}")

    print(f"\nTop 5 most important features:")
    for feat, imp in importance.tail(5).items():
        print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
