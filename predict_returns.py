"""
Train an LSTM to predict next-day *returns* instead of raw prices.

Returns are stationary (ADF p < 0.05), so the LSTM should learn temporal
patterns more effectively than when predicting non-stationary prices.
This script validates that hypothesis by comparing returns-based LSTM
performance against the price-based LSTM from lstm_model.py.

Strategy: predict next-day return → convert back to price for RMSE comparison.

Outputs: models/best_lstm_returns.pth, results/returns_lstm_results.json,
         plots/returns_lstm_training.png, plots/returns_lstm_pred.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:
    raise ImportError(
        "PyTorch is required. Install it with: pip install torch"
    ) from exc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_features.csv"
MODELS_FOLDER = PROJECT_FOLDER / "models"
RESULTS_FOLDER = PROJECT_FOLDER / "results"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"

FEATURE_COLS = [
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "MACD", "RSI", "Stoch_K", "BB_upper", "BB_lower",
    "ATR", "OBV", "Returns_1d", "Returns_5d",
    "Returns_20d", "Vol_ratio", "Price_range",
]

SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
PATIENCE = 10


class StockDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return x, y


class LSTMForecaster(nn.Module):
    """Same architecture as lstm_model.py — predicts a single scalar."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def main():
    """Train returns-based LSTM and compare with price-based LSTM."""
    MODELS_FOLDER.mkdir(exist_ok=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    PLOTS_FOLDER.mkdir(exist_ok=True)

    # ── Load data ──
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

    # Target: next-day return (instead of next-day price)
    df["Target_Return"] = (df["Target"] - df["Close"]) / df["Close"]
    df = df.dropna(subset=["Target_Return"])

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    # ── Scale features ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[FEATURE_COLS])
    X_val_scaled = scaler.transform(val[FEATURE_COLS])
    X_test_scaled = scaler.transform(test[FEATURE_COLS])

    # ── Scale target returns ──
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(train[["Target_Return"]]).flatten()
    y_val = target_scaler.transform(val[["Target_Return"]]).flatten()
    y_test_scaled = target_scaler.transform(test[["Target_Return"]]).flatten()

    # ── Dataloaders ──
    train_ds = StockDataset(X_train_scaled, y_train, SEQ_LEN)
    val_ds = StockDataset(X_val_scaled, y_val, SEQ_LEN)
    test_ds = StockDataset(X_test_scaled, y_test_scaled, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Training ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMForecaster(input_dim=len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    print(f"\nTraining LSTM (returns-based)...")
    print(f"  Target: next-day return (not price)")
    print(f"  Training samples: {len(train_ds)}, Validation: {len(val_ds)}")
    print("-" * 60)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(pred, y_batch).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODELS_FOLDER / "best_lstm_returns.pth")
            patience_counter = 0
            marker = " * saved"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 5 == 0 or marker:
            print(f"  Epoch {epoch:3d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Training curves ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", color="#C4A265")
    ax.plot(val_losses, label="Validation Loss", color="#7B9BC4")
    ax.set_title("LSTM (Returns) Training & Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "returns_lstm_training.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Test evaluation ──
    model.load_state_dict(
        torch.load(MODELS_FOLDER / "best_lstm_returns.pth", weights_only=True)
    )
    model.eval()

    all_preds = []
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(preds)

    # Inverse-transform predicted returns
    pred_returns_scaled = np.array(all_preds).reshape(-1, 1)
    pred_returns = target_scaler.inverse_transform(pred_returns_scaled).flatten()

    # Convert predicted returns back to predicted prices for RMSE comparison
    close_test = test["Close"].values[SEQ_LEN:]
    pred_prices = close_test * (1 + pred_returns)
    actual_prices = test["Target"].values[SEQ_LEN:]
    actual_returns = test["Target_Return"].values[SEQ_LEN:]

    # Price-level metrics (comparable with lstm_model.py)
    rmse_price = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mae_price = mean_absolute_error(actual_prices, pred_prices)

    # Return-level metrics
    rmse_return = np.sqrt(mean_squared_error(actual_returns, pred_returns))

    # Directional accuracy
    actual_dir = np.sign(actual_returns)
    pred_dir = np.sign(pred_returns)
    dir_acc = np.mean(actual_dir == pred_dir)

    print(f"\n--- LSTM (RETURNS) TEST RESULTS ---")
    print(f"  Price RMSE:       {rmse_price:.2f}  (comparable to price-based LSTM)")
    print(f"  Price MAE:        {mae_price:.2f}")
    print(f"  Return RMSE:      {rmse_return:.6f}")
    print(f"  Directional Acc:  {dir_acc:.2%}")

    # Save results
    result = {
        "model": "LSTM (Returns)",
        "rmse_price": round(float(rmse_price), 2),
        "mae_price": round(float(mae_price), 2),
        "rmse_return": round(float(rmse_return), 6),
        "dir_acc": round(float(dir_acc), 4),
    }
    with open(RESULTS_FOLDER / "returns_lstm_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── Actual vs Predicted plot ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Top: predicted vs actual prices
    axes[0].plot(actual_prices, label="Actual Price", color="#C4A265", linewidth=1)
    axes[0].plot(pred_prices, label="Predicted (from returns)", color="#9B59B6", linewidth=1, alpha=0.8)
    axes[0].set_title("LSTM (Returns-Based): Predicted vs Actual Prices", fontsize=14)
    axes[0].set_ylabel("Price (INR)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: predicted vs actual returns
    axes[1].plot(actual_returns, label="Actual Return", color="#C4A265", linewidth=0.8, alpha=0.7)
    axes[1].plot(pred_returns, label="Predicted Return", color="#9B59B6", linewidth=0.8, alpha=0.7)
    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_title("Predicted vs Actual Daily Returns", fontsize=12)
    axes[1].set_xlabel("Trading Days")
    axes[1].set_ylabel("Return")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "returns_lstm_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved: returns_lstm_training.png, returns_lstm_pred.png")
    print(f"Results saved: {RESULTS_FOLDER / 'returns_lstm_results.json'}")


if __name__ == "__main__":
    main()
