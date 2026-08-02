"""
Train a GRU model for next-day NTPC closing price prediction.

Same architecture, hyperparameters, and evaluation pipeline as lstm_model.py
but replaces the LSTM cell with GRU. This enables a direct apples-to-apples
comparison between the two recurrent architectures.

GRU has fewer parameters (no separate cell state) and often trains faster
while achieving comparable accuracy on financial time-series tasks.

Outputs: models/best_gru.pth, results/gru_results.json,
         plots/gru_training_curves.png, plots/gru_actual_vs_predicted.png
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


# ============================================
# SEQUENCE DATASET (shared with LSTM)
# ============================================
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


# ============================================
# GRU MODEL
# ============================================
class GRUForecaster(nn.Module):
    """GRU-based price forecaster — mirrors LSTMForecaster architecture."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
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
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden).squeeze()


# ============================================
# HYPERPARAMETERS (identical to LSTM)
# ============================================
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
PATIENCE = 10

FEATURE_COLS = [
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "MACD", "RSI", "Stoch_K", "BB_upper", "BB_lower",
    "ATR", "OBV", "Returns_1d", "Returns_5d",
    "Returns_20d", "Vol_ratio", "Price_range",
]


def main():
    """Train and evaluate the GRU model."""
    MODELS_FOLDER.mkdir(exist_ok=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    PLOTS_FOLDER.mkdir(exist_ok=True)

    # ── Load & split ──
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

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

    # ── Scale target (same fix as LSTM) ──
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(train[["Target"]]).flatten()
    y_val = target_scaler.transform(val[["Target"]]).flatten()
    y_test_scaled = target_scaler.transform(test[["Target"]]).flatten()

    # ── Dataloaders ──
    train_ds = StockDataset(X_train_scaled, y_train, SEQ_LEN)
    val_ds = StockDataset(X_val_scaled, y_val, SEQ_LEN)
    test_ds = StockDataset(X_test_scaled, y_test_scaled, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Training ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GRUForecaster(input_dim=len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nGRU parameters: {total_params:,}")
    print(f"Training GRU...")
    print(f"  Architecture: 2-layer GRU, 64 hidden, 30-day lookback")
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
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODELS_FOLDER / "best_gru.pth")
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
    ax.set_title("GRU Training & Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "gru_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Test evaluation ──
    model.load_state_dict(torch.load(MODELS_FOLDER / "best_gru.pth", weights_only=True))
    model.eval()

    all_preds = []
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(preds)

    gru_pred_scaled = np.array(all_preds).reshape(-1, 1)
    gru_pred = target_scaler.inverse_transform(gru_pred_scaled).flatten()
    y_true = test["Target"].values[SEQ_LEN:]
    close_test = test["Close"].values[SEQ_LEN:]

    rmse = np.sqrt(mean_squared_error(y_true, gru_pred))
    mae = mean_absolute_error(y_true, gru_pred)
    actual_dir = np.sign(y_true - close_test)
    pred_dir = np.sign(gru_pred - close_test)
    dir_acc = np.mean(actual_dir == pred_dir)

    print(f"\n--- GRU TEST RESULTS ---")
    print(f"  RMSE:             {rmse:.2f}")
    print(f"  MAE:              {mae:.2f}")
    print(f"  Directional Acc:  {dir_acc:.2%}")
    print(f"  Parameters:       {total_params:,}")

    # Save results
    gru_result = {
        "model": "GRU",
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "dir_acc": round(float(dir_acc), 4),
        "parameters": total_params,
    }
    with open(RESULTS_FOLDER / "gru_results.json", "w") as f:
        json.dump(gru_result, f, indent=2)

    # ── Actual vs Predicted plot ──
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_true, label="Actual", color="#C4A265", linewidth=1)
    ax.plot(gru_pred, label="GRU Predicted", color="#2E8B57", linewidth=1, alpha=0.8)
    ax.set_title("GRU: Actual vs Predicted Prices (Test Set)", fontsize=14)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "gru_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save predictions
    np.save(RESULTS_FOLDER / "gru_predictions.npy", gru_pred)

    print(f"\nModel saved: {MODELS_FOLDER / 'best_gru.pth'}")
    print(f"Plots saved: gru_training_curves.png, gru_actual_vs_predicted.png")


if __name__ == "__main__":
    main()
