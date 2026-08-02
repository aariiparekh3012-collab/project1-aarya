from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch  # type: ignore[reportMissingImports]
    import torch.nn as nn  # type: ignore[reportMissingImports]
    from torch.utils.data import Dataset, DataLoader  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required to run this LSTM forecasting script. "
        "Install it with: pip install torch"
    ) from exc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "data" / "ntpc_features.csv"
MODELS_FOLDER = PROJECT_FOLDER / "models"
RESULTS_FOLDER = PROJECT_FOLDER / "results"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"

MODELS_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
PLOTS_FOLDER.mkdir(exist_ok=True)


# ============================================
# LOAD & PREPARE DATA
# ============================================
df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

feature_cols = [
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "MACD", "RSI", "Stoch_K", "BB_upper", "BB_lower",
    "ATR", "OBV", "Returns_1d", "Returns_5d",
    "Returns_20d", "Vol_ratio", "Price_range",
]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[feature_cols])
X_val_scaled = scaler.transform(val[feature_cols])
X_test_scaled = scaler.transform(test[feature_cols])

y_train = train["Target"].values
y_val = val["Target"].values
y_test = test["Target"].values


# ============================================
# SEQUENCE DATASET
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
# LSTM MODEL
# ============================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze()


# ============================================
# HYPERPARAMETERS
# ============================================
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
PATIENCE = 10


# ============================================
# CREATE DATALOADERS
# ============================================
train_ds = StockDataset(X_train_scaled, y_train, SEQ_LEN)
val_ds = StockDataset(X_val_scaled, y_val, SEQ_LEN)
test_ds = StockDataset(X_test_scaled, y_test, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# ============================================
# TRAINING
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMForecaster(input_dim=len(feature_cols), hidden_dim=64, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float("inf")
patience_counter = 0
train_losses = []
val_losses = []

print(f"\nTraining LSTM...")
print(f"  Architecture: 2-layer LSTM, 64 hidden, 30-day lookback")
print(f"  Training samples: {len(train_ds)}, Validation: {len(val_ds)}")
print("-" * 60)

for epoch in range(EPOCHS):
    # Train
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

    # Validate
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

    # Early stopping
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), MODELS_FOLDER / "best_lstm.pth")
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


# ============================================
# TRAINING CURVES PLOT
# ============================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label="Train Loss", color="#C4A265")
ax.plot(val_losses, label="Validation Loss", color="#7B9BC4")
ax.set_title("LSTM Training & Validation Loss", fontsize=14)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================
# TEST SET EVALUATION
# ============================================
model.load_state_dict(torch.load(MODELS_FOLDER / "best_lstm.pth", weights_only=True))
model.eval()

all_preds = []
all_actuals = []
test_loader = DataLoader(test_ds, batch_size=len(test_ds))

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        all_preds.extend(preds)
        all_actuals.extend(y_batch.numpy())

lstm_pred = np.array(all_preds)
y_true = np.array(all_actuals)
close_for_lstm = test["Close"].values[SEQ_LEN:]

# Metrics
rmse = np.sqrt(mean_squared_error(y_true, lstm_pred))
mae = mean_absolute_error(y_true, lstm_pred)
actual_dir = np.sign(y_true - close_for_lstm)
pred_dir = np.sign(lstm_pred - close_for_lstm)
dir_acc = np.mean(actual_dir == pred_dir)

print(f"\n--- LSTM TEST RESULTS ---")
print(f"  RMSE:             {rmse:.2f}")
print(f"  MAE:              {mae:.2f}")
print(f"  Directional Acc:  {dir_acc:.2%}")

# Load baseline results for comparison
with open(RESULTS_FOLDER / "baseline_results.json", "r") as f:
    baseline_results = json.load(f)

lstm_result = {"model": "LSTM", "rmse": float(rmse), "mae": float(mae), "dir_acc": float(dir_acc)}
all_results = baseline_results + [lstm_result]

with open(RESULTS_FOLDER / "all_model_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n--- MODEL COMPARISON ---")
print(f"  {'Model':20s} | {'RMSE':>10s} | {'MAE':>10s} | {'Dir.Acc':>10s}")
print(f"  {'-' * 58}")
for r in all_results:
    print(f"  {r['model']:20s} | {r['rmse']:10.2f} | {r['mae']:10.2f} | {r['dir_acc']:>9.2%}")


# === ACTUAL vs PREDICTED PLOT ===
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(y_true, label="Actual", color="#C4A265", linewidth=1)
ax.plot(lstm_pred, label="LSTM Predicted", color="#7B9BC4", linewidth=1, alpha=0.8)
ax.set_title("LSTM: Actual vs Predicted Prices (Test Set)", fontsize=14)
ax.set_xlabel("Trading Days")
ax.set_ylabel("Price (INR)")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_FOLDER / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# Save predictions for backtesting
np.save(RESULTS_FOLDER / "lstm_predictions.npy", lstm_pred)
np.save(RESULTS_FOLDER / "test_actuals.npy", y_true)
np.save(RESULTS_FOLDER / "test_close.npy", close_for_lstm)

print(f"\nModel saved: {MODELS_FOLDER / 'best_lstm.pth'}")
print(f"Plots saved: training_curves.png, actual_vs_predicted.png")
print(f"Predictions saved for backtesting.")
