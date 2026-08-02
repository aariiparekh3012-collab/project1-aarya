# Energy Stock Price Forecasting Using LSTM

Forecasting next-day closing prices of **NTPC** (National Thermal Power Corporation) — India's largest power utility — using LSTM neural networks, benchmarked against Linear Regression and Random Forest baselines.

> **Key finding:** Linear Regression (RMSE 5.44) outperformed LSTM (RMSE 18.16) because the feature-target relationship is inherently linear — a strong lesson in matching model complexity to data structure.

---

## Results

| Model | RMSE (₹) | MAE (₹) | Directional Accuracy |
|-------|----------|---------|---------------------|
| **Linear Regression** | **5.44** | **4.06** | **51.74%** |
| LSTM (2-layer, 64 hidden) | 18.16 | 14.08 | 50.00% |
| Random Forest (200 trees) | 24.85 | 22.30 | 49.57% |

### Why Linear Regression Won

The top features — SMA_50 (31.8%), SMA_200 (30.6%), EMA_12 (14.7%) — are smoothed versions of the closing price, making the feature→target relationship nearly perfectly linear. Linear Regression fits this exactly. Random Forest rounds predictions at tree-split boundaries, and LSTM overcomplicates what is fundamentally a linear mapping.

---

## Pipeline

```
Raw OHLCV Data → EDA & Analysis → Feature Engineering → Train/Val/Test Split → Model Training → Evaluation
```

### 1. Data Collection
- **Source:** Yahoo Finance via `yfinance`
- **Stock:** NTPC.NS (proxy for NIFTY Energy Index)
- **Period:** 5 years of daily OHLCV data (~1,250 trading days)

### 2. Exploratory Data Analysis
- Closing price trend, daily returns, 30-day rolling volatility
- Returns distribution with KDE overlay
- Monthly returns heatmap
- ACF/PACF autocorrelation plots
- ADF stationarity test (prices: non-stationary, returns: stationary)

<p align="center">
  <img src="plots/ntpc_closing_price.png" width="45%" />
  <img src="plots/returns_distribution.png" width="45%" />
</p>
<p align="center">
  <img src="plots/monthly_heatmap.png" width="45%" />
  <img src="plots/acf_pacf.png" width="45%" />
</p>

### 3. Feature Engineering (17 Technical Indicators)

| Category | Features | Purpose |
|----------|----------|---------|
| Trend | SMA(20, 50, 200), EMA(12, 26), MACD | Market direction & momentum |
| Momentum | RSI(14), Stochastic K | Overbought/oversold detection |
| Volatility | Bollinger Bands (upper/lower), ATR | Price spread & range |
| Volume | OBV | Trend confirmation via volume |
| Custom | Returns(1d, 5d, 20d), Volume ratio, Price range | Multi-horizon returns |

### 4. Data Split

Chronological split (no shuffling — prevents data leakage):
- **Train:** 70% | **Validation:** 15% | **Test:** 15%
- StandardScaler fit on training set only

### 5. Models

**Linear Regression** — Simple baseline that exploits the linear feature-target relationship.

**Random Forest** (200 trees, max_depth=10) — Non-linear ensemble baseline.

**LSTM** — 2-layer LSTM with 64 hidden units, 30-day lookback window:
- Dropout (0.2 between LSTM layers, 0.1 before output)
- Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping (patience=10) with gradient clipping (max_norm=1.0)
- Target scaling via separate StandardScaler (inverse-transformed for evaluation)

<p align="center">
  <img src="plots/training_curves.png" width="45%" />
  <img src="plots/actual_vs_predicted.png" width="45%" />
</p>
<p align="center">
  <img src="plots/feature_importance.png" width="55%" />
</p>

---

## Project Structure

```
energy-stock-forecasting/
├── data/
│   ├── ntpc_raw.csv              # Raw OHLCV data
│   └── ntpc_features.csv         # Processed features (17 indicators + target)
├── plots/
│   ├── ntpc_closing_price.png    # Price trend
│   ├── ntpc_daily_returns.png    # Daily returns
│   ├── ntpc_volatility.png       # Rolling volatility
│   ├── returns_distribution.png  # Returns histogram + KDE
│   ├── monthly_heatmap.png       # Monthly returns heatmap
│   ├── acf_pacf.png              # ACF/PACF plots
│   ├── feature_importance.png    # RF feature importance
│   ├── training_curves.png       # LSTM train/val loss
│   └── actual_vs_predicted.png   # LSTM predictions vs actual
├── results/
│   ├── baseline_results.json     # LR + RF metrics
│   └── all_model_results.json    # All 3 models compared
├── models/
│   └── best_lstm.pth             # Saved LSTM weights
├── downloaddata.py               # Data collection script
├── exploredata.py                # Initial EDA (price chart)
├── returns_analysis.py           # Returns & volatility analysis
├── eda_remaining.py              # Distribution, heatmap, ACF, ADF test
├── feature_engineering.py        # 17 technical indicators + target
├── baseline_models.py            # Linear Regression + Random Forest
├── lstm_model.py                 # LSTM training & evaluation
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Setup

```bash
# Clone
git clone https://github.com/aariiparekh3012-collab/project1-aarya.git
cd project1-aarya

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python downloaddata.py          # Download NTPC data
python exploredata.py           # Closing price chart
python returns_analysis.py      # Returns & volatility
python eda_remaining.py         # Remaining EDA plots + ADF test
python feature_engineering.py   # Engineer 17 features
python baseline_models.py       # Train baselines
python lstm_model.py            # Train LSTM
```

---

## Tech Stack

- **Data:** yfinance, pandas, NumPy
- **ML:** scikit-learn (Linear Regression, Random Forest, StandardScaler)
- **Deep Learning:** PyTorch (LSTM, DataLoader, LR scheduling)
- **Technical Indicators:** ta library
- **Visualization:** matplotlib, seaborn, statsmodels
- **Statistics:** ADF test, ACF/PACF, rolling volatility

---

## Key Learnings

1. **Simpler models can win.** When features have a linear relationship with the target, Linear Regression will beat complex models that spend capacity learning the same linear function plus noise.

2. **Data leakage is subtle.** Shuffling time-series data or fitting the scaler on the full dataset gives artificially good results that collapse in production.

3. **Target scaling matters for LSTMs.** Without scaling the target to match feature scale, the LSTM's gradients become unstable and predictions diverge.

4. **Feature importance reveals data structure.** The Random Forest importance plot showed that SMA/EMA features dominate — explaining why a linear model is the natural fit.

---

## Future Work

- Predict **returns** instead of raw prices (better suited for LSTM's non-linear capacity)
- Add **external features** (crude oil prices, USD/INR, sector sentiment)
- **Multi-stock** forecasting (Reliance Energy, ONGC, Power Grid)
- **Backtesting** with Sharpe ratio, max drawdown, cumulative returns
- Compare against **GRU** and lightweight **Transformers**

---

**Author:** Aarya Parekh | B.Tech Energy Engineering | IIT Bombay | August 2026
