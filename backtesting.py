"""
Backtest a simple long-only trading strategy using LSTM predictions.

Strategy: go long (buy) when the model predicts price will rise tomorrow,
stay in cash otherwise. Compares against a buy-and-hold baseline.

Metrics: Sharpe ratio, max drawdown, cumulative return, win rate.

Outputs: results/backtest_results.json, plots/backtest_cumulative.png,
         plots/backtest_drawdown.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

PROJECT_FOLDER = Path(__file__).resolve().parent
RESULTS_FOLDER = PROJECT_FOLDER / "results"
PLOTS_FOLDER = PROJECT_FOLDER / "plots"


def load_data():
    """Load saved predictions and actuals from LSTM training."""
    lstm_pred = np.load(RESULTS_FOLDER / "lstm_predictions.npy")
    y_true = np.load(RESULTS_FOLDER / "test_actuals.npy")
    close = np.load(RESULTS_FOLDER / "test_close.npy")
    return lstm_pred, y_true, close


def compute_returns(close):
    """Compute daily returns from closing prices."""
    return np.diff(close) / close[:-1]


def long_only_strategy(predictions, close):
    """
    Long-only strategy: buy when predicted price > current close.

    Returns daily strategy returns (same length as market returns).
    Position is determined at end of day t, return is earned on day t+1.
    """
    # predictions[i] = predicted close for day i+1 (next day)
    # close[i] = actual close on day i
    # Signal: go long if predicted next-day close > current close
    signals = (predictions[:-1] > close[:-1]).astype(float)

    # Market returns for the next day
    market_returns = compute_returns(close)

    # Strategy returns = signal * next-day market return
    strategy_returns = signals * market_returns

    return strategy_returns, market_returns, signals


def sharpe_ratio(returns, trading_days=252):
    """Annualised Sharpe ratio (assuming risk-free rate = 0)."""
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(trading_days) * returns.mean() / returns.std())


def max_drawdown(cumulative):
    """Maximum drawdown from peak."""
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def win_rate(returns):
    """Fraction of trading days with positive returns."""
    traded = returns[returns != 0]
    if len(traded) == 0:
        return 0.0
    return float(np.mean(traded > 0))


def run_backtest():
    """Run the full backtest and save results + plots."""
    lstm_pred, y_true, close = load_data()

    strat_ret, mkt_ret, signals = long_only_strategy(lstm_pred, close)

    # Cumulative returns (growth of ₹1)
    cum_strategy = np.cumprod(1 + strat_ret)
    cum_market = np.cumprod(1 + mkt_ret)

    # Metrics
    results = {
        "strategy": {
            "total_return": f"{(cum_strategy[-1] - 1) * 100:.2f}%",
            "sharpe_ratio": round(sharpe_ratio(strat_ret), 3),
            "max_drawdown": f"{max_drawdown(cum_strategy) * 100:.2f}%",
            "win_rate": f"{win_rate(strat_ret) * 100:.1f}%",
            "days_invested": int(signals.sum()),
            "total_trading_days": len(signals),
            "exposure": f"{signals.mean() * 100:.1f}%",
        },
        "buy_and_hold": {
            "total_return": f"{(cum_market[-1] - 1) * 100:.2f}%",
            "sharpe_ratio": round(sharpe_ratio(mkt_ret), 3),
            "max_drawdown": f"{max_drawdown(cum_market) * 100:.2f}%",
        },
    }

    # Save results
    with open(RESULTS_FOLDER / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS — LSTM Long-Only Strategy")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Strategy':>15} {'Buy & Hold':>15}")
    print("-" * 55)
    print(f"{'Total Return':<25} {results['strategy']['total_return']:>15} {results['buy_and_hold']['total_return']:>15}")
    print(f"{'Sharpe Ratio':<25} {results['strategy']['sharpe_ratio']:>15} {results['buy_and_hold']['sharpe_ratio']:>15}")
    print(f"{'Max Drawdown':<25} {results['strategy']['max_drawdown']:>15} {results['buy_and_hold']['max_drawdown']:>15}")
    print(f"{'Win Rate':<25} {results['strategy']['win_rate']:>15} {'—':>15}")
    print(f"{'Market Exposure':<25} {results['strategy']['exposure']:>15} {'100.0%':>15}")
    print(f"{'Days Invested':<25} {results['strategy']['days_invested']:>15} {results['strategy']['total_trading_days']:>15}")

    # ── Plot 1: Cumulative Returns ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum_strategy, label="LSTM Strategy", color="#C4A265", linewidth=1.5)
    ax.plot(cum_market, label="Buy & Hold", color="#7B9BC4", linewidth=1.5)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Cumulative Returns: LSTM Strategy vs Buy & Hold", fontsize=14)
    ax.set_xlabel("Trading Days (Test Period)")
    ax.set_ylabel("Growth of ₹1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "backtest_cumulative.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: backtest_cumulative.png")

    # ── Plot 2: Drawdown ──
    peak_strat = np.maximum.accumulate(cum_strategy)
    dd_strat = (cum_strategy - peak_strat) / peak_strat
    peak_mkt = np.maximum.accumulate(cum_market)
    dd_mkt = (cum_market - peak_mkt) / peak_mkt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(dd_strat)), dd_strat, 0, alpha=0.4, color="#C4A265", label="Strategy Drawdown")
    ax.fill_between(range(len(dd_mkt)), dd_mkt, 0, alpha=0.4, color="#7B9BC4", label="Buy & Hold Drawdown")
    ax.set_title("Drawdown Comparison", fontsize=14)
    ax.set_xlabel("Trading Days (Test Period)")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "backtest_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: backtest_drawdown.png")

    print(f"\nResults saved to {RESULTS_FOLDER / 'backtest_results.json'}")


if __name__ == "__main__":
    run_backtest()
