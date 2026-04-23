"""
Signal Generation — Phase 2 (Updated)
=======================================
Converts day-ahead LMP forecasts into actionable trading signals.

Improvements over v1:
  1. Transaction cost included ($7/MWh — realistic PJM estimate)
  2. Top 20% signal strength filter — reduces overtrading
  3. Max drawdown calculation — key risk metric
  4. Sharpe-like ratio for strategy quality assessment

Signal logic:
  - LONG     : predicted > $45/MWh AND in top 20% of prediction strength
  - SHORT    : predicted < $22/MWh AND in top 20% of prediction strength
  - NO TRADE : signal too weak or filtered out

Output: signals.parquet, signal_report.png, signal_summary.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LONG_THRESHOLD   = 45.0   # predicted > $45/MWh → LONG candidate
SHORT_THRESHOLD  = 22.0   # predicted < $22/MWh → SHORT candidate
SPIKE_THRESHOLD  = 100.0  # $/MWh — extreme price flag
TRANSACTION_COST = 2.0    # $/MWh — realistic PJM day-ahead bid/offer spread
TOP_PCT_FILTER   = 0.20   # only trade top 20% strongest signals
POSITION_MW      = 100    # 100 MW position size


# ─────────────────────────────────────────────
# 1. LOAD PREDICTIONS
# ─────────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    """
    Loads model predictions from Phase 1.
    Computes signal strength as absolute deviation from median prediction.
    """
    print("[Signals] Loading predictions.parquet...")
    df = pd.read_parquet("predictions.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Rolling mean — used for reference in plots
    df["rolling_mean_24h"] = df["actual"].rolling(24, min_periods=1).mean()

    # Signal strength = how far predicted is from median (absolute)
    median_pred = df["predicted"].median()
    df["signal_strength"] = (df["predicted"] - median_pred).abs()

    print(f"  → {len(df):,} hourly predictions loaded")
    print(f"  Period : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    print(f"  Predicted range: ${df['predicted'].min():.2f} – ${df['predicted'].max():.2f}/MWh\n")
    return df


# ─────────────────────────────────────────────
# 2. GENERATE SIGNALS
# ─────────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage signal generation:
      Stage 1: Price level filter (LONG > $45, SHORT < $22)
      Stage 2: Top 20% signal strength filter — keeps only
               the strongest conviction trades, reducing overtrading.

    This approach deliberately sacrifices some recall for better precision
    and lower transaction cost drag.
    """
    # Stage 1: price level conditions
    long_price  = df["predicted"] > LONG_THRESHOLD
    short_price = df["predicted"] < SHORT_THRESHOLD

    # Stage 2: top 20% signal strength filter
    strength_cutoff = df["signal_strength"].quantile(1 - TOP_PCT_FILTER)
    strong_signal   = df["signal_strength"] >= strength_cutoff

    # Apply both filters
    df["signal"] = "NO TRADE"
    df.loc[long_price  & strong_signal, "signal"] = "LONG"
    df.loc[short_price & strong_signal, "signal"] = "SHORT"

    # Spike flag
    df["spike_flag"] = (df["actual"] > SPIKE_THRESHOLD).astype(int)

    print("[Signals] Signal distribution (after top 20% filter):")
    counts = df["signal"].value_counts()
    total  = len(df)
    for sig, cnt in counts.items():
        print(f"  {sig:<10}: {cnt:>4} hours ({cnt/total*100:.1f}%)")
    print()
    return df


# ─────────────────────────────────────────────
# 3. P&L SIMULATION WITH TRANSACTION COSTS
# ─────────────────────────────────────────────
def simulate_pnl(df: pd.DataFrame) -> tuple:
    """
    Simulates P&L with realistic transaction costs.

    Gross P&L:
      LONG  = (actual - predicted) * MW
      SHORT = (predicted - actual) * MW

    Net P&L = Gross P&L - (TRANSACTION_COST * MW) per trade

    Transaction cost of $7/MWh reflects realistic PJM bid-offer
    spreads and ancillary costs for a 100 MW position.
    """
    df["pnl_gross"] = 0.0
    df["pnl_net"]   = 0.0

    long_mask  = df["signal"] == "LONG"
    short_mask = df["signal"] == "SHORT"
    trade_mask = df["signal"] != "NO TRADE"

    # Gross P&L
    df.loc[long_mask,  "pnl_gross"] = (df.loc[long_mask,  "actual"] - df.loc[long_mask,  "predicted"]) * POSITION_MW
    df.loc[short_mask, "pnl_gross"] = (df.loc[short_mask, "predicted"] - df.loc[short_mask, "actual"])  * POSITION_MW

    # Net P&L (after transaction costs)
    df.loc[trade_mask, "pnl_net"] = df.loc[trade_mask, "pnl_gross"] - (TRANSACTION_COST * POSITION_MW)

    # Cumulative P&L
    df["cumulative_pnl_gross"] = df["pnl_gross"].cumsum()
    df["cumulative_pnl_net"]   = df["pnl_net"].cumsum()

    n_trades    = trade_mask.sum()
    gross_pnl   = df["pnl_gross"].sum()
    net_pnl     = df["pnl_net"].sum()
    win_rate    = (df.loc[trade_mask, "pnl_net"] > 0).mean() * 100
    avg_pnl     = net_pnl / n_trades if n_trades > 0 else 0
    tc_drag     = gross_pnl - net_pnl

    print("[P&L] Trading simulation results:")
    print(f"  Total trades      : {n_trades}")
    print(f"  Win rate (net)    : {win_rate:.1f}%")
    print(f"  Gross P&L         : ${gross_pnl:>10,.0f}")
    print(f"  Transaction costs : ${tc_drag:>10,.0f}  (${TRANSACTION_COST}/MWh × {POSITION_MW}MW × {n_trades} trades)")
    print(f"  Net P&L           : ${net_pnl:>10,.0f}")
    print(f"  Avg net P&L/trade : ${avg_pnl:>10,.0f}")
    print()

    summary = {
        "n_trades"        : int(n_trades),
        "win_rate_pct"    : round(win_rate, 2),
        "gross_pnl"       : round(gross_pnl, 2),
        "transaction_costs": round(tc_drag, 2),
        "net_pnl"         : round(net_pnl, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
    }
    return df, summary


# ─────────────────────────────────────────────
# 4. RISK METRICS
# ─────────────────────────────────────────────
def compute_risk_metrics(df: pd.DataFrame) -> dict:
    """
    Computes key risk metrics for the strategy:

    Max Drawdown: largest peak-to-trough decline in cumulative P&L.
      Critical for position sizing and risk limit setting.

    Calmar Ratio: net P&L / max drawdown.
      Higher = better risk-adjusted return.

    These metrics directly connect to Ali's risk management background —
    same concepts used in financial risk, different asset class.
    """
    cum_pnl = df["cumulative_pnl_net"]

    # Max drawdown
    rolling_max  = cum_pnl.cummax()
    drawdown     = cum_pnl - rolling_max
    max_drawdown = drawdown.min()

    # Calmar ratio (simplified — net P&L / |max drawdown|)
    net_pnl = cum_pnl.iloc[-1]
    calmar  = net_pnl / abs(max_drawdown) if max_drawdown != 0 else 0

    # Daily P&L for Sharpe-like ratio
    df["date"]     = df["datetime"].dt.date
    daily_pnl      = df.groupby("date")["pnl_net"].sum()
    sharpe_like    = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    print("[Risk Metrics]")
    print(f"  Max Drawdown    : ${max_drawdown:,.0f}")
    print(f"  Calmar Ratio    : {calmar:.2f}")
    print(f"  Sharpe-like     : {sharpe_like:.2f}  (annualized, daily P&L)")
    print()

    return {
        "max_drawdown"  : round(max_drawdown, 2),
        "calmar_ratio"  : round(calmar, 4),
        "sharpe_like"   : round(sharpe_like, 4),
    }


# ─────────────────────────────────────────────
# 5. SPIKE DETECTION EVALUATION
# ─────────────────────────────────────────────
def evaluate_spike_detection(df: pd.DataFrame) -> dict:
    """
    Evaluates LONG signal quality specifically for spike hours (>$100/MWh).
    High recall is prioritized — missing a spike is more costly than a false alarm.
    """
    long_mask  = df["signal"] == "LONG"
    spike_mask = df["spike_flag"] == 1

    tp = (long_mask & spike_mask).sum()
    fp = (long_mask & ~spike_mask).sum()
    fn = (~long_mask & spike_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("[Spike Detection] LONG signal vs actual spikes (>$100/MWh):")
    print(f"  Actual spikes   : {spike_mask.sum()}")
    print(f"  LONG signals    : {long_mask.sum()}")
    print(f"  True positives  : {tp}")
    print(f"  Precision       : {precision:.2%}")
    print(f"  Recall          : {recall:.2%}")
    print(f"  F1 Score        : {f1:.2%}")
    print()

    return {
        "n_spikes" : int(spike_mask.sum()),
        "precision": round(precision, 4),
        "recall"   : round(recall, 4),
        "f1"       : round(f1, 4),
    }


# ─────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────
def plot_signals(df: pd.DataFrame):
    """
    Four-panel signal report:
      1. LMP forecast with LONG/SHORT signal markers
      2. Cumulative P&L — gross vs net (transaction cost impact visible)
      3. Drawdown curve — key risk visual
      4. Signal heatmap by hour of day
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    fig.suptitle("PJM AEP — Trading Signal Generation (Phase 2)\nWith Transaction Costs & Risk Metrics",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Price + signals ──
    ax = axes[0]
    ax.plot(df["datetime"], df["actual"],         color="#2196F3", linewidth=1,   label="Actual LMP",    alpha=0.8)
    ax.plot(df["datetime"], df["predicted"],      color="#FF9800", linewidth=1,   label="Predicted LMP", alpha=0.8, linestyle="--")
    ax.plot(df["datetime"], df["rolling_mean_24h"], color="#9C27B0", linewidth=1.2, label="24h Mean",    linestyle=":")

    long_df  = df[df["signal"] == "LONG"]
    short_df = df[df["signal"] == "SHORT"]
    ax.scatter(long_df["datetime"],  long_df["actual"],  color="#4CAF50", s=25, zorder=5, label="LONG",  alpha=0.8)
    ax.scatter(short_df["datetime"], short_df["actual"], color="#F44336", s=25, zorder=5, label="SHORT", alpha=0.8)

    ax.set_title("LMP Forecast with Trading Signals — Q1 2024")
    ax.set_ylabel("LMP ($/MWh)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # ── Panel 2: Cumulative P&L gross vs net ──
    ax = axes[1]
    ax.plot(df["datetime"], df["cumulative_pnl_gross"], color="#4CAF50", linewidth=1.5,
            label=f"Gross P&L: ${df['cumulative_pnl_gross'].iloc[-1]:,.0f}", linestyle="--")
    ax.plot(df["datetime"], df["cumulative_pnl_net"],   color="#2196F3", linewidth=2,
            label=f"Net P&L (after costs): ${df['cumulative_pnl_net'].iloc[-1]:,.0f}")
    ax.fill_between(df["datetime"], df["cumulative_pnl_net"], alpha=0.1, color="#2196F3")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Cumulative P&L — Gross vs Net (100 MW, ${TRANSACTION_COST}/MWh transaction cost)")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # ── Panel 3: Drawdown curve ──
    ax = axes[2]
    cum_pnl     = df["cumulative_pnl_net"]
    rolling_max = cum_pnl.cummax()
    drawdown    = cum_pnl - rolling_max
    ax.fill_between(df["datetime"], drawdown, 0, color="#F44336", alpha=0.5)
    ax.plot(df["datetime"], drawdown, color="#B71C1C", linewidth=1)
    ax.set_title(f"Drawdown Curve (Max Drawdown: ${drawdown.min():,.0f})")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # ── Panel 4: Signal heatmap by hour ──
    ax = axes[3]
    df["hour"]       = df["datetime"].dt.hour
    signal_map       = {"LONG": 1, "NO TRADE": 0, "SHORT": -1}
    df["signal_num"] = df["signal"].map(signal_map)
    hourly           = df.groupby("hour")["signal_num"].mean()
    bar_colors       = ["#4CAF50" if v > 0.02 else "#F44336" if v < -0.02 else "#BDBDBD" for v in hourly]
    ax.bar(hourly.index, hourly.values, color=bar_colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Average Signal Bias by Hour of Day")
    ax.set_xlabel("Hour of Day (EPT)")
    ax.set_ylabel("Signal Score (1=LONG, -1=SHORT)")
    ax.set_xticks(range(0, 24))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("signal_report.png", dpi=150, bbox_inches="tight")
    print("✅ signal_report.png saved")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_predictions()
    df = generate_signals(df)
    df, pnl_summary = simulate_pnl(df)
    risk_metrics    = compute_risk_metrics(df)
    spike_metrics   = evaluate_spike_detection(df)

    df.to_parquet("signals.parquet", index=False)

    summary = {
        "config": {
            "long_threshold"  : LONG_THRESHOLD,
            "short_threshold" : SHORT_THRESHOLD,
            "spike_threshold" : SPIKE_THRESHOLD,
            "transaction_cost": TRANSACTION_COST,
            "position_mw"     : POSITION_MW,
            "top_pct_filter"  : TOP_PCT_FILTER,
        },
        "signal_counts" : df["signal"].value_counts().to_dict(),
        "pnl"           : pnl_summary,
        "risk"          : risk_metrics,
        "spike_detection": spike_metrics,
    }
    with open("signal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ signals.parquet saved")
    print("✅ signal_summary.json saved")
    print("\n🎯 Phase 2 complete.")
    print("   Next step: 05_risk_interpretability.py (Phase 3)")
