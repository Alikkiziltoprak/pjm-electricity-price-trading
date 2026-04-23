"""
Risk & Interpretability — Phase 3
====================================
Two components:

1. SHAP (SHapley Additive exPlanations)
   - Why did the model make each prediction?
   - Which features drive high/low price forecasts?
   - Global feature importance vs local explanation

2. Value at Risk (VaR)
   - Historical simulation VaR on trading P&L
   - 95% and 99% confidence levels
   - Expected Shortfall (CVaR) — average loss beyond VaR

Output: shap_summary.png, var_report.png, risk_report.json
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARNING] shap not installed. Run: pip install shap")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "temperature", "wind_speed", "load_mw",
    "hour", "day_of_week", "month", "year", "day_of_year",
    "week_of_year", "quarter",
    "is_weekend", "is_holiday", "is_peak",
    "hdh", "cdh", "temp_extreme",
    "lmp_lag_24h", "lmp_lag_48h", "lmp_lag_168h",
    "lmp_roll_mean_24h", "lmp_roll_std_24h", "lmp_roll_max_24h",
    "lmp_same_hour_yesterday", "lmp_same_hour_last_week",
    "load_lag_1h", "load_lag_24h", "load_roll_mean_24h",
]

VAR_CONFIDENCE_LEVELS = [0.95, 0.99]


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    """Loads model, features, predictions, and signals."""
    print("[Phase 3] Loading data...")

    model = xgb.XGBRegressor()
    model.load_model("model.json")

    features = pd.read_parquet("features.parquet")
    features["datetime"] = pd.to_datetime(features["datetime"])

    predictions = pd.read_parquet("predictions.parquet")
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])

    signals = pd.read_parquet("signals.parquet")
    signals["datetime"] = pd.to_datetime(signals["datetime"])

    # Test set only (2024 Q1)
    test_features = features[features["datetime"] >= "2024-01-01"].copy()
    available_cols = [c for c in FEATURE_COLS if c in test_features.columns]
    X_test = test_features[available_cols]

    print(f"  Model loaded: {model.best_iteration} trees")
    print(f"  Test features: {len(X_test):,} rows × {len(available_cols)} cols")
    print(f"  Predictions  : {len(predictions):,} rows")
    print(f"  Signals      : {len(signals):,} rows\n")

    return model, X_test, predictions, signals, available_cols


# ─────────────────────────────────────────────
# 2. SHAP ANALYSIS
# ─────────────────────────────────────────────
def run_shap_analysis(model, X_test, feature_cols):
    """
    Computes SHAP values for the test set.

    SHAP answers: "Why did the model predict $X for this hour?"
    Each feature gets a SHAP value = its contribution to that prediction.
    Positive SHAP = pushed price up, Negative = pushed price down.

    This is the gold standard for ML model interpretability and directly
    addresses the "black box" concern in regulated financial environments.
    """
    if not HAS_SHAP:
        print("[SHAP] Skipped — shap package not installed.")
        return None, None

    print("[SHAP] Computing SHAP values (this may take 1-2 minutes)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    # Mean absolute SHAP per feature
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

    print("\nTop 10 features by mean |SHAP|:")
    for feat, val in mean_abs_shap.head(10).items():
        print(f"  {feat:<30}: {val:.4f}")
    print()

    return shap_values, shap_df


# ─────────────────────────────────────────────
# 3. VALUE AT RISK
# ─────────────────────────────────────────────
def compute_var(signals: pd.DataFrame) -> dict:
    """
    Computes Historical Simulation VaR on net P&L.

    Historical VaR: sort actual P&L outcomes, take the loss
    at the chosen percentile. No distributional assumptions.

    VaR(95%) = "We expect to lose no more than $X on 95% of trading days"
    CVaR(95%) = "When we do exceed VaR, the average loss is $Y"

    CVaR (Expected Shortfall) is more informative than VaR alone —
    it captures tail risk, which is especially important in electricity
    markets where price spikes create extreme outcomes.
    """
    print("[VaR] Computing Value at Risk...")

    trade_pnl = signals[signals["signal"] != "NO TRADE"]["pnl_net"].values

    if len(trade_pnl) == 0:
        print("  No trades found.")
        return {}

    results = {}
    for conf in VAR_CONFIDENCE_LEVELS:
        var   = np.percentile(trade_pnl, (1 - conf) * 100)
        cvar  = trade_pnl[trade_pnl <= var].mean() if (trade_pnl <= var).sum() > 0 else var

        results[f"var_{int(conf*100)}"]  = round(var, 2)
        results[f"cvar_{int(conf*100)}"] = round(cvar, 2)

        print(f"  VaR  ({int(conf*100)}%): ${var:>10,.2f}  per trade")
        print(f"  CVaR ({int(conf*100)}%): ${cvar:>10,.2f}  per trade (expected shortfall)")

    # Daily VaR
    signals["date"]   = signals["datetime"].dt.date
    daily_pnl         = signals.groupby("date")["pnl_net"].sum()
    daily_var_95      = np.percentile(daily_pnl, 5)
    daily_cvar_95     = daily_pnl[daily_pnl <= daily_var_95].mean()

    results["daily_var_95"]  = round(daily_var_95, 2)
    results["daily_cvar_95"] = round(daily_cvar_95, 2)

    print(f"\n  Daily VaR  (95%): ${daily_var_95:>10,.2f}")
    print(f"  Daily CVaR (95%): ${daily_cvar_95:>10,.2f}")
    print()

    return results, trade_pnl, daily_pnl


# ─────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────
def plot_shap(shap_values, X_test, feature_cols):
    """SHAP summary and bar plots."""
    if shap_values is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("SHAP Feature Importance — PJM AEP LMP Forecasting", fontsize=13, fontweight="bold")

    # Mean absolute SHAP — bar chart
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=True)
    top15    = feat_imp.tail(15)

    colors = ["#FF5722" if i >= len(top15) - 3 else "#2196F3" for i in range(len(top15))]
    axes[0].barh(top15.index, top15.values, color=colors)
    axes[0].set_title("Mean |SHAP| Value (Global Importance)")
    axes[0].set_xlabel("Mean |SHAP Value| ($/MWh impact)")
    axes[0].grid(axis="x", alpha=0.3)

    # SHAP beeswarm-style scatter (manual — no shap.plots dependency)
    ax = axes[1]
    top10_feats = feat_imp.tail(10).index.tolist()
    top10_idx   = [feature_cols.index(f) for f in top10_feats]

    for i, (feat_idx, feat_name) in enumerate(zip(top10_idx[::-1], top10_feats[::-1])):
        shap_vals  = shap_values[:, feat_idx]
        feat_vals  = X_test.iloc[:, feat_idx].values

        # Normalize feature values for color
        feat_norm  = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
        jitter     = np.random.uniform(-0.2, 0.2, len(shap_vals))

        sc = ax.scatter(shap_vals, np.full_like(shap_vals, i) + jitter,
                        c=feat_norm, cmap="RdYlBu_r", alpha=0.3, s=8)

    ax.set_yticks(range(len(top10_feats)))
    ax.set_yticklabels(top10_feats[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("SHAP Value Distribution (Top 10 Features)\nColor: low=blue, high=red")
    ax.set_xlabel("SHAP Value ($/MWh)")
    ax.grid(axis="x", alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Feature value (normalized)")

    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    print("✅ shap_summary.png saved")
    plt.close()


def plot_var(trade_pnl, daily_pnl, var_results):
    """VaR visualization — P&L distribution and daily P&L."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Value at Risk Analysis — PJM AEP Trading Strategy", fontsize=13, fontweight="bold")

    # Panel 1: Trade-level P&L distribution with VaR lines
    ax = axes[0]
    ax.hist(trade_pnl, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", label="Trade P&L")
    ax.axvline(var_results["var_95"],  color="#FF9800", linewidth=2, linestyle="--",
               label=f"VaR 95%: ${var_results['var_95']:,.0f}")
    ax.axvline(var_results["var_99"],  color="#F44336", linewidth=2, linestyle="--",
               label=f"VaR 99%: ${var_results['var_99']:,.0f}")
    ax.axvline(var_results["cvar_95"], color="#FF9800", linewidth=2, linestyle=":",
               label=f"CVaR 95%: ${var_results['cvar_95']:,.0f}")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Trade-Level P&L Distribution with VaR")
    ax.set_xlabel("Net P&L per Trade ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Daily P&L with VaR reference
    ax = axes[1]
    daily_dates = list(daily_pnl.index)
    daily_vals  = daily_pnl.values
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in daily_vals]
    ax.bar(range(len(daily_vals)), daily_vals, color=colors, alpha=0.8)
    ax.axhline(var_results["daily_var_95"], color="#FF9800", linewidth=2, linestyle="--",
               label=f"Daily VaR 95%: ${var_results['daily_var_95']:,.0f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily P&L with VaR Reference Line")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Daily P&L ($)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("var_report.png", dpi=150, bbox_inches="tight")
    print("✅ var_report.png saved")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, X_test, predictions, signals, feature_cols = load_data()

    # SHAP
    shap_values, shap_df = run_shap_analysis(model, X_test, feature_cols)

    # VaR
    var_results, trade_pnl, daily_pnl = compute_var(signals)

    # Visualizations
    plot_shap(shap_values, X_test, feature_cols)
    plot_var(trade_pnl, daily_pnl, var_results)

    # Save risk report
    risk_report = {
        "var_analysis": var_results,
        "shap_top_features": (
            pd.DataFrame(np.abs(shap_values).mean(axis=0),
                         index=feature_cols, columns=["mean_abs_shap"])
            .sort_values("mean_abs_shap", ascending=False)
            .head(10)
            .round(4)
            .to_dict()["mean_abs_shap"]
        ) if shap_values is not None else {}
    }
    with open("risk_report.json", "w") as f:
        json.dump(risk_report, f, indent=2)

    print("\n✅ risk_report.json saved")
    print("\n🎯 Phase 3 complete — project pipeline finished!")
    print("\nAll outputs:")
    print("  Phase 1: model.json, predictions.parquet, forecast_results.png")
    print("  Phase 2: signals.parquet, signal_report.png, signal_summary.json")
    print("  Phase 3: shap_summary.png, var_report.png, risk_report.json")
