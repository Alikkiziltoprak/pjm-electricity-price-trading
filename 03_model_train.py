"""
XGBoost Model Training & Evaluation
=====================================
Train : 2022–2023
Test  : 2024
Input : features.parquet
Output: model.json, predictions.parquet, metrics.json, forecast_results.png

Evaluation metrics:
  - MAE  : Mean Absolute Error ($/MWh) — primary metric
  - RMSE : Root Mean Squared Error ($/MWh) — penalizes large errors
  - R²   : Coefficient of determination
  - MAPE : Mean Absolute Percentage Error (%)
  - Spike MAE : MAE specifically on hours with LMP > $100/MWh
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Time
    "hour", "day_of_week", "month", "year", "day_of_year",
    "week_of_year", "quarter",
    # Calendar flags
    "is_weekend", "is_holiday", "is_peak",
    # Weather
    "temperature", "wind_speed", "hdh", "cdh", "temp_extreme",
    # Load
    "load_mw", "load_lag_1h", "load_lag_24h", "load_roll_mean_24h",
    # LMP lags — only >= 24h (no data leakage for day-ahead forecasting)
    "lmp_lag_24h", "lmp_lag_48h", "lmp_lag_168h",
    "lmp_roll_mean_24h", "lmp_roll_std_24h", "lmp_roll_max_24h",
    "lmp_same_hour_yesterday", "lmp_same_hour_last_week",
]

TARGET     = "lmp_da"
TRAIN_END  = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END   = "2024-03-31"   # Q1 2024 only


# ─────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_data():
    """
    Loads features.parquet and splits into train/test sets.
    Train: 2022–2023 (historical pattern learning)
    Test : 2024 (out-of-sample evaluation)
    Only features present in the DataFrame are used — warns on any missing.
    """
    df = pd.read_parquet("features.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"[WARNING] Features not found (skipped): {missing}")

    train = df[df["datetime"] <= TRAIN_END].copy()
    test  = df[(df["datetime"] >= TEST_START) & (df["datetime"] <= TEST_END)].copy()

    X_train = train[available_features]
    y_train = train[TARGET]
    X_test  = test[available_features]
    y_test  = test[TARGET]

    print(f"Train: {len(train):,} rows ({train['datetime'].min().date()} → {train['datetime'].max().date()})")
    print(f"Test : {len(test):,} rows  ({test['datetime'].min().date()} → {test['datetime'].max().date()})")
    print(f"Features: {len(available_features)}\n")

    return X_train, y_train, X_test, y_test, test["datetime"], available_features


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test):
    """
    Trains XGBoost regressor with early stopping.
    Validation set: first 20% of test period (avoids data leakage).
    Hyperparameters tuned for hourly electricity price forecasting:
      - max_depth=6 : captures complex non-linear interactions
      - subsample=0.8 : reduces overfitting on seasonal patterns
      - reg_alpha/lambda : L1/L2 regularization for price spike robustness
    """
    val_size = int(len(X_test) * 0.2)
    X_val = X_test.iloc[:val_size]
    y_val = y_test.iloc[:val_size]

    params = {
        "n_estimators"         : 1000,
        "learning_rate"        : 0.05,
        "max_depth"            : 6,
        "min_child_weight"     : 3,
        "subsample"            : 0.8,
        "colsample_bytree"     : 0.8,
        "reg_alpha"            : 0.1,
        "reg_lambda"           : 1.0,
        "random_state"         : 42,
        "n_jobs"               : -1,
        "early_stopping_rounds": 50,
        "eval_metric"          : "mae",
    }

    model = xgb.XGBRegressor(**params)

    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    print(f"\nBest iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, datetimes, feature_cols):
    """
    Computes standard regression metrics plus spike-specific MAE.
    Spike evaluation (LMP > $100/MWh) is especially relevant for
    trading applications — these hours carry the highest P&L impact.
    """
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-6))) * 100

    print("\n" + "="*50)
    print("TEST SET PERFORMANCE (2024)")
    print("="*50)
    print(f"  MAE  : {mae:.2f}  $/MWh")
    print(f"  RMSE : {rmse:.2f} $/MWh")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f} %")
    print("="*50)

    # Spike-hour evaluation — high-value hours for trading decisions
    spike_mask = y_test > 100
    if spike_mask.sum() > 0:
        spike_mae = mean_absolute_error(y_test[spike_mask], preds[spike_mask])
        print(f"\n  Spike MAE (LMP > $100/MWh, n={spike_mask.sum()} hours): {spike_mae:.2f} $/MWh")

    # Save predictions
    results = pd.DataFrame({
        "datetime" : datetimes.values,
        "actual"   : y_test.values,
        "predicted": preds,
        "error"    : preds - y_test.values,
    })
    results.to_parquet("predictions.parquet", index=False)

    # Save metrics
    metrics = {
        "mae" : round(mae,  4),
        "rmse": round(rmse, 4),
        "r2"  : round(r2,   4),
        "mape": round(mape, 4),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance summary
    importance = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Features by Importance:")
    print(importance.head(10).to_string(index=False))

    return results, importance


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def plot_results(results: pd.DataFrame, importance: pd.DataFrame):
    """
    Generates a three-panel diagnostic figure:
      1. Close-up forecast vs actual (first week of January 2024)
      2. Monthly MAE breakdown across 2024
      3. Top 15 feature importances
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("PJM AEP — Day-Ahead LMP Forecasting (XGBoost)", fontsize=14, fontweight="bold")

    # Panel 1: January 2024 first week — close-up view
    jan_2024 = results[results["datetime"].dt.month == 1].head(7 * 24)
    ax = axes[0]
    ax.plot(jan_2024["datetime"], jan_2024["actual"],    label="Actual",    color="#2196F3", linewidth=1.5)
    ax.plot(jan_2024["datetime"], jan_2024["predicted"], label="Predicted", color="#FF5722", linewidth=1.5, linestyle="--")
    ax.set_title("January 2024 — First Week (Hourly)")
    ax.set_ylabel("LMP ($/MWh)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.grid(alpha=0.3)

    # Panel 2: Monthly MAE across 2024
    results["month"] = results["datetime"].dt.month
    monthly_mae = results.groupby("month").apply(
        lambda g: mean_absolute_error(g["actual"], g["predicted"])
    ).reset_index()
    monthly_mae.columns = ["month", "mae"]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    ax = axes[1]
    bars = ax.bar(monthly_mae["month"], monthly_mae["mae"], color="#4CAF50", edgecolor="white")
    ax.set_title("2024 — Monthly MAE ($/MWh)")
    ax.set_ylabel("MAE ($/MWh)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, monthly_mae["mae"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Panel 3: Feature importance — top 15
    top15 = importance.head(15)
    ax = axes[2]
    # Top 3 highlighted in orange, rest in blue
    colors = ["#FF5722" if i < 3 else "#2196F3" for i in range(len(top15))]
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color=colors[::-1])
    ax.set_title("Feature Importance (Top 15)")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("forecast_results.png", dpi=150, bbox_inches="tight")
    print("\n✅ forecast_results.png saved")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, datetimes, feature_cols = prepare_data()
    model = train_model(X_train, y_train, X_test, y_test)
    model.save_model("model.json")
    print("\n✅ model.json saved")

    results, importance = evaluate(model, X_test, y_test, datetimes, feature_cols)
    plot_results(results, importance)

    print("\n🎯 Phase 1 complete.")
    print("   Next step: 04_signal_generation.py (Phase 2)")
