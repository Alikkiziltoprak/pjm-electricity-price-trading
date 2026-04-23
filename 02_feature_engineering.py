"""
Feature Engineering
====================
Input:  raw_data.parquet
Output: features.parquet

Feature groups:
  - Time features       : hour, day_of_week, month, quarter, etc.
  - Calendar flags      : is_weekend, is_holiday, is_peak
  - Temperature derived : heating/cooling degree hours, extreme flag
  - Load features       : load_mw with lags and rolling stats
  - LMP lag features    : 1h, 2h, 3h, 24h, 48h, 168h lags + rolling stats
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False
    print("[WARNING] 'holidays' package not found. Run: pip install holidays")


# ─────────────────────────────────────────────
# TIME FEATURES
# ─────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and cyclical time components from datetime index."""
    df["hour"]         = df["datetime"].dt.hour
    df["day_of_week"]  = df["datetime"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"]        = df["datetime"].dt.month
    df["year"]         = df["datetime"].dt.year
    df["day_of_year"]  = df["datetime"].dt.dayofyear
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["datetime"].dt.quarter
    return df


# ─────────────────────────────────────────────
# CALENDAR FLAGS
# ─────────────────────────────────────────────
def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary flags for weekends, US federal holidays, and peak hours.
    Peak hours: 07:00–22:00 on non-weekend, non-holiday days.
    These are strong price drivers in PJM markets.
    """
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if HAS_HOLIDAYS:
        us_holidays = holidays.US(years=[2022, 2023, 2024])
        df["is_holiday"] = df["datetime"].dt.date.apply(
            lambda d: 1 if d in us_holidays else 0
        )
    else:
        # Fallback: major US holidays only
        major_holidays = [
            "2022-01-01", "2022-07-04", "2022-11-24", "2022-12-25",
            "2023-01-01", "2023-07-04", "2023-11-23", "2023-12-25",
            "2024-01-01", "2024-07-04", "2024-11-28", "2024-12-25",
        ]
        holiday_dates = pd.to_datetime(major_holidays).date
        df["is_holiday"] = df["datetime"].dt.date.apply(
            lambda d: 1 if d in holiday_dates else 0
        )

    # On-peak definition: weekday business hours (common PJM convention)
    df["is_peak"] = (
        (df["hour"].between(7, 22)) &
        (df["is_weekend"] == 0) &
        (df["is_holiday"] == 0)
    ).astype(int)

    return df


# ─────────────────────────────────────────────
# TEMPERATURE-DERIVED FEATURES
# ─────────────────────────────────────────────
def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives heating/cooling degree hours and an extreme temperature flag.
    Base temperature: 18°C (64.4°F) — standard energy industry reference.
    HDH and CDH are strong proxies for heating/cooling electricity demand.
    """
    base = 18.0
    df["hdh"] = np.maximum(base - df["temperature"], 0)   # heating demand proxy
    df["cdh"] = np.maximum(df["temperature"] - base, 0)   # cooling demand proxy

    # Extreme temperatures tend to cause price spikes
    df["temp_extreme"] = (
        (df["temperature"] < -10) | (df["temperature"] > 35)
    ).astype(int)

    return df


# ─────────────────────────────────────────────
# LAG FEATURES — LMP
# ─────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lagged and rolling features from historical LMP values.
    These capture autocorrelation patterns critical for time series models.
    Lag 168h = same hour last week — strong seasonal signal in power markets.
    """
    df = df.sort_values("datetime").copy()

    # Day-ahead forecasting: at time t, we predict t+24.
    # Only lags >= 24h are available at prediction time — shorter lags are future leakage.
    for lag in [24, 48, 168]:
        df[f"lmp_lag_{lag}h"] = df["lmp_da"].shift(lag)

    # Rolling statistics based on data available 24h ago (no leakage)
    df["lmp_roll_mean_24h"] = df["lmp_da"].shift(24).rolling(24).mean()
    df["lmp_roll_std_24h"]  = df["lmp_da"].shift(24).rolling(24).std()
    df["lmp_roll_max_24h"]  = df["lmp_da"].shift(24).rolling(24).max()

    # Same-hour reference points (both available at prediction time)
    df["lmp_same_hour_yesterday"] = df["lmp_da"].shift(24)
    df["lmp_same_hour_last_week"] = df["lmp_da"].shift(168)

    return df


# ─────────────────────────────────────────────
# LAG FEATURES — LOAD
# ─────────────────────────────────────────────
def add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lagged load features. Load is a primary driver of electricity prices —
    high demand periods typically correspond to higher LMP values.
    """
    if "load_mw" not in df.columns or df["load_mw"].isnull().all():
        print("[WARNING] Load data unavailable — load features skipped.")
        return df

    df["load_lag_1h"]        = df["load_mw"].shift(1)
    df["load_lag_24h"]       = df["load_mw"].shift(24)
    df["load_roll_mean_24h"] = df["load_mw"].shift(1).rolling(24).mean()

    return df


# ─────────────────────────────────────────────
# MISSING VALUE HANDLING
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles two types of missing values:
    1. Load NaN: imputed with hour-of-day median (preserves intraday pattern)
    2. Lag warm-up rows: dropped (first 168 hours lack full lag history)
    Remaining NaNs: forward-fill then backward-fill as a safety net.
    """
    # Impute missing load with hour-of-day median
    if "load_mw" in df.columns and df["load_mw"].isnull().any():
        hourly_median = df.groupby("hour")["load_mw"].transform("median")
        df["load_mw"] = df["load_mw"].fillna(hourly_median)

    # Drop rows where longest lag (168h) is still NaN — no valid history yet
    df = df.dropna(subset=["lmp_lag_168h"]).copy()

    # Safety fill for any remaining NaNs
    df = df.ffill().bfill()

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def build_features() -> pd.DataFrame:
    print("[Features] Reading raw_data.parquet...")
    df = pd.read_parquet("raw_data.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])

    print("[Features] Adding time features...")
    df = add_time_features(df)

    print("[Features] Adding calendar flags...")
    df = add_calendar_flags(df)

    print("[Features] Adding temperature-derived features...")
    df = add_temperature_features(df)

    print("[Features] Adding LMP lag features...")
    df = add_lag_features(df)

    print("[Features] Adding load lag features...")
    df = add_load_features(df)

    print("[Features] Handling missing values...")
    df = handle_missing(df)

    df.to_parquet("features.parquet", index=False)

    feature_cols = [c for c in df.columns if c not in ["datetime", "lmp_da"]]
    print(f"\n✅ features.parquet saved")
    print(f"   Rows     : {len(df)}")
    print(f"   Features : {len(feature_cols)}")
    print(f"   List     : {feature_cols}")
    print(f"\nTarget (lmp_da) statistics:")
    print(df["lmp_da"].describe().round(2))

    return df


if __name__ == "__main__":
    df = build_features()
