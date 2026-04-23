"""
PJM Day-Ahead LMP — Data Collection
=====================================
Data sources:
  1. PJM DataMiner CSV files (manually downloaded, AEP zone)
     - da_hrl_lmps.csv       : 2022
     - da_hrl_lmps2023.csv   : 2023
     - da_hrl_lmps_2024_1q.csv : 2024 Q1
  2. EIA Grid Monitor — PJM hourly load (demand)
  3. Open-Meteo — historical weather (free, no key)

Train/Test split:
  Train : 2022–2023
  Test  : 2024 Q1 (Jan–Mar)

Output: raw_data.parquet
"""

import pandas as pd
import numpy as np
import requests
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WEATHER_LAT = 39.9612   # Columbus, OH — center of AEP zone
WEATHER_LON = -82.9988
EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# CSV file paths — place these in the same folder as this script
LMP_FILES = [
    "da_hrl_lmps.csv",
    "da_hrl_lmps2023.csv",
    "da_hrl_lmps_2024_1q.csv",
]


# ─────────────────────────────────────────────
# 1. LOAD AND MERGE PJM CSV FILES
# ─────────────────────────────────────────────
def load_lmp_files() -> pd.DataFrame:
    """
    Reads all PJM DataMiner CSV files, filters for AEP zone,
    and merges into a single clean DataFrame.
    """
    print("[LMP] Reading PJM CSV files...")
    all_dfs = []

    for fname in LMP_FILES:
        if not os.path.exists(fname):
            print(f"  ! File not found: {fname} — skipping")
            continue
        df = pd.read_csv(fname)
        # Filter AEP zone only
        df = df[df["pnode_name"] == "AEP"].copy()
        df = df[["datetime_beginning_ept", "total_lmp_da"]].copy()
        df.columns = ["datetime", "lmp_da"]
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["lmp_da"]   = pd.to_numeric(df["lmp_da"], errors="coerce")
        all_dfs.append(df)
        print(f"  ✓ {fname} — {len(df)} AEP rows")

    if not all_dfs:
        raise ValueError("No LMP files found. Place CSV files in the project folder.")

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna().sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    print(f"\n[LMP] Total: {len(df):,} hourly records")
    print(f"      Period: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    print(f"      LMP range: ${df['lmp_da'].min():.2f} – ${df['lmp_da'].max():.2f} /MWh\n")
    return df


# ─────────────────────────────────────────────
# 2. EIA LOAD DATA
# ─────────────────────────────────────────────
def fetch_eia_load(start: str, end: str) -> pd.DataFrame:
    """
    Pulls PJM hourly actual demand from EIA Grid Monitor.
    Paginated in chunks of 5000 rows.
    """
    print(f"[EIA Load] Fetching PJM hourly demand: {start} → {end}")
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "frequency"            : "hourly",
        "data[0]"              : "value",
        "facets[respondent][]" : "PJM",
        "facets[type][]"       : "D",
        "start"                : start,
        "end"                  : end,
        "sort[0][column]"      : "period",
        "sort[0][direction]"   : "asc",
        "length"               : 5000,
        "offset"               : 0,
        "api_key"              : EIA_API_KEY,
    }

    all_rows = []
    while True:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json().get("response", {}).get("data", [])
            if not rows:
                break
            all_rows.extend(rows)
            print(f"  ✓ {len(all_rows)} rows retrieved...")
            if len(rows) < 5000:
                break
            params["offset"] += 5000
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ EIA Load error: {e}")
            break

    if not all_rows:
        print("  [WARNING] No load data returned.\n")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["period"])
    df["load_mw"]  = pd.to_numeric(df["value"], errors="coerce")
    df = df[["datetime", "load_mw"]].dropna().sort_values("datetime").reset_index(drop=True)
    print(f"  → {len(df):,} hourly records\n")
    return df


# ─────────────────────────────────────────────
# 3. OPEN-METEO WEATHER
# ─────────────────────────────────────────────
def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """
    Pulls hourly temperature and wind speed from Open-Meteo archive.
    Location: Columbus, OH — geographic center of AEP control zone.
    """
    print(f"[Weather] Fetching Open-Meteo data: {start} → {end}")
    params = {
        "latitude"        : WEATHER_LAT,
        "longitude"       : WEATHER_LON,
        "start_date"      : start,
        "end_date"        : end,
        "hourly"          : "temperature_2m,windspeed_10m",
        "timezone"        : "America/New_York",
        "temperature_unit": "celsius",
        "windspeed_unit"  : "kmh",
    }
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame({
        "datetime"   : pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "wind_speed" : data["hourly"]["windspeed_10m"],
    })
    print(f"  → {len(df):,} hourly records\n")
    return df


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def build_raw_dataset() -> pd.DataFrame:
    """
    Merges LMP, load, and weather data into a single parquet file.
    """
    # 1. Real LMP from PJM CSV files
    lmp = load_lmp_files()

    # Determine date range from LMP data
    start_str = lmp["datetime"].min().strftime("%Y-%m-%d")
    end_str   = lmp["datetime"].max().strftime("%Y-%m-%d")

    # 2. Load from EIA
    load = fetch_eia_load(start_str + "T00", end_str + "T23")

    # 3. Weather from Open-Meteo
    weather = fetch_weather(start_str, end_str)

    # 4. Merge all on datetime
    df = lmp.merge(weather, on="datetime", how="left")
    if not load.empty:
        df = df.merge(load, on="datetime", how="left")
    else:
        df["load_mw"] = np.nan

    df = df.sort_values("datetime").reset_index(drop=True)
    df.to_parquet("raw_data.parquet", index=False)

    print("✅ raw_data.parquet saved")
    print(f"   Rows   : {len(df):,}")
    print(f"   Period : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nLMP statistics ($/MWh):")
    print(df["lmp_da"].describe().round(2))
    return df


if __name__ == "__main__":
    df = build_raw_dataset()
