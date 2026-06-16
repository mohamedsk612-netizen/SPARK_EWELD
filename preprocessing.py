"""
spark/preprocessing.py
----------------------
Data cleaning and extreme weather classification.
Replicates the exact pipeline from Liu et al. (2023) Section Methods.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from spark.config import (CITY_THRESHOLDS, CONDITION_EW_MAP, EW_COLS,
                           ZSCORE_THRESHOLD, IQR_LOOKBACK_PREV,
                           IQR_LOOKBACK_LONG)
from spark.utils import zscore_outlier_mask, iqr_outlier_mask

logger = logging.getLogger(__name__)


# ── Electricity preprocessing ─────────────────────────────────────────────────

def preprocess_electricity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper pipeline (Section Methods, p.7):
    1. Remove duplicate timestamps (keep first).
    2. Forward-fill missing values.
    3. Z-score outlier detection (Z=3) → replace with mean of prev 2 obs.
    4. IQR outlier detection → replace with mean of prev 96 obs.
    """
    df  = df.copy()
    df  = df[~df.index.duplicated(keep="first")]
    val = df["Value"].ffill()
    arr = val.values.copy()

    # Z-score pass
    z_mask = zscore_outlier_mask(val, z_thresh=ZSCORE_THRESHOLD)
    for i in np.where(z_mask)[0]:
        if i >= IQR_LOOKBACK_PREV:
            arr[i] = np.mean(arr[i - IQR_LOOKBACK_PREV: i])
        elif i >= 1:
            arr[i] = arr[i - 1]

    # IQR pass
    iqr_mask = iqr_outlier_mask(pd.Series(arr, index=val.index))
    for i in np.where(iqr_mask)[0]:
        start = max(0, i - IQR_LOOKBACK_LONG)
        if start < i:
            arr[i] = np.mean(arr[start:i])

    df["Value"] = arr
    return df


def preprocess_all_electricity(user_dfs: dict) -> dict:
    """Apply preprocessing to all 386 users."""
    logger.info("Preprocessing electricity data ...")
    processed = {uid: preprocess_electricity(df) for uid, df in user_dfs.items()}
    logger.info("Done. %d users processed.", len(processed))
    return processed


# ── Weather preprocessing ─────────────────────────────────────────────────────

def standardize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw weather headers to short standard names."""
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "temp" in cl and "dew" not in cl and c != "temp":
            col_map[c] = "temp"
        elif "dew" in cl and c != "dew":
            col_map[c] = "dew"
        elif "humidity" in cl and c != "humidity":
            col_map[c] = "humidity"
        elif ("wind speed" in cl or "windspeed" in cl) and c != "wind_speed":
            col_map[c] = "wind_speed"
        elif ("wind gust" in cl or "windgust" in cl) and c != "wind_gust":
            col_map[c] = "wind_gust"
        elif "pressure" in cl and c != "pressure":
            col_map[c] = "pressure"
        elif "condition" in cl and c != "condition":
            col_map[c] = "condition"
        elif "wind" in cl and "speed" not in cl and "gust" not in cl \
                and c != "wind_dir":
            col_map[c] = "wind_dir"
    return df.rename(columns=col_map)


def preprocess_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper pipeline (Section Methods, p.7-8):
    1. Standardise column names.
    2. Remove duplicates, forward-fill.
    3. IQR + Z-score outlier replacement.
    4. Linear interpolation: 30-min → 15-min.
    """
    df      = standardize_weather_columns(df.copy())
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df = df[~df.index.duplicated(keep="first")]
    df[num_cols] = df[num_cols].ffill()

    for col in num_cols:
        series   = df[col].copy()
        iqr_mask = iqr_outlier_mask(series)
        arr      = series.values.copy()
        sigma    = series.std() if series.std() != 0 else 1
        for i in np.where(iqr_mask)[0]:
            if i >= 2:
                prev_mean = np.mean([arr[i - 1], arr[i - 2]])
                if abs(arr[i] - prev_mean) / sigma > 2:
                    arr[i] = prev_mean
            elif i >= 1:
                arr[i] = arr[i - 1]
        df[col] = arr

    # Upsample 30-min → 15-min
    if len(df) > 1:
        new_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min")
        df      = df.reindex(df.index.union(new_idx))
        df[num_cols] = df[num_cols].interpolate(method="time")
        df      = df.reindex(new_idx)

    return df


def preprocess_all_weather(weather_dfs: dict) -> dict:
    """Apply preprocessing to all city weather DataFrames."""
    logger.info("Preprocessing weather data ...")
    processed = {city: preprocess_weather(df) for city, df in weather_dfs.items()}
    logger.info("Done. Cities processed: %s", list(processed.keys()))
    return processed


# ── Extreme weather classification ────────────────────────────────────────────

def classify_extreme_weather(weather_df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Classify each 15-min record into EW1–EW20 using Table 3 criteria.
    Returns DataFrame with boolean EW columns + 'EW_types' + 'is_extreme'.
    """
    thresholds = CITY_THRESHOLDS.get(city, CITY_THRESHOLDS["CT1"])
    df         = weather_df.copy()

    def safe(col):
        return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

    temp       = safe("temp")
    humidity   = safe("humidity")
    wind_speed = safe("wind_speed")
    wind_gust  = safe("wind_gust")
    condition  = safe("condition").astype(str).str.strip()

    df["EW1"]  = temp < thresholds["low_temp"]
    df["EW2"]  = temp > thresholds["high_temp"]
    df["EW3"]  = humidity > thresholds["high_hum"]
    df["EW4"]  = (temp > 95) & (humidity > 60)
    df["EW5"]  = (wind_gust > 58)  & (wind_gust <= 74)
    df["EW6"]  = (wind_gust > 74)  & (wind_gust <= 91)
    df["EW7"]  =  wind_gust > 91
    df["EW8"]  = (wind_speed > 39) & (wind_speed <= 54)
    df["EW9"]  = (wind_speed > 54) & (wind_speed <= 73)
    df["EW10"] = (wind_speed > 73) & (wind_speed <= 93)
    df["EW11"] = (wind_speed > 93) & (wind_speed <= 114)
    df["EW12"] =  wind_speed > 114

    for cond_str, ew_code in CONDITION_EW_MAP.items():
        df[ew_code] = condition == cond_str

    ew_cols = [f"EW{i}" for i in range(1, 21)]

    def active_ews(row):
        return ", ".join(ew for ew in ew_cols if row.get(ew, False))

    df["EW_types"]  = df[ew_cols].apply(active_ews, axis=1)
    df["is_extreme"] = df["EW_types"].str.len() > 0
    return df


def classify_all_cities(weather_dfs: dict) -> dict:
    """Classify EW events for all cities."""
    logger.info("Classifying extreme weather events ...")
    ew_dfs = {}
    for city, df in weather_dfs.items():
        ew_dfs[city] = classify_extreme_weather(df, city)
        n = ew_dfs[city]["is_extreme"].sum()
        logger.info("  %s: %d extreme-weather 15-min records found.", city, n)
    return ew_dfs
