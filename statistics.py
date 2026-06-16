"""
spark/statistics.py
-------------------
Replicates Tables 5, 6, 7 and EW event counts from Liu et al. (2023).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)


def compute_table5(user_dfs: dict) -> pd.DataFrame:
    """Table 5: Per-user summary statistics."""
    records = []
    for uid, df in user_dfs.items():
        val = df["Value"]
        records.append({
            "User":       uid,
            "Start Time": df.index.min(),
            "End Time":   df.index.max(),
            "Records":    len(df),
            "Max Value":  val.max(),
            "Min Value":  val.min(),
            "Mean Value": val.mean(),
        })
    return pd.DataFrame(records).set_index("User").sort_index()


def compute_table6(user_dfs: dict, user_section_map: dict) -> pd.DataFrame:
    """Table 6: Statistics per industrial section."""
    all_data: dict = {}
    for uid, df in user_dfs.items():
        sec = user_section_map.get(uid, "Unknown")
        all_data.setdefault(sec, []).append(df["Value"].values)

    records = []
    for section, arrays in sorted(all_data.items()):
        vals = np.concatenate(arrays)
        vals = vals[~np.isnan(vals)]
        records.append({
            "Section":  section,
            "Mean":     vals.mean(),
            "Std":      vals.std(),
            "Skew":     stats.skew(vals),
            "Kurtosis": stats.kurtosis(vals),
            "p0":       np.percentile(vals, 0),
            "p2.5":     np.percentile(vals, 2.5),
            "p50":      np.percentile(vals, 50),
            "p97.5":    np.percentile(vals, 97.5),
            "p100":     np.percentile(vals, 100),
        })
    return pd.DataFrame(records).set_index("Section")


def compute_table7(weather_dfs: dict) -> pd.DataFrame:
    """Table 7: Weather data statistics per city."""
    rename = {
        "temp":       "Temperature",
        "dew":        "Dew Point",
        "humidity":   "Humidity",
        "wind_speed": "Wind Speed",
        "wind_gust":  "Wind Gust",
        "pressure":   "Pressure",
    }
    records = []
    for city, df in weather_dfs.items():
        for col, label in rename.items():
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            records.append({
                "City":       city,
                "Meteorology": label,
                "Mean":        vals.mean(),
                "Std":         vals.std(),
                "Skew":        stats.skew(vals),
                "Kurtosis":    stats.kurtosis(vals),
                "Min":         vals.min(),
                "p2.5":        np.percentile(vals, 2.5),
                "p50":         np.percentile(vals, 50),
                "p97.5":       np.percentile(vals, 97.5),
                "Max":         vals.max(),
            })
    return pd.DataFrame(records)


def count_ew_events(ew_dfs: dict) -> pd.DataFrame:
    """Count 15-min records per EW type per city."""
    from spark.config import EW_DEFINITIONS, EW_COLS
    records = []
    for city, df in ew_dfs.items():
        for ew in EW_COLS:
            if ew not in df.columns:
                continue
            records.append({
                "City":  city,
                "EW":    ew,
                "Name":  EW_DEFINITIONS[ew],
                "Count": int(df[ew].sum()),
            })
    return pd.DataFrame(records)


def print_tables(table5: pd.DataFrame,
                 table6: pd.DataFrame,
                 table7: pd.DataFrame,
                 ew_counts: pd.DataFrame) -> None:
    """Pretty-print all tables to stdout."""
    print("\n── Table 5 (first 5 rows) ──")
    print(table5.head().to_string())
    print("\n── Table 6 ──")
    print(table6.round(2).to_string())
    print("\n── Table 7 (first 10 rows) ──")
    print(table7.head(10).round(2).to_string())
    print("\n── Extreme Weather Event Counts ──")
    print(ew_counts.to_string(index=False))


def save_statistics(table5: pd.DataFrame,
                    table6: pd.DataFrame,
                    table7: pd.DataFrame,
                    ew_counts: pd.DataFrame,
                    out_dir: Path) -> None:
    """Save all statistics to a single Excel file."""
    path = out_dir / "spark_statistics.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table5.to_excel(writer, sheet_name="Table5_UserSummary")
        table6.to_excel(writer, sheet_name="Table6_SectionStats")
        table7.to_excel(writer, sheet_name="Table7_WeatherStats")
        ew_counts.to_excel(writer, sheet_name="EW_EventCounts", index=False)
    logger.info("Statistics saved to %s", path)
