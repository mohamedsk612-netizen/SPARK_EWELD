"""
spark/io.py
-----------
Data loading functions for electricity, weather, and user location files.
"""

import glob
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_user_location(root: Path) -> dict:
    """
    Load user → city mapping from User Location CSVs.
    Returns {uid: city_label}
    """
    loc_dir = root / "User Location"
    mapping = {}
    for city in ["CT1", "CT2", "CT3"]:
        fpath = loc_dir / f"U_{city}.csv"
        if not fpath.exists():
            logger.warning("Location file not found: %s", fpath)
            continue
        df  = pd.read_csv(fpath)
        col = df.columns[0]
        for uid in df[col].dropna():
            mapping[str(uid).strip()] = city
    logger.info("Loaded user locations: %d users across %d cities",
                len(mapping), len(set(mapping.values())))
    return mapping


def load_electricity_data(root: Path) -> dict:
    """
    Load all electricity consumption CSVs.
    Returns {uid: DataFrame(index=DatetimeIndex, columns=['Value'])}
    """
    elec_dir  = root / "Electricity Consumption"
    csv_files = sorted(glob.glob(str(elec_dir / "**" / "*.csv"), recursive=True))
    if not csv_files:
        logger.error("No electricity CSV files found under: %s", elec_dir)
        return {}

    user_dfs = {}
    for fpath in csv_files:
        uid = Path(fpath).stem
        try:
            df = pd.read_csv(fpath, parse_dates=[0])
            df.columns = ["Time", "Value"]
            df = df.set_index("Time").sort_index()
            user_dfs[uid] = df
        except Exception as exc:
            logger.warning("Could not load %s: %s", fpath, exc)

    logger.info("Loaded electricity data for %d users", len(user_dfs))
    return user_dfs


def load_weather_data(root: Path) -> dict:
    """
    Load weather CSVs W1, W2, W3.
    Returns {city_label: DataFrame} with standardised column names.
    """
    weather_dir = root / "Weather Data"
    city_map    = {"W1": "CT1", "W2": "CT2", "W3": "CT3"}
    weather_dfs = {}

    for wfile, city in city_map.items():
        fpath = weather_dir / f"{wfile}.csv"
        if not fpath.exists():
            logger.warning("Weather file not found: %s", fpath)
            continue
        df = pd.read_csv(fpath, parse_dates=[0])
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "Time"})
        df = df.set_index("Time").sort_index()
        weather_dfs[city] = df

    logger.info("Loaded weather data for cities: %s", list(weather_dfs.keys()))
    return weather_dfs


def save_preprocessed_electricity(user_dfs: dict, out_dir: Path) -> None:
    """Save preprocessed electricity CSVs."""
    elec_out = out_dir / "preprocessed_electricity"
    elec_out.mkdir(parents=True, exist_ok=True)
    for uid, df in user_dfs.items():
        df.to_csv(elec_out / f"{uid}_preprocessed.csv")
    logger.info("Preprocessed electricity saved to %s", elec_out)


def save_preprocessed_weather(weather_dfs: dict, out_dir: Path) -> None:
    """Save preprocessed weather CSVs."""
    weather_out = out_dir / "preprocessed_weather"
    weather_out.mkdir(parents=True, exist_ok=True)
    for city, df in weather_dfs.items():
        df.to_csv(weather_out / f"{city}_preprocessed.csv")
    logger.info("Preprocessed weather saved to %s", weather_out)


def save_extreme_weather(ew_dfs: dict, out_dir: Path) -> None:
    """Save EW classification CSVs."""
    ew_out  = out_dir / "extreme_weather_classified"
    ew_out.mkdir(parents=True, exist_ok=True)
    ew_cols = [f"EW{i}" for i in range(1, 21)] + ["EW_types", "is_extreme"]
    for city, df in ew_dfs.items():
        cols = [c for c in ew_cols if c in df.columns]
        df[cols].to_csv(ew_out / f"{city}_extreme_weather.csv")
    logger.info("Extreme weather classification saved to %s", ew_out)


def build_user_section_map(root: Path, user_dfs: dict) -> dict:
    """
    Infer industrial section letter from subfolder structure.
    Folder names like 'A01', 'C10' → section letter 'A', 'C'.
    Returns {uid: section_letter}
    """
    elec_dir    = root / "Electricity Consumption"
    uid_section = {}
    for subfolder in glob.glob(str(elec_dir / "**"), recursive=True):
        sfname = Path(subfolder).name
        if len(sfname) >= 1 and sfname[0].isalpha():
            section = sfname[0].upper()
            for fpath in glob.glob(str(Path(subfolder) / "*.csv")):
                uid_section[Path(fpath).stem] = section
    # Fallback for any uid not found in folder map
    for uid in user_dfs:
        if uid not in uid_section:
            uid_section[uid] = "Unknown"
    return uid_section
