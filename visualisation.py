"""
spark/visualisation.py
----------------------
Figures 2–8 from Liu et al. (2023), Scientific Data.
Figure implementations ported from the original notebook analysis,
adapted as importable functions for the SPARK pipeline.
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from pathlib import Path

from spark.config import (FIG3_USERS, FIG456_USERS, FIG78_UID,
                           FIG78_CITY, FIG78_YEAR)

# Suppress Times New Roman font-not-found spam and fall back gracefully
import warnings
import logging as _logging
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Use a serif fallback if Times New Roman is unavailable
import matplotlib.font_manager as _fm
_tnr_available = any("Times New Roman" in f.name for f in _fm.fontManager.ttflist)
if not _tnr_available:
    plt.rcParams["font.family"] = "DejaVu Serif"

logger = logging.getLogger(__name__)

# ── colours matching the paper ────────────────────────────────────────────────
COLOR_BLUE       = '#4472C4'
COLOR_RED        = '#C0504D'
COLOR_YELLOW     = '#F1D77E'
COLOR_ORANGE     = '#EF7A6D'
COLOR_LIGHT_BLUE = '#9DC3E7'
COLOR_PURPLE     = '#9394E7'
COLOR_LIGHT_GREEN= '#B1CE46'
COLOR_GREEN      = '#63E398'

EVENT_COLORS = [COLOR_YELLOW, COLOR_ORANGE, COLOR_LIGHT_BLUE,
                COLOR_PURPLE, COLOR_LIGHT_GREEN, COLOR_GREEN]


def _build_filter_data(user_dfs: dict) -> pd.DataFrame:
    """Build the concatenated filter_data DataFrame used by several figures."""
    frames = []
    for uid, df in user_dfs.items():
        tmp = df.copy().reset_index()
        tmp.columns = ["Time", "Value"]
        tmp.insert(0, "file", uid)
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["file", "Time", "Value"])
    fd = pd.concat(frames, ignore_index=True)
    fd["Time"] = pd.to_datetime(fd["Time"])
    return fd.set_index("Time")


# ── Fig 2: Data Availability Heatmap ─────────────────────────────────────────

def fig2_data_availability(user_dfs: dict, user_section_map: dict,
                            out_dir: Path) -> None:
    logger.info("Generating Fig 2: Data Availability ...")

    SECTION_ORDER = list("ACDEFGHIJKLMNOPQS")
    section_users  = {}
    section_counts = {}

    for uid, df in user_dfs.items():
        sec = user_section_map.get(uid, "Unknown")
        section_users.setdefault(sec, []).append(df)
        section_counts[sec] = section_counts.get(sec, 0) + 1

    sections = [s for s in SECTION_ORDER if s in section_users] or \
               sorted(section_users.keys())

    start_date = pd.Timestamp("2016-06-01")
    end_date   = pd.Timestamp("2022-08-31")
    day_index  = pd.date_range(start_date, end_date, freq="D")
    n_days     = len(day_index)
    n_sec      = len(sections)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"]   = 10

    fig, ax = plt.subplots(figsize=(14, max(6, n_sec * 0.52 + 1.5)))

    for row, sec in enumerate(sections):
        y            = n_sec - 1 - row
        covered_days = set()
        for df in section_users[sec]:
            covered_days |= set(df.index.normalize().unique())
        avail        = np.array([1.0 if d in covered_days else 0.0
                                 for d in day_index], dtype=float)
        completeness = avail.mean() * 100

        ax.imshow(avail.reshape(1, -1), aspect="auto",
                  extent=[0, n_days, y - 0.42, y + 0.42],
                  cmap="Blues", vmin=0, vmax=1, interpolation="nearest")

        user_count = section_counts.get(sec, 0)
        ax.text(-n_days * 0.008, y,
                f"{sec}\n[{user_count} users | {completeness:.1f}%]",
                ha="right", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="gray"))

    yearly   = pd.date_range(start_date, end_date, freq="YS")
    tick_pos = [(t - start_date).days for t in yearly]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([t.strftime("%b-%Y") for t in yearly],
                       rotation=30, ha="right", fontsize=8)
    ax.set_xlim(0, n_days)
    ax.set_ylim(-0.5, n_sec - 0.5)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (Daily Resolution from Jun 2016 to Aug 2022)",
                  fontsize=10, labelpad=10)
    ax.set_ylabel("Industrial Sections", fontsize=11, labelpad=15)
    ax.set_title("Fig 2: Data Availability per Industrial Section\n"
                 "(Blue = Data Available, White = Missing Data)",
                 fontsize=12, fontweight="bold", pad=15)

    axins = inset_axes(ax, width="2%", height="30%", loc="lower left",
                       bbox_to_anchor=(1.01, 0.35, 1, 1),
                       bbox_transform=ax.transAxes)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1)),
        cax=axins, orientation="vertical")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Missing", "Available"])
    cbar.set_label("Data Status", fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "fig2_data_availability.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)


# ── Fig 3: Annual Profiles ────────────────────────────────────────────────────

def fig3_annual_profiles(user_dfs: dict, out_dir: Path) -> None:
    logger.info("Generating Fig 3: Annual Profiles ...")

    filter_data = _build_filter_data(user_dfs)
    num_set = list("abcdefghijklmnopqr")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"]   = 10

    fig = plt.figure(figsize=(25, 20), dpi=150)

    for i, uid in enumerate(FIG3_USERS[:18]):
        d = filter_data[filter_data["file"] == uid]["Value"]
        ax = fig.add_subplot(6, 3, i + 1)
        ax.plot(d, "b", linewidth=0.8)
        ax.set_title(f"{num_set[i]}) {uid}", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.set_ylabel("kWh", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.text(0.5, -0.01, "Time (15 minutes interval)",
             ha="center", fontsize=16)
    fig.text(-0.01, 0.5, "Electricity consumption (kWh)",
             va="center", rotation="vertical", fontsize=16)
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=2.0)

    out_path = out_dir / "fig3_annual_profiles.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)


# ── Fig 4: Monthly Patterns ───────────────────────────────────────────────────

def fig4_daily_mon_sun(user_dfs: dict, out_dir: Path) -> None:
    logger.info("Generating Fig 4: Daily Mon vs Sun profiles ...")

    filter_data = _build_filter_data(user_dfs)
    num_set     = list("abcdefghi")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"]   = 10

    fig = plt.figure(figsize=(25, 20), dpi=150)

    for i, uid in enumerate(FIG456_USERS[:9]):
        try:
            d = filter_data[filter_data["file"] == uid].loc[
                "2018-09-01":"2018-10-01"]["Value"]
        except KeyError:
            d = pd.Series(dtype=float)

        ax = fig.add_subplot(3, 3, i + 1)
        if not d.empty:
            ax.plot(d, "b", linewidth=1)
        else:
            ax.text(0.5, 0.5, f"No data for {uid}",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{num_set[i]}) {uid}", fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.set_ylabel("kWh", fontsize=10)
        ax.grid(which="major", axis="x", alpha=0.5)

    fig.text(0.5, -0.02, "Time (15 minutes interval)",
             ha="center", fontsize=16)
    fig.text(-0.01, 0.5, "Electricity consumption (kWh)",
             va="center", rotation="vertical", fontsize=16)
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=2.0)

    out_path = out_dir / "fig4_monthly_patterns.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)


# ── Fig 5: Weekly Patterns ────────────────────────────────────────────────────

def fig5_monthly_by_dom(user_dfs: dict, out_dir: Path) -> None:
    logger.info("Generating Fig 5: Monthly patterns (by day of month) ...")

    filter_data = _build_filter_data(user_dfs)
    num_set     = list("abcdefghi")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"]   = 10

    fig = plt.figure(figsize=(25, 20), dpi=150)

    for i, uid in enumerate(FIG456_USERS[:9]):
        try:
            d = filter_data[filter_data["file"] == uid].loc[
                "2018-09-09":"2018-09-16"]["Value"]
        except KeyError:
            d = pd.Series(dtype=float)

        ax = fig.add_subplot(3, 3, i + 1)
        if not d.empty:
            ax.plot(d, "b", linewidth=1.5)
        else:
            ax.text(0.5, 0.5, f"No data for {uid}",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{num_set[i]}) {uid}", fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(byweekday=[0, 1, 2, 3, 4, 5, 6]))
        ax.set_ylabel("kWh", fontsize=10)
        ax.grid(which="major", axis="x", alpha=0.5)

    fig.text(0.5, -0.02, "Time (15 minutes interval)",
             ha="center", fontsize=16)
    fig.text(-0.01, 0.5, "Electricity consumption (kWh)",
             va="center", rotation="vertical", fontsize=16)
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=2.0)

    out_path = out_dir / "fig5_weekly_patterns.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)


# ── Fig 6: Daily Patterns (Mon vs Sun) ───────────────────────────────────────

def fig6_weekly_by_dow(user_dfs: dict, out_dir: Path) -> None:
    logger.info("Generating Fig 6: Weekly patterns (by day of week) ...")

    filter_data = _build_filter_data(user_dfs)
    num_set     = list("abcdefghi")
    idt         = pd.date_range(pd.to_datetime("00:00:00"),
                                pd.to_datetime("23:45:00"), freq="15min")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"]   = 10

    fig = plt.figure(figsize=(25, 20), dpi=150)

    for i, uid in enumerate(FIG456_USERS[:9]):
        try:
            d_mon = filter_data[filter_data["file"] == uid].loc[
                "2018-09-10 00:00:00":"2018-09-10 23:45:00"]["Value"]
            d_sun = filter_data[filter_data["file"] == uid].loc[
                "2018-09-09 00:00:00":"2018-09-09 23:45:00"]["Value"]
            d_mon = d_mon.copy(); d_mon.index = idt[:len(d_mon)]
            d_sun = d_sun.copy(); d_sun.index = idt[:len(d_sun)]
        except KeyError:
            d_mon = d_sun = pd.Series(dtype=float)

        ax = fig.add_subplot(3, 3, i + 1)
        if not d_mon.empty and not d_sun.empty:
            ax.plot(d_mon, "b", linewidth=2, label="Monday")
            ax.plot(d_sun, "r", linewidth=2, label="Sunday")
        else:
            ax.text(0.5, 0.5, f"No data for {uid}",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{num_set[i]}) {uid}", fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.set_ylabel("kWh", fontsize=10)
        ax.grid(which="major", axis="x", alpha=0.5)
        ax.legend(fontsize=10, loc="upper right")

    fig.text(0.5, -0.02, "Hours (15 minutes interval)",
             ha="center", fontsize=16)
    fig.text(-0.01, 0.5, "Electricity consumption (kWh)",
             va="center", rotation="vertical", fontsize=16)
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=2.0)

    out_path = out_dir / "fig6_daily_patterns.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)


# ── Fig 7: Weather Correlation ────────────────────────────────────────────────

def fig7_weather_correlation(user_dfs: dict, weather_dfs: dict,
                              uid=None, city=None, year=None,
                              out_dir: Path = None) -> None:
    uid  = uid  or FIG78_UID
    city = city or FIG78_CITY
    year = year or FIG78_YEAR
    if out_dir is None:
        raise ValueError("out_dir is required")
    logger.info("Generating Fig 7: Weather Correlation for %s (%d) ...", uid, year)

    if uid not in user_dfs or city not in weather_dfs:
        logger.warning("  Missing data for Fig 7 — skipping.")
        return

    start = f"{year}-01-01"
    end   = f"{year}-12-31 23:45:00"

    elec_df    = user_dfs[uid].loc[start:end, ["Value"]]
    weather_df = weather_dfs[city].loc[start:end]

    # find temperature and humidity columns (raw names from notebook)
    temp_col = next((c for c in weather_df.columns
                     if "temp" in c.lower() and "dew" not in c.lower()), None)
    hum_col  = next((c for c in weather_df.columns
                     if "humid" in c.lower()), None)

    if temp_col is None or hum_col is None:
        logger.warning("  Could not find Temperature/Humidity columns — skipping Fig 7.")
        return

    combined = pd.concat([elec_df, weather_df[[temp_col, hum_col]]], axis=1).dropna()
    combined = combined.rename(columns={
        "Value":    "Electricity\nconsumption(kWh)",
        temp_col:   "Temperature(F)",
        hum_col:    "Humidity(%)",
    })

    # clean: remove negatives and top 0.5%
    ec_col   = "Electricity\nconsumption(kWh)"
    combined = combined[combined[ec_col] >= 0]
    combined = combined[combined[ec_col] <= combined[ec_col].quantile(0.995)]

    variable_order = ["Temperature(F)", "Humidity(%)", ec_col]
    combined       = combined[variable_order]

    plt.rcParams["font.family"]    = "Times New Roman"
    plt.rcParams["figure.figsize"] = (8.6, 8.6)

    pairplot_grid = sns.pairplot(
        combined,
        vars=variable_order,
        plot_kws={"marker": ".", "linewidth": 0.5, "s": 8,
                  "alpha": 0.15, "edgecolor": "#4A78C9", "facecolor": "none"},
        diag_kws={"bins": 28, "color": "#5B84C4", "edgecolor": "#355C9A",
                  "alpha": 0.85, "linewidth": 0.4},
    )

    subplot_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)"]
    for idx, ax in enumerate(pairplot_grid.axes.flatten()):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.text(0.03, 0.95, subplot_labels[idx],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="left", va="top")

    for row in range(3):
        for col in range(3):
            cur = pairplot_grid.axes[row, col]
            if row == 2:
                cur.set_xlabel(cur.get_xlabel(), fontsize=8)
            if col == 0 and row != col:
                cur.set_ylabel(cur.get_ylabel(), fontsize=8)

    pairplot_grid.fig.suptitle(
        f"Fig. 7 The relationship between electricity consumption and "
        f"weather indications\n(temperature and humidity) of {uid} in {year}.",
        fontsize=10, y=0.96)

    plt.tight_layout()
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.08,
                        top=0.90, wspace=0.08, hspace=0.08)

    out_path = out_dir / "fig7_weather_correlation.png"
    pairplot_grid.savefig(out_path, dpi=300, bbox_inches="tight",
                          facecolor="white")
    plt.close("all")
    logger.info("  Saved: %s", out_path)


# ── Fig 8: Extreme Weather Impact ─────────────────────────────────────────────

def fig8_extreme_weather_impact(user_dfs: dict, ew_dfs: dict,
                                 uid=None, city=None, year=None,
                                 out_dir: Path = None) -> None:
    uid  = uid  or FIG78_UID
    year = year or FIG78_YEAR
    if out_dir is None:
        raise ValueError("out_dir is required")
    logger.info("Generating Fig 8: Extreme Weather Impact for %s (%d) ...",
                uid, year)

    if uid not in user_dfs:
        logger.warning("  User %s not found — skipping Fig 8.", uid)
        return

    edata = user_dfs[uid].loc[f"{year}-01-01":f"{year}-12-31 23:45:00",
                               "Value"]
    if edata.empty:
        logger.warning("  No data for %s in %d — skipping Fig 8.", uid, year)
        return

    idt  = pd.date_range(pd.to_datetime("00:00:00"),
                         pd.to_datetime("23:45:00"), freq="15min")
    ymin = max(0, int(edata.min()) - 3)
    ymax = int(edata.max()) + 4

    events = [
        ("2018-02-01", "00:04:00", "08:00:00", "Low temperature"),
        ("2018-05-31", "11:15:00", "16:15:00", "High temperature"),
        ("2018-08-27", "01:45:00", "23:45:00", "High humidity"),
        ("2018-06-29", "12:15:00", "16:30:00", "High heat and humidity"),
        ("2018-11-01", "08:00:00", "23:45:00", "Severe tropical storm"),
        ("2018-09-16", "04:00:00", "19:00:00", "Strong typhoon"),
    ]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"]   = 10

    fig = plt.figure(figsize=(20, 20), dpi=150)

    for idx, (date, start, end, name) in enumerate(events):
        ax = fig.add_subplot(3, 3, idx + 1)
        event_date = pd.Timestamp(date)
        try:
            d0 = edata.loc[event_date.strftime("%Y-%m-%d")].copy()
            d1 = edata.loc[(event_date - pd.Timedelta(days=1))
                           .strftime("%Y-%m-%d")].copy()
            d7 = edata.loc[(event_date - pd.Timedelta(days=7))
                           .strftime("%Y-%m-%d")].copy()
            d0.index = idt[:len(d0)]
            d1.index = idt[:len(d1)]
            d7.index = idt[:len(d7)]
            ax.plot(d0, "b-", linewidth=1.5, label="D-0")
            ax.plot(d1, "g-", linewidth=1.2, label="D-1")
            ax.plot(d7, "r-", linewidth=1.2, label="D-7")
            start_h = pd.to_datetime(start)
            end_h   = pd.to_datetime(end)
            ax.fill_between([start_h, end_h], ymin, ymax,
                            facecolor=EVENT_COLORS[idx], alpha=0.5)
        except KeyError:
            ax.text(0.5, 0.5, f"No data for {date}",
                    ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"{chr(97+idx)}) {name}", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Electricity (kWh)", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    # subplot g — full-year timeline
    ax_g = fig.add_subplot(3, 1, 3)
    ax_g.plot(edata, "b-", linewidth=1, label="EC")
    for idx, (date, _, _, name) in enumerate(events):
        event_date = pd.Timestamp(date)
        if event_date in edata.index:
            ax_g.axvline(event_date, color=EVENT_COLORS[idx],
                         linewidth=2, alpha=0.8)
            ax_g.text(event_date, ymax - 2, chr(97 + idx), fontsize=13,
                      ha="center", va="bottom", fontweight="bold",
                      bbox=dict(boxstyle="circle,pad=0.2",
                                facecolor="white", alpha=0.8))

    ax_g.set_title(
        "g) The time of different types of extreme weather in 2018",
        fontsize=14)
    ax_g.set_xlabel("Time", fontsize=12)
    ax_g.set_ylabel("Electricity (kWh)", fontsize=12)
    ax_g.set_ylim(ymin, ymax)
    ax_g.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_g.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_g.tick_params(axis="x", rotation=30, labelsize=10)
    ax_g.grid(True, alpha=0.3, axis="y")
    ax_g.legend(fontsize=10)

    fig.suptitle(
        f"Fig 8: Impact of Extreme Weather Events on {uid} ({year})",
        fontsize=16, y=0.98)
    plt.tight_layout()

    out_path = out_dir / "fig8_extreme_weather_impact.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)
