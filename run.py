"""
run.py
------
Main entry point for the SPARK pipeline.

Usage:
    python run.py --root /path/to/EWELD --output /path/to/output [flags]

Flags:
    --cluster     Phase 5: clustering
    --forecast    Phase 6: LSTM forecasting per cluster
    --attention   Phase 7: Attention-LSTM
    --demand      Phase 8: AI Demand Response Analysis
    --peak        Phase 9: Peak Demand Forecasting
    --viz_uid     User ID for Figs 7 & 8 (default: U380)
    --viz_city    City for Figs 7 & 8   (default: CT2)
    --viz_year    Year for Figs 7 & 8   (default: 2018)
    --log         Logging level: DEBUG / INFO / WARNING (default: INFO)
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from spark.utils  import setup_logging
from spark.config import MPL_STYLE, FIG78_UID, FIG78_CITY, FIG78_YEAR


def main():
    parser = argparse.ArgumentParser(
        description="SPARK Pipeline — Liu et al. (2023)")
    parser.add_argument("--root",      required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--output",    default="./spark_output",
                        help="Output directory")
    parser.add_argument("--cluster",   action="store_true",
                        help="Run Phase 5: clustering")
    parser.add_argument("--forecast",  action="store_true",
                        help="Run Phase 6: LSTM forecasting per cluster")
    parser.add_argument("--attention", action="store_true",
                        help="Run Phase 7: Attention-LSTM")
    parser.add_argument("--demand",    action="store_true",
                        help="Run Phase 8: AI Demand Response Analysis")
    parser.add_argument("--peak",      action="store_true",
                        help="Run Phase 9: Peak Demand Forecasting")
    parser.add_argument("--viz_uid",   default=None,
                        help=f"User for Figs 7&8 (default: {FIG78_UID})")
    parser.add_argument("--viz_city",  default=None,
                        help=f"City for Figs 7&8 (default: {FIG78_CITY})")
    parser.add_argument("--viz_year",  type=int, default=None,
                        help=f"Year for Figs 7&8 (default: {FIG78_YEAR})")
    parser.add_argument("--log",       default="INFO",
                        help="Logging level (default: INFO)")
    args = parser.parse_args()

    setup_logging(args.log)
    plt.rcParams.update(MPL_STYLE)

    root    = Path(args.root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SPARK Pipeline — Liu et al. (2023), Scientific Data")
    print("=" * 60)

    # ── Phases 1-4: always run ────────────────────────────────────────────────
    from spark.io            import (load_user_location, load_electricity_data,
                                     load_weather_data, build_user_section_map)
    from spark.preprocessing import (preprocess_all_electricity,
                                     preprocess_all_weather,
                                     classify_all_cities)
    from spark.statistics    import (compute_table5, compute_table6,
                                     compute_table7, count_ew_events,
                                     print_tables, save_statistics)
    from spark.io            import (save_preprocessed_electricity,
                                     save_preprocessed_weather,
                                     save_extreme_weather)
    from spark.visualisation import (fig2_data_availability,
                                     fig3_annual_profiles,
                                     fig4_daily_mon_sun,
                                     fig5_monthly_by_dom,
                                     fig6_weekly_by_dow,
                                     fig7_weather_correlation,
                                     fig8_extreme_weather_impact)

    # Load
    user_location    = load_user_location(root)
    raw_elec         = load_electricity_data(root)
    raw_weather      = load_weather_data(root)

    if not raw_elec:
        print("[ERROR] No electricity data found. Check --root path.")
        return

    # Preprocess
    proc_elec    = preprocess_all_electricity(raw_elec)
    proc_weather = preprocess_all_weather(raw_weather)
    ew_classified = classify_all_cities(proc_weather)

    # Statistics
    user_section_map = build_user_section_map(root, proc_elec)
    for uid in proc_elec:
        if uid not in user_section_map:
            user_section_map[uid] = user_location.get(uid, "Unknown")

    table5    = compute_table5(proc_elec)
    table6    = compute_table6(proc_elec, user_section_map)
    table7    = compute_table7(proc_weather)
    ew_counts = count_ew_events(ew_classified)
    print_tables(table5, table6, table7, ew_counts)

    # Save raw outputs
    save_preprocessed_electricity(proc_elec, out_dir)
    save_preprocessed_weather(proc_weather, out_dir)
    save_extreme_weather(ew_classified, out_dir)
    save_statistics(table5, table6, table7, ew_counts, out_dir)

    # Visualisations (Figs 2-8)
    fig2_data_availability(proc_elec, user_section_map, out_dir)
    fig3_annual_profiles(proc_elec, out_dir)
    fig4_daily_mon_sun(proc_elec, out_dir)
    fig5_monthly_by_dom(proc_elec, out_dir)
    fig6_weekly_by_dow(proc_elec, out_dir)

    viz_uid  = args.viz_uid  or FIG78_UID
    viz_city = args.viz_city or FIG78_CITY
    viz_year = args.viz_year or FIG78_YEAR

    if viz_uid in proc_elec and viz_city in proc_weather:
        fig7_weather_correlation(proc_elec, proc_weather,
                                  viz_uid, viz_city, viz_year, out_dir)
        fig8_extreme_weather_impact(proc_elec, ew_classified,
                                     viz_uid, viz_city, viz_year, out_dir)
    else:
        print(f"[SKIP] Figs 7&8: uid={viz_uid} or city={viz_city} not found.")

    # ── Phase 5: Clustering ───────────────────────────────────────────────────
    if args.cluster:
        from spark.clustering import run_clustering_pipeline
        run_clustering_pipeline(proc_elec, ew_classified,
                                user_location, user_section_map, out_dir)

    # ── Phase 6: LSTM Forecasting ─────────────────────────────────────────────
    if args.forecast:
        from spark.forecasting import run_forecasting_pipeline
        run_forecasting_pipeline(proc_elec, ew_classified,
                                  user_location, user_section_map, out_dir)

    # ── Phase 7: Attention-LSTM ───────────────────────────────────────────────
    if args.attention:
        from spark.attention import run_attention_lstm_pipeline
        run_attention_lstm_pipeline(proc_elec, ew_classified,
                                     user_location, user_section_map, out_dir)

    # ── Phase 8: AI Demand Response ───────────────────────────────────────────
    if args.demand:
        from spark.demand import run_demand_response_pipeline
        run_demand_response_pipeline(proc_elec, ew_classified,
                                      user_location, user_section_map, out_dir)

    # ── Phase 9: Peak Demand Forecasting ──────────────────────────────────────
    if args.peak:
        from spark.peak import run_peak_demand_pipeline
        run_peak_demand_pipeline(proc_elec, ew_classified,
                                  user_location, user_section_map, out_dir)

    print("\n" + "=" * 60)
    print(f"  Pipeline complete. All outputs in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
