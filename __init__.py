"""
SPARK — Extreme Weather Events Load Dataset Pipeline
=====================================================
Liu et al. (2023), Scientific Data, 10, 615.
https://doi.org/10.1038/s41597-023-02503-6

Package structure:
    spark/
        config.py          — all hyperparameters & constants
        io.py              — data loading
        preprocessing.py   — data cleaning & EW classification
        statistics.py      — Tables 5-7
        visualisation.py   — Figures 2-8 (paper replication)
        clustering.py      — Phase 5: K-means / DBSCAN / SHAP
        forecasting.py     — Phase 6: LSTM per cluster
        attention.py       — Phase 7: Attention-LSTM
        demand.py          — Phase 8: Demand response analysis
        utils.py           — shared helpers
"""

__version__ = "1.0.0"
__author__  = "SPARK Pipeline"
