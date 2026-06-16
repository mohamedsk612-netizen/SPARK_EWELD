"""
spark/utils.py
--------------
Shared helper functions used across multiple modules.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamp and level."""
    logging.basicConfig(
        format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ── Outlier detection ─────────────────────────────────────────────────────────
def zscore_outlier_mask(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """Boolean mask where |Z-score| > z_thresh."""
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series(False, index=series.index)
    return ((series - mu) / sigma).abs() > z_thresh


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Boolean mask for IQR-based outliers."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)


# ── Forecasting metrics ───────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE as a dict."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask  = (np.abs(y_true) + np.abs(y_pred)) > 0
    smape = (np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) /
                     (np.abs(y_true[mask]) + np.abs(y_pred[mask]))) * 100
             if mask.sum() > 0 else np.nan)
    return {"MAE": mae, "RMSE": rmse, "MAPE": smape}


# ── Cluster assignment loader ─────────────────────────────────────────────────
def load_cluster_assignments(out_dir: Path) -> dict | None:
    """
    Load building_cluster_assignments.csv from Phase 5.
    Returns {cluster_id: {'label': str, 'uids': [str]}} or None.
    """
    cluster_csv = out_dir / "clustering" / "building_cluster_assignments.csv"
    if not cluster_csv.exists():
        logger.warning("No cluster assignments found at %s — run --cluster first.",
                       cluster_csv)
        return None
    df = pd.read_csv(cluster_csv)
    result = {}
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        if cid not in result:
            result[cid] = {"label": str(row["cluster_label"]), "uids": []}
        result[cid]["uids"].append(str(row["building_name"]))
    logger.info("Loaded %d clusters from %s", len(result), cluster_csv)
    return result


def clusters_to_groups(cluster_users: dict) -> dict:
    """Convert cluster dict to {group_name: [uids]} for training loops."""
    return {
        f"C{cid} — {cluster_users[cid]['label'][:35]}":
        cluster_users[cid]["uids"]
        for cid in sorted(cluster_users.keys())
    }


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(uid: str,
                    proc_elec: dict,
                    ew_classified: dict,
                    user_location: dict,
                    lookback: int = 96,
                    horizon: int  = 96,
                    max_windows: int = 500):
    """
    Build sliding-window sequences for one building.

    Returns (X, y, ew_flags, scaler, n_features) or None if too short.
    X shape : (N, lookback, n_features)
    y shape : (N, horizon)
    ew_flags: (N,) bool
    n_features = 1 load + 2 cyclic time + 1 weekend + 20 EW = 24
    """
    from sklearn.preprocessing import MinMaxScaler
    from spark.config import EW_COLS

    df   = proc_elec[uid].copy()
    city = user_location.get(uid)
    load = df["Value"].ffill().fillna(0).values.astype(np.float32)
    idx  = df.index

    # Cyclic time features
    slot   = (idx.hour * 4 + idx.minute // 15).values
    hour_s = np.sin(2 * np.pi * slot / 96).astype(np.float32)
    hour_c = np.cos(2 * np.pi * slot / 96).astype(np.float32)
    is_wkd = (idx.dayofweek >= 5).astype(np.float32)

    # EW binary flags
    if city and city in ew_classified:
        ew_df    = ew_classified[city]
        ew_align = ew_df.reindex(idx, method="nearest", tolerance="15min")
        ew_mat   = np.zeros((len(idx), 20), dtype=np.float32)
        for i, ew in enumerate(EW_COLS):
            if ew in ew_align.columns:
                ew_mat[:, i] = ew_align[ew].fillna(0).values.astype(np.float32)
    else:
        ew_mat = np.zeros((len(load), 20), dtype=np.float32)

    features = np.column_stack([load, hour_s, hour_c, is_wkd, ew_mat])
    scaler   = MinMaxScaler()
    features[:, 0:1] = scaler.fit_transform(features[:, 0:1])

    T = len(features)
    if T < lookback + horizon:
        return None

    Xs, ys, ew_flags = [], [], []
    for t in range(T - lookback - horizon + 1):
        Xs.append(features[t: t + lookback])
        ys.append(features[t + lookback: t + lookback + horizon, 0])
        ew_flags.append(ew_mat[t + lookback: t + lookback + horizon].any())

    Xs_arr = np.array(Xs, dtype=np.float32)
    ys_arr = np.array(ys, dtype=np.float32)
    ew_arr = np.array(ew_flags, dtype=bool)

    # Cap windows per building
    if len(Xs_arr) > max_windows:
        stride = max(1, len(Xs_arr) // max_windows)
        idx2   = np.arange(0, len(Xs_arr), stride)[:max_windows]
        Xs_arr, ys_arr, ew_arr = Xs_arr[idx2], ys_arr[idx2], ew_arr[idx2]

    return Xs_arr, ys_arr, ew_arr, scaler, features.shape[1]


def aggregate_group_sequences(uids: list, proc_elec: dict,
                               ew_classified: dict, user_location: dict,
                               lookback: int = 96, horizon: int = 96,
                               max_windows: int = 500):
    """
    Build and concatenate sequences for all buildings in a group.
    Returns (X, y, ew_mask, n_feat) or (None, ...) if no valid data.
    """
    Xs_all, ys_all, ew_all = [], [], []
    n_feat = None
    for uid in uids:
        res = build_sequences(uid, proc_elec, ew_classified,
                               user_location, lookback, horizon, max_windows)
        if res is None:
            continue
        Xs, ys, ew, _, nf = res
        if n_feat is None:
            n_feat = nf
        Xs_all.append(Xs); ys_all.append(ys); ew_all.append(ew)

    if not Xs_all:
        return None, None, None, None

    return (np.concatenate(Xs_all),
            np.concatenate(ys_all),
            np.concatenate(ew_all),
            n_feat)


def train_val_test_split(X, y, ew_mask,
                          train_frac=0.70, val_frac=0.15):
    """Chronological train/val/test split."""
    N     = len(X)
    n_tr  = int(N * train_frac)
    n_val = int(N * val_frac)
    return (X[:n_tr],            y[:n_tr],            ew_mask[:n_tr],
            X[n_tr:n_tr+n_val],  y[n_tr:n_tr+n_val],  ew_mask[n_tr:n_tr+n_val],
            X[n_tr+n_val:],      y[n_tr+n_val:],       ew_mask[n_tr+n_val:])


def ew_split_metrics(y_te, pred, ew_te):
    """Compute MAE for normal vs EW windows separately."""
    mae_n = (mean_absolute_error(y_te[~ew_te].flatten(), pred[~ew_te].flatten())
             if (~ew_te).sum() > 0 else np.nan)
    mae_e = (mean_absolute_error(y_te[ew_te].flatten(),  pred[ew_te].flatten())
             if ew_te.sum() > 0 else np.nan)
    deg   = ((mae_e - mae_n) / (mae_n + 1e-10) * 100
             if not np.isnan(mae_e) else np.nan)
    return mae_n, mae_e, deg
