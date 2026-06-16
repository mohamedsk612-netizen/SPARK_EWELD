"""
spark/peak.py
-------------
Phase 9: Peak Demand Forecasting per Cluster (Transformer / MLP).

For each cluster a model is trained to predict the next-day PEAK load
(single scalar, kWh) from the previous PEAK_LOOKBACK_DAYS × 96 slots
of 15-min load + time + EW features.

Architecture selection (per cluster size):
  ≥ PEAK_SMALL_CLUSTER buildings → Transformer encoder
  <  PEAK_SMALL_CLUSTER buildings → lightweight MLP

Target is log1p-normalised before training; expm1-inverted after.
Random seed is fixed for reproducibility.
"""

import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from spark.config import (
    PEAK_LOOKBACK_DAYS, PEAK_EPOCHS, PEAK_BATCH_SIZE,
    PEAK_LR, PEAK_WEIGHT_DECAY, PEAK_PATIENCE,
    PEAK_D_MODEL, PEAK_NHEAD, PEAK_NUM_LAYERS, PEAK_DIM_FF,
    PEAK_DROPOUT, PEAK_MAX_WIN_PER_USER,
    PEAK_SMALL_CLUSTER, PEAK_SMALL_EPOCHS, PEAK_SMALL_PATIENCE,
    FORE_TRAIN_FRAC, FORE_VAL_FRAC, EW_COLS,
)
from spark.utils import load_cluster_assignments, compute_metrics

logger = logging.getLogger(__name__)

LOOKBACK_SLOTS = PEAK_LOOKBACK_DAYS * 96   # 7 × 96 = 672


def _safe_key(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name[:20]).strip("_")


def _build_peak_sequences(uids, proc_elec, ew_classified, user_location,
                           max_windows=300):
    """
    Build sliding-window sequences (stride = 1 day).
    X : (N, LOOKBACK_SLOTS, 25)  — load + time + 20 EW flags
    y : (N,)  — raw kWh peak of next day
    ew: (N,)  — bool, any EW on target day
    """
    Xs, ys, ews = [], [], []
    n_feat_ref  = None

    for uid in uids:
        df   = proc_elec.get(uid)
        city = user_location.get(uid)
        if df is None:
            continue

        load = df["Value"].ffill().fillna(0).values.astype(np.float32)
        idx  = df.index

        slot   = (idx.hour * 4 + idx.minute // 15).values
        hour_s = np.sin(2 * np.pi * slot / 96).astype(np.float32)
        hour_c = np.cos(2 * np.pi * slot / 96).astype(np.float32)
        dow_s  = np.sin(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
        dow_c  = np.cos(2 * np.pi * idx.dayofweek / 7).astype(np.float32)

        if city and city in ew_classified:
            ew_df    = ew_classified[city]
            ew_align = ew_df.reindex(idx, method="nearest", tolerance="15min")
            ew_mat   = np.zeros((len(idx), 20), dtype=np.float32)
            for i, ew in enumerate(EW_COLS):
                if ew in ew_align.columns:
                    ew_mat[:, i] = ew_align[ew].fillna(0).values.astype(np.float32)
        else:
            ew_mat = np.zeros((len(load), 20), dtype=np.float32)

        load_sc = MinMaxScaler().fit_transform(
            load.reshape(-1, 1)).ravel().astype(np.float32)

        base_feat = np.column_stack([load_sc, hour_s, hour_c, dow_s, dow_c, ew_mat])
        # shape: (T, 25) — fixed n_feat=25

        T = len(base_feat)
        if T < LOOKBACK_SLOTS + 96:
            continue

        if n_feat_ref is None:
            n_feat_ref = base_feat.shape[1]   # always 25

        win_count = 0
        t = 0
        while t + LOOKBACK_SLOTS + 96 <= T:
            x_win    = base_feat[t: t + LOOKBACK_SLOTS].copy()
            y_window = load[t + LOOKBACK_SLOTS: t + LOOKBACK_SLOTS + 96]
            peak_raw = float(y_window.max())
            ew_day   = ew_mat[t + LOOKBACK_SLOTS: t + LOOKBACK_SLOTS + 96].any()

            Xs.append(x_win)
            ys.append(peak_raw)
            ews.append(bool(ew_day))
            win_count += 1
            t += 96

            if win_count >= max_windows:
                break

    if not Xs:
        return None

    return (np.array(Xs,  dtype=np.float32),
            np.array(ys,  dtype=np.float32),
            np.array(ews, dtype=bool),
            n_feat_ref)


def _make_transformer(n_feat, device):
    import torch.nn as nn

    class PeakTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_feat, PEAK_D_MODEL)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=PEAK_D_MODEL, nhead=PEAK_NHEAD,
                dim_feedforward=PEAK_DIM_FF, dropout=PEAK_DROPOUT,
                batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer,
                                                  num_layers=PEAK_NUM_LAYERS)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(PEAK_D_MODEL, 64), nn.ReLU(),
                nn.Dropout(PEAK_DROPOUT), nn.Linear(64, 1))
        def forward(self, x):
            x = self.input_proj(x)
            x = self.encoder(x)
            x = self.pool(x.transpose(1, 2)).squeeze(-1)
            return self.head(x).squeeze(-1)

    return PeakTransformer().to(device)


def _make_mlp(n_feat, device):
    import torch.nn as nn
    in_dim = LOOKBACK_SLOTS * n_feat

    class PeakMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Dropout(PEAK_DROPOUT),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Dropout(PEAK_DROPOUT),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)

    return PeakMLP().to(device)


def _train(model, X_tr, y_tr, X_val, y_val, device, epochs, patience, lr):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    opt     = torch.optim.AdamW(model.parameters(),
                                 lr=lr, weight_decay=PEAK_WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  opt, patience=max(3, patience // 3), factor=0.5)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=PEAK_BATCH_SIZE, shuffle=True)

    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_val, pat_left, best_state = np.inf, patience, None

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        if not np.isfinite(vl):
            break
        sched.step(vl)

        if vl < best_val - 1e-6:
            best_val, pat_left = vl, patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_left -= 1
            if pat_left == 0:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _predict(model, X, device):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i+256]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def run_peak_demand_pipeline(proc_elec, ew_classified,
                              user_location, user_section_map,
                              out_dir: Path) -> None:

    print("\n" + "=" * 60)
    print("  PHASE 9: Peak Demand Forecasting (Transformer / MLP)")
    print("=" * 60)

    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("  [SKIP] PyTorch not installed — run: pip install torch")
        return

    import matplotlib.pyplot as plt

    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    PEAK_OUT = out_dir / "peak_demand"
    PEAK_OUT.mkdir(parents=True, exist_ok=True)
    print(f"  Device  : {DEVICE}")
    print(f"  Lookback: {PEAK_LOOKBACK_DAYS} days ({LOOKBACK_SLOTS} slots) → next-day peak")
    print(f"  Target  : log1p(peak kWh), inverted with expm1\n")

    cluster_users = load_cluster_assignments(out_dir)
    if cluster_users is None:
        print("  [ERROR] No cluster assignments. Run --cluster first.")
        return

    print("[9a-9b] Building sequences & training per cluster ...\n")

    results  = []
    all_pred = {}

    for cid in sorted(cluster_users.keys()):
        lbl   = cluster_users[cid]["label"]
        uids  = cluster_users[cid]["uids"]
        n_bld = len(uids)

        is_small   = n_bld < PEAK_SMALL_CLUSTER
        epochs     = PEAK_SMALL_EPOCHS   if is_small else PEAK_EPOCHS
        patience   = PEAK_SMALL_PATIENCE if is_small else PEAK_PATIENCE
        lr         = PEAK_LR * 0.5       if is_small else PEAK_LR
        model_type = "MLP" if is_small else "Transformer"

        print(f"  [C{cid} — {lbl}] ({n_bld} bldgs | {model_type} | "
              f"epochs={epochs} patience={patience}) ...")

        out = _build_peak_sequences(uids, proc_elec, ew_classified,
                                     user_location, PEAK_MAX_WIN_PER_USER)
        if out is None:
            print("    [SKIP] insufficient data")
            continue

        X, y_raw, ew, n_feat = out
        N     = len(X)
        n_tr  = int(N * FORE_TRAIN_FRAC)
        n_val = int(N * FORE_VAL_FRAC)

        X_tr      = X[:n_tr];            y_raw_tr  = y_raw[:n_tr]
        X_val_arr = X[n_tr:n_tr+n_val];  y_raw_val = y_raw[n_tr:n_tr+n_val]
        X_te      = X[n_tr+n_val:];      y_raw_te  = y_raw[n_tr+n_val:]
        ew_te     = ew[n_tr+n_val:]

        if len(X_tr) < 10 or len(X_te) < 2:
            print("    [SKIP] not enough windows")
            continue

        # log1p normalise target
        y_tr_sc  = np.log1p(y_raw_tr).astype(np.float32)
        y_val_sc = np.log1p(y_raw_val).astype(np.float32)

        model = _make_mlp(n_feat, DEVICE) if is_small else _make_transformer(n_feat, DEVICE)
        model = _train(model, X_tr, y_tr_sc, X_val_arr, y_val_sc,
                       DEVICE, epochs, patience, lr)

        pred_sc = _predict(model, X_te, DEVICE)
        pred_te = np.expm1(pred_sc).astype(np.float32)
        bad = ~np.isfinite(pred_te)
        if bad.any():
            pred_te[bad] = y_raw_te.mean()

        key = _safe_key(f"C{cid}_{lbl}")
        torch.save(model.state_dict(),
                   PEAK_OUT / f"peak_{model_type.lower()}_{key}.pt")

        m = compute_metrics(y_raw_te, pred_te)
        ew_mask = ew_te.astype(bool)
        mae_n = (np.mean(np.abs(y_raw_te[~ew_mask] - pred_te[~ew_mask]))
                 if (~ew_mask).sum() > 0 else np.nan)
        mae_e = (np.mean(np.abs(y_raw_te[ew_mask]  - pred_te[ew_mask]))
                 if ew_mask.sum() > 0 else np.nan)
        ew_deg = ((mae_e - mae_n) / (mae_n + 1e-10) * 100
                  if not np.isnan(mae_e) else np.nan)

        results.append({
            "Cluster":          f"C{cid}",
            "Label":            lbl,
            "Model":            model_type,
            "n_buildings":      n_bld,
            "n_test_windows":   len(y_raw_te),
            "MAE_kWh":          round(m["MAE"], 4),
            "RMSE_kWh":         round(m["RMSE"], 4),
            "sMAPE_%":          round(m["MAPE"], 2),
            "MAE_normal":       round(mae_n, 4) if not np.isnan(mae_n) else None,
            "MAE_ew":           round(mae_e, 4) if not np.isnan(mae_e) else None,
            "EW_degradation_%": round(ew_deg, 2) if not np.isnan(ew_deg) else None,
        })
        all_pred[cid] = (y_raw_te, pred_te, ew_te)

        ew_str = (f"  EW={mae_e:.2f}  Normal={mae_n:.2f}"
                  if not np.isnan(mae_e) else "")
        print(f"    {model_type}  MAE={m['MAE']:.2f} kWh  "
              f"RMSE={m['RMSE']:.2f}  sMAPE={m['MAPE']:.1f}%{ew_str}")

    if not results:
        print("\n  [ERROR] No clusters produced valid results.")
        return

    res_df = pd.DataFrame(results)
    res_df.to_excel(PEAK_OUT / "peak_demand_metrics.xlsx", index=False)
    logger.info("Saved peak_demand_metrics.xlsx")

    # figures (same as before — scatter, time-series, violin, EW bar, model type)
    print("\n[9c] Generating figures ...")
    cmap = plt.colormaps["tab20"].resampled(len(all_pred))

    fig, ax = plt.subplots(figsize=(9, 8))
    all_y, all_p = [], []
    for i, (cid, (y_te, pred_te, _)) in enumerate(all_pred.items()):
        lbl = cluster_users[cid]["label"]
        mdl = res_df.loc[res_df["Cluster"]==f"C{cid}", "Model"].values[0]
        ax.scatter(y_te, pred_te, s=12, alpha=0.45, color=cmap(i),
                   label=f"C{cid} [{mdl}] {lbl[:18]}")
        all_y.extend(y_te); all_p.extend(pred_te)
    lo = min(min(all_y), min(all_p)); hi = max(max(all_y), max(all_p))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect")
    ax.set_xlabel("Actual next-day peak (kWh)", fontsize=11)
    ax.set_ylabel("Predicted next-day peak (kWh)", fontsize=11)
    ax.set_title("Peak Demand — Predicted vs Actual (all clusters, test set)",
                 fontsize=12)
    ax.legend(fontsize=6, ncol=2, framealpha=0.8)
    ax.grid(ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PEAK_OUT / "p9f1_scatter_pred_vs_actual.png", dpi=150)
    plt.close()

    # summary
    print("\n" + "=" * 60)
    print("  PHASE 9 SUMMARY — Peak Demand Forecasting")
    print("=" * 60)
    hdr = (f"  {'Cluster':<38}  {'Mdl':>11}  {'MAE':>7}  "
           f"{'RMSE':>7}  {'sMAPE%':>7}  {'EW_deg%':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in res_df.iterrows():
        ew_s = (f"{r['EW_degradation_%']:+.1f}"
                if r["EW_degradation_%"] is not None else "  n/a")
        print(f"  {r['Cluster']+' — '+r['Label'][:31]:<38}  "
              f"{r['Model']:>11}  {r['MAE_kWh']:>7.2f}  "
              f"{r['RMSE_kWh']:>7.2f}  {r['sMAPE_%']:>7.1f}  {ew_s:>8}")
    print(f"\n  Overall mean MAE  : {res_df['MAE_kWh'].mean():.2f} kWh")
    print(f"  Overall mean sMAPE: {res_df['sMAPE_%'].mean():.1f}%")
    n_tf  = (res_df["Model"] == "Transformer").sum()
    n_mlp = (res_df["Model"] == "MLP").sum()
    print(f"  Models used       : {n_tf} Transformer, {n_mlp} MLP")

    print(f"\n[PHASE 9 DONE]  Outputs in: {PEAK_OUT}")
    print("    p9f1_scatter_pred_vs_actual.png")
    print("    peak_demand_metrics.xlsx")
