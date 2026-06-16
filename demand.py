"""
spark/demand.py
---------------
Phase 8: AI-Driven Demand Response Analysis per Cluster (v2).

Improvements over v1:
  1. Rolling 30-day baseline  — avoids seasonal drift in % deviation targets
  2. Lag features             — yesterday's deviation + 7-day rolling mean
  3. EW severity features     — co-occurrence count, consecutive EW days,
                                temperature anomaly magnitude
  4. Class-weighted loss      — corrects INCREASE/DECREASE/NEUTRAL imbalance
  5. Skip tiny clusters       — clusters < MIN_BLDGS reported separately
  6. Residual MLP             — skip connection helps separate EW vs calendar signal

Input features (37 total):
  20  EW binary flags
   5  weather means (T, DP, RH, WS, P) — normalised
   4  calendar (dow_sin/cos, month_sin/cos)
   1  yesterday's % deviation (lag-1)
   1  7-day rolling mean % deviation (lag-7 mean)
   3  severity: n_ew_types active, consecutive_ew_days, temp_anomaly
  ──
  34  + 3 severity = 37

Target  : % daily load deviation from rolling 30-day non-EW baseline
"""

import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path

from spark.config import (
    FORE_TRAIN_FRAC, FORE_VAL_FRAC, EW_COLS,
)
from spark.utils import load_cluster_assignments

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
WEATHER_COLS  = ["Temperature", "Dew Point", "Humidity", "Wind Speed", "Pressure"]
N_WEATHER     = len(WEATHER_COLS)   # 5
N_FEAT        = 20 + N_WEATHER + 4 + 2 + 3   # 34 total

DR_EPOCHS      = 200
DR_LR          = 3e-4
DR_WEIGHT_DECAY= 1e-4
DR_PATIENCE    = 25
DR_BATCH       = 128
DR_HIDDEN      = [256, 128, 64]
DR_DROPOUT     = 0.25
DR_CHANGE_THR  = 2.0        # % threshold INCREASE / DECREASE / NEUTRAL
MIN_BLDGS      = 5          # clusters below this → skip AI, report as n/a
ROLLING_WIN    = 30         # days for rolling baseline


def _safe_key(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name[:20]).strip("_")


def classify(arr, thr=DR_CHANGE_THR):
    c = np.full(len(arr), "NEUTRAL", dtype=object)
    c[arr >  thr] = "INCREASE"
    c[arr < -thr] = "DECREASE"
    return c


# ── data builder ──────────────────────────────────────────────────────────────

def _build_dr_dataset(uids, proc_elec, ew_classified, user_location):
    """
    Build daily-level dataset for one cluster.
    Returns (X, y_pct, ew_flags, class_weights_tensor) or None.
    """
    daily_rows = []

    for uid in uids:
        df   = proc_elec.get(uid)
        city = user_location.get(uid)
        if df is None or city not in ew_classified:
            continue
        ew_df = ew_classified[city]

        # ── daily total load ──────────────────────────────────────────────
        daily_load = df["Value"].resample("D").sum()

        # ── daily EW flags ────────────────────────────────────────────────
        daily_ew = pd.DataFrame(index=daily_load.index)
        for ew in EW_COLS:
            if ew in ew_df.columns:
                daily_ew[ew] = (ew_df[ew].resample("D").max()
                                .reindex(daily_load.index).fillna(0).astype(float))
            else:
                daily_ew[ew] = 0.0
        any_ew_flag = (daily_ew[list(EW_COLS)] > 0).any(axis=1)

        # ── daily weather means ───────────────────────────────────────────
        daily_weather = pd.DataFrame(index=daily_load.index)
        temp_raw = None
        for col in WEATHER_COLS:
            if col in ew_df.columns:
                raw = (ew_df[col].resample("D").mean()
                       .reindex(daily_load.index).ffill().bfill())
                if col == "Temperature":
                    temp_raw = raw.values.copy()
                lo, hi = raw.min(), raw.max()
                daily_weather[col] = ((raw - lo) / (hi - lo + 1e-10)).fillna(0)
            else:
                daily_weather[col] = 0.0
                if col == "Temperature":
                    temp_raw = np.zeros(len(daily_load))

        # ── improvement 1: rolling 30-day non-EW baseline ────────────────
        # compute rolling mean of load on non-EW days, forward-filled
        load_ser   = daily_load.copy()
        masked     = load_ser.where(~any_ew_flag)   # NaN on EW days
        rolling_bl = (masked.rolling(ROLLING_WIN, min_periods=5)
                            .mean().ffill().bfill())
        rolling_bl = rolling_bl.replace(0, np.nan).ffill().bfill()
        rolling_bl = rolling_bl.fillna(load_ser.mean())

        pct_dev = ((load_ser - rolling_bl) / (rolling_bl + 1e-6) * 100.0)

        # ── improvement 3: EW severity features ──────────────────────────
        n_ew_types = daily_ew[list(EW_COLS)].sum(axis=1)   # co-occurrence count

        # consecutive EW days
        consec = np.zeros(len(daily_load), dtype=np.float32)
        cnt = 0
        for i, flag in enumerate(any_ew_flag.values):
            cnt = cnt + 1 if flag else 0
            consec[i] = cnt

        # temperature anomaly: deviation from 30-day rolling mean temp
        if temp_raw is not None:
            temp_ser    = pd.Series(temp_raw, index=daily_load.index)
            temp_roll   = temp_ser.rolling(ROLLING_WIN, min_periods=5).mean().ffill().bfill()
            temp_anomaly= (temp_ser - temp_roll).fillna(0).values
        else:
            temp_anomaly = np.zeros(len(daily_load))

        # calendar features
        idx   = daily_load.index
        dow_s = np.sin(2 * np.pi * idx.dayofweek / 7)
        dow_c = np.cos(2 * np.pi * idx.dayofweek / 7)
        mon_s = np.sin(2 * np.pi * (idx.month - 1) / 12)
        mon_c = np.cos(2 * np.pi * (idx.month - 1) / 12)

        for i, date in enumerate(idx):
            daily_rows.append({
                "uid":          uid,
                "date":         date,
                "pct_dev":      float(pct_dev.iloc[i]),
                **{ew: float(daily_ew[ew].iloc[i]) for ew in EW_COLS},
                **{col: float(daily_weather[col].iloc[i]) for col in WEATHER_COLS},
                "dow_s":        float(dow_s[i]),
                "dow_c":        float(dow_c[i]),
                "mon_s":        float(mon_s[i]),
                "mon_c":        float(mon_c[i]),
                "n_ew_types":   float(n_ew_types.iloc[i]),
                "consec_ew":    float(consec[i]),
                "temp_anomaly": float(temp_anomaly[i]),
            })

    if not daily_rows:
        return None

    df_all = pd.DataFrame(daily_rows).sort_values(["uid", "date"])

    # ── improvement 2: lag features per building ──────────────────────────
    df_all["lag1_dev"]  = df_all.groupby("uid")["pct_dev"].shift(1).fillna(0)
    df_all["lag7_mean"] = (df_all.groupby("uid")["pct_dev"]
                           .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
                           .fillna(0))

    feat_cols = (list(EW_COLS) + WEATHER_COLS +
                 ["dow_s", "dow_c", "mon_s", "mon_c",
                  "lag1_dev", "lag7_mean",
                  "n_ew_types", "consec_ew", "temp_anomaly"])

    X        = df_all[feat_cols].values.astype(np.float32)
    y_pct    = df_all["pct_dev"].values.astype(np.float32)
    ew_flags = df_all[list(EW_COLS)].values.astype(np.float32)

    # ── improvement 4: compute class weights ─────────────────────────────
    labels   = classify(y_pct)
    classes  = ["INCREASE", "DECREASE", "NEUTRAL"]
    counts   = {c: max((labels == c).sum(), 1) for c in classes}
    total    = len(labels)
    weights  = {c: total / (3 * counts[c]) for c in classes}

    return X, y_pct, ew_flags, weights, df_all["date"].dt.strftime("%Y-%m-%d").values


# ── Residual MLP ──────────────────────────────────────────────────────────────

def _make_residual_mlp(n_in, device):
    """
    improvement 6: residual MLP with skip connection.
    Architecture: input → block1 → block2 → block3 → head
    Skip: input (projected) added to block3 output before head.
    """
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                nn.Dropout(DR_DROPOUT),
                nn.Linear(dim, dim), nn.LayerNorm(dim),
            )
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(x + self.net(x))

    class ResMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem    = nn.Sequential(
                nn.Linear(n_in, DR_HIDDEN[0]),
                nn.LayerNorm(DR_HIDDEN[0]), nn.GELU(),
                nn.Dropout(DR_DROPOUT),
            )
            self.blocks  = nn.Sequential(
                ResBlock(DR_HIDDEN[0]),
                nn.Linear(DR_HIDDEN[0], DR_HIDDEN[1]),
                nn.LayerNorm(DR_HIDDEN[1]), nn.GELU(),
                nn.Dropout(DR_DROPOUT),
                ResBlock(DR_HIDDEN[1]),
                nn.Linear(DR_HIDDEN[1], DR_HIDDEN[2]),
                nn.LayerNorm(DR_HIDDEN[2]), nn.GELU(),
            )
            # skip: project input directly to last hidden dim
            self.skip    = nn.Linear(n_in, DR_HIDDEN[2])
            self.head    = nn.Linear(DR_HIDDEN[2], 1)

        def forward(self, x):
            h = self.blocks(self.stem(x))
            h = h + self.skip(x)           # residual from input
            return self.head(h).squeeze(-1)

    return ResMLP().to(device)


# ── training ──────────────────────────────────────────────────────────────────

def _train_mlp(model, X_tr, y_tr, X_val, y_val, class_weights, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    opt     = torch.optim.AdamW(model.parameters(),
                                 lr=DR_LR, weight_decay=DR_WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=DR_EPOCHS, eta_min=DR_LR * 0.01)

    # improvement 4: weighted Huber loss — upweight EW samples
    def weighted_huber(pred, target):
        labels  = classify(target.cpu().numpy())
        weights = torch.tensor(
            [class_weights.get(l, 1.0) for l in labels],
            dtype=torch.float32, device=device)
        loss = torch.nn.functional.huber_loss(pred, target,
                                               delta=10.0, reduction="none")
        return (loss * weights).mean()

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=DR_BATCH, shuffle=True)

    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_val, pat_left, best_state = np.inf, DR_PATIENCE, None

    for epoch in range(DR_EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = weighted_huber(model(xb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.huber_loss(
                model(Xv), yv, delta=10.0).item()
        if not np.isfinite(vl):
            break

        if vl < best_val - 1e-6:
            best_val, pat_left = vl, DR_PATIENCE
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
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
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i+512]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_demand_response_pipeline(proc_elec, ew_classified,
                                  user_location, user_section_map,
                                  out_dir: Path) -> None:

    print("\n" + "=" * 60)
    print("  PHASE 8: AI Demand Response Analysis (Residual MLP v2)")
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

    DR_OUT = out_dir / "demand_response_ai"
    DR_OUT.mkdir(parents=True, exist_ok=True)

    print(f"  Device  : {DEVICE}")
    print(f"  Features: {N_FEAT} (EW flags + weather + calendar + lags + severity)")
    print(f"  Target  : % daily load deviation from rolling {ROLLING_WIN}-day baseline")
    print(f"  Model   : Residual MLP with weighted Huber loss\n")

    cluster_users = load_cluster_assignments(out_dir)
    if cluster_users is None:
        print("  [ERROR] No cluster assignments. Run --cluster first.")
        return

    print("[8a-8c] Building datasets & training per cluster ...\n")

    results  = []
    all_data = {}
    skipped  = []

    for cid in sorted(cluster_users.keys()):
        lbl   = cluster_users[cid]["label"]
        uids  = cluster_users[cid]["uids"]
        n_bld = len(uids)

        # improvement 5: skip tiny clusters
        if n_bld < MIN_BLDGS:
            print(f"  [C{cid} — {lbl}] ({n_bld} bldgs) → SKIP "
                  f"(< {MIN_BLDGS} buildings, insufficient for AI modelling)")
            skipped.append({"Cluster": f"C{cid}", "Label": lbl,
                            "n_buildings": n_bld, "Reason": "Too few buildings"})
            continue

        print(f"  [C{cid} — {lbl}] ({n_bld} bldgs) ...")

        out = _build_dr_dataset(uids, proc_elec, ew_classified, user_location)
        if out is None:
            print("    [SKIP] insufficient data")
            continue

        X, y_pct, ew_flags, class_weights, dates = out
        N     = len(X)
        n_tr  = int(N * FORE_TRAIN_FRAC)
        n_val = int(N * FORE_VAL_FRAC)

        X_tr   = X[:n_tr];            y_tr   = y_pct[:n_tr]
        X_val  = X[n_tr:n_tr+n_val];  y_val  = y_pct[n_tr:n_tr+n_val]
        X_te   = X[n_tr+n_val:];      y_te   = y_pct[n_tr+n_val:]
        ew_te  = ew_flags[n_tr+n_val:]

        if len(X_tr) < 30 or len(X_te) < 10:
            print("    [SKIP] not enough samples for train/test")
            continue

        n_feat = X.shape[1]
        model  = _make_residual_mlp(n_feat, DEVICE)
        model  = _train_mlp(model, X_tr, y_tr, X_val, y_val,
                             class_weights, DEVICE)

        pred_te = _predict(model, X_te, DEVICE)
        bad = ~np.isfinite(pred_te)
        if bad.any():
            pred_te[bad] = 0.0

        key = _safe_key(f"C{cid}_{lbl}")
        torch.save(model.state_dict(), DR_OUT / f"dr_mlp_{key}.pt")

        # metrics
        mae  = float(np.mean(np.abs(y_te - pred_te)))
        rmse = float(np.sqrt(np.mean((y_te - pred_te) ** 2)))

        true_cls = classify(y_te)
        pred_cls = classify(pred_te)
        acc      = float(np.mean(true_cls == pred_cls)) * 100

        # per-class accuracy
        acc_inc = float(np.mean(pred_cls[true_cls=="INCREASE"]=="INCREASE"))*100 \
                  if (true_cls=="INCREASE").sum() > 0 else np.nan
        acc_dec = float(np.mean(pred_cls[true_cls=="DECREASE"]=="DECREASE"))*100 \
                  if (true_cls=="DECREASE").sum() > 0 else np.nan
        acc_neu = float(np.mean(pred_cls[true_cls=="NEUTRAL"]=="NEUTRAL"))*100 \
                  if (true_cls=="NEUTRAL").sum() > 0 else np.nan

        # EW vs normal split
        any_ew_te = (ew_te > 0).any(axis=1)
        mae_ew   = (float(np.mean(np.abs(y_te[any_ew_te]  - pred_te[any_ew_te])))
                    if any_ew_te.sum() > 0 else np.nan)
        mae_norm = (float(np.mean(np.abs(y_te[~any_ew_te] - pred_te[~any_ew_te])))
                    if (~any_ew_te).sum() > 0 else np.nan)

        # class distribution in test set
        n_inc = int((true_cls=="INCREASE").sum())
        n_dec = int((true_cls=="DECREASE").sum())
        n_neu = int((true_cls=="NEUTRAL").sum())

        results.append({
            "Cluster":        f"C{cid}",
            "Label":          lbl,
            "n_buildings":    n_bld,
            "n_test_days":    len(y_te),
            "MAE_%":          round(mae, 2),
            "RMSE_%":         round(rmse, 2),
            "ClassAcc_%":     round(acc, 1),
            "Acc_INCREASE_%": round(acc_inc, 1) if not np.isnan(acc_inc) else None,
            "Acc_DECREASE_%": round(acc_dec, 1) if not np.isnan(acc_dec) else None,
            "Acc_NEUTRAL_%":  round(acc_neu, 1) if not np.isnan(acc_neu) else None,
            "n_INCREASE":     n_inc,
            "n_DECREASE":     n_dec,
            "n_NEUTRAL":      n_neu,
            "MAE_EW_%":       round(mae_ew,   2) if not np.isnan(mae_ew)   else None,
            "MAE_Normal_%":   round(mae_norm, 2) if not np.isnan(mae_norm) else None,
        })
        all_data[cid] = (y_te, pred_te, ew_te, true_cls, pred_cls)

        ew_str = (f"  EW={mae_ew:.1f}%  Norm={mae_norm:.1f}%"
                  if not np.isnan(mae_ew) else "")
        inc_s  = f"{acc_inc:.0f}%" if not np.isnan(acc_inc) else "n/a"
        dec_s  = f"{acc_dec:.0f}%" if not np.isnan(acc_dec) else "n/a"
        print(f"    MAE={mae:.1f}%  ClassAcc={acc:.1f}%  "
              f"[INC={inc_s} DEC={dec_s} NEU={acc_neu:.0f}%]{ew_str}")

    if not results:
        print("\n  [ERROR] No clusters produced valid results.")
        return

    res_df = pd.DataFrame(results)
    with pd.ExcelWriter(DR_OUT / "demand_response_metrics.xlsx") as writer:
        res_df.to_excel(writer, sheet_name="Cluster Results", index=False)
        if skipped:
            pd.DataFrame(skipped).to_excel(writer, sheet_name="Skipped", index=False)
    logger.info("Saved demand_response_metrics.xlsx")

    # ── figures ───────────────────────────────────────────────────────────────
    print("\n[8d] Generating figures ...")
    cmap = plt.colormaps["tab20"].resampled(max(len(all_data), 1))

    # p8f1: scatter predicted vs actual % deviation
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, (cid, (y_te, pred_te, _, _, _)) in enumerate(all_data.items()):
        lbl = cluster_users[cid]["label"]
        ax.scatter(y_te, pred_te, s=6, alpha=0.3,
                   color=cmap(i), label=f"C{cid} {lbl[:20]}")
    all_y = np.concatenate([v[0] for v in all_data.values()])
    all_p = np.concatenate([v[1] for v in all_data.values()])
    lo, hi = min(all_y.min(), all_p.min()), max(all_y.max(), all_p.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect")
    ax.axhline(0,  color="gray", lw=0.7, ls=":")
    ax.axvline(0,  color="gray", lw=0.7, ls=":")
    ax.axhline( DR_CHANGE_THR,  color="tomato",    lw=0.7, ls="--", alpha=0.5)
    ax.axhline(-DR_CHANGE_THR,  color="steelblue", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Actual % deviation from rolling baseline", fontsize=11)
    ax.set_ylabel("Predicted % deviation from rolling baseline", fontsize=11)
    ax.set_title("AI Demand Response — Predicted vs Actual\n"
                 "(rolling baseline, test set)", fontsize=12)
    ax.legend(fontsize=6, ncol=2, framealpha=0.8)
    ax.grid(ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(DR_OUT / "p8f1_scatter_pred_vs_actual.png", dpi=150)
    plt.close()

    # p8f2: mean predicted DR per EW type (top 4 clusters by n_buildings)
    top4 = (res_df.nlargest(4, "n_buildings")["Cluster"]
            .str.replace("C", "", regex=False).astype(int).tolist())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for ax_i, cid in enumerate(top4[:4]):
        if cid not in all_data:
            axes[ax_i].set_visible(False)
            continue
        lbl = cluster_users[cid]["label"]
        _, pred_te, ew_te, _, _ = all_data[cid]
        means = []
        for j, ew in enumerate(EW_COLS):
            mask = ew_te[:, j] > 0
            means.append(float(pred_te[mask].mean()) if mask.sum() > 0 else 0.0)
        colors = ["tomato" if v > DR_CHANGE_THR
                  else ("steelblue" if v < -DR_CHANGE_THR else "silver")
                  for v in means]
        ax = axes[ax_i]
        ax.bar(range(20), means, color=colors, edgecolor="k", lw=0.3)
        ax.axhline(0,              color="black",     lw=0.8)
        ax.axhline( DR_CHANGE_THR, color="tomato",    lw=0.8, ls="--", alpha=0.5)
        ax.axhline(-DR_CHANGE_THR, color="steelblue", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(range(20))
        ax.set_xticklabels([f"EW{i+1}" for i in range(20)],
                            rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Mean predicted % deviation")
        ax.set_title(f"C{cid} — {lbl[:30]}", fontsize=9, fontweight="bold")
        ax.grid(ls="--", alpha=0.3, axis="y")
    fig.suptitle("Predicted Demand Response by EW Event Type\n"
                 "(red=INCREASE, blue=DECREASE, grey=NEUTRAL)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DR_OUT / "p8f2_dr_per_ew_type.png", dpi=150)
    plt.close()

    # p8f3: confusion-style per-class accuracy bar chart
    fig, ax = plt.subplots(figsize=(max(10, len(res_df)*1.4), 6))
    x   = np.arange(len(res_df))
    bw  = 0.25
    inc_vals = [r["Acc_INCREASE_%"] or 0 for _, r in res_df.iterrows()]
    dec_vals = [r["Acc_DECREASE_%"] or 0 for _, r in res_df.iterrows()]
    neu_vals = [r["Acc_NEUTRAL_%"]  or 0 for _, r in res_df.iterrows()]
    ax.bar(x - bw, inc_vals, width=bw, label="INCREASE", color="tomato",
           alpha=0.85, edgecolor="k", lw=0.4)
    ax.bar(x,      neu_vals, width=bw, label="NEUTRAL",  color="silver",
           alpha=0.85, edgecolor="k", lw=0.4)
    ax.bar(x + bw, dec_vals, width=bw, label="DECREASE", color="steelblue",
           alpha=0.85, edgecolor="k", lw=0.4)
    ax.axhline(100/3, color="black", lw=1.2, ls="--",
               label=f"Random baseline ({100/3:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['Cluster']}\n({r['n_buildings']}b)" for _, r in res_df.iterrows()],
        rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Per-Class DR Accuracy (INCREASE / NEUTRAL / DECREASE)\n"
                 "per Cluster — Residual MLP", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(DR_OUT / "p8f3_per_class_accuracy.png", dpi=150)
    plt.close()

    # p8f4: MAE per cluster
    fig, ax = plt.subplots(figsize=(max(10, len(res_df)*1.4), 5))
    bars = ax.bar(range(len(res_df)), res_df["MAE_%"],
                  color="steelblue", alpha=0.85, edgecolor="k", lw=0.4)
    ax.set_xticks(range(len(res_df)))
    ax.set_xticklabels(
        [f"{r['Cluster']}\n({r['n_buildings']}b)" for _, r in res_df.iterrows()],
        rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("MAE (% deviation)")
    ax.set_title("Demand Response MAE per Cluster\n(lower = better)", fontsize=12)
    ax.grid(ls="--", alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(DR_OUT / "p8f4_mae_per_cluster.png", dpi=150)
    plt.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 8 SUMMARY — AI Demand Response (Residual MLP v2)")
    print("=" * 60)
    hdr = (f"  {'Cluster':<36}  {'MAE%':>6}  {'ClassAcc%':>9}  "
           f"{'INC%':>5}  {'DEC%':>5}  {'NEU%':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in res_df.iterrows():
        inc_s = f"{r['Acc_INCREASE_%']:>5.0f}" if r["Acc_INCREASE_%"] else "  n/a"
        dec_s = f"{r['Acc_DECREASE_%']:>5.0f}" if r["Acc_DECREASE_%"] else "  n/a"
        neu_s = f"{r['Acc_NEUTRAL_%']:>5.0f}"  if r["Acc_NEUTRAL_%"]  else "  n/a"
        print(f"  {r['Cluster']+' — '+r['Label'][:29]:<36}  "
              f"{r['MAE_%']:>6.1f}  {r['ClassAcc_%']:>9.1f}  "
              f"{inc_s}  {dec_s}  {neu_s}")

    if skipped:
        print(f"\n  Skipped (< {MIN_BLDGS} buildings):")
        for s in skipped:
            print(f"    {s['Cluster']} — {s['Label']}  ({s['n_buildings']} bldgs)")

    print(f"\n  Overall mean MAE      : {res_df['MAE_%'].mean():.1f}%")
    print(f"  Overall mean ClassAcc : {res_df['ClassAcc_%'].mean():.1f}%")
    print(f"  Random-chance baseline: {100/3:.0f}%")
    best = res_df.nlargest(3, "ClassAcc_%")
    print("\n  Best DR classification accuracy:")
    for _, r in best.iterrows():
        print(f"    {r['Cluster']} — {r['Label'][:40]}  {r['ClassAcc_%']:.1f}%")

    print(f"\n[PHASE 8 DONE]  Outputs in: {DR_OUT}")
    for f in ["p8f1_scatter_pred_vs_actual.png",
              "p8f2_dr_per_ew_type.png",
              "p8f3_per_class_accuracy.png",
              "p8f4_mae_per_cluster.png",
              "demand_response_metrics.xlsx"]:
        print(f"    {f}")
