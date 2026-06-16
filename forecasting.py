"""
spark/forecasting.py
--------------------
Phase 6: LSTM load forecasting per cluster.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from spark.config import (FORE_LOOKBACK, FORE_HORIZON, FORE_TRAIN_FRAC,
                           FORE_VAL_FRAC, FORE_EPOCHS, FORE_BATCH_SIZE,
                           FORE_MAX_WIN_PER_USER, FORE_LSTM_HIDDEN,
                           FORE_LSTM_LAYERS, FORE_LSTM_DROPOUT)
from spark.utils  import (load_cluster_assignments, clusters_to_groups,
                           aggregate_group_sequences, train_val_test_split,
                           compute_metrics, ew_split_metrics)

logger = logging.getLogger(__name__)


def run_forecasting_pipeline(proc_elec: dict, ew_classified: dict,
                              user_location: dict, user_section_map: dict,
                              out_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 6: Load Forecasting (LSTM per Cluster)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("  [SKIP] PyTorch not installed — run: pip install torch")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    FORE_OUT = out_dir / "forecasting"
    FORE_OUT.mkdir(parents=True, exist_ok=True)
    print(f"  Device: {DEVICE}")
    print(f"  Config: lookback={FORE_LOOKBACK} steps, horizon={FORE_HORIZON} steps")

    # ── LSTM model ────────────────────────────────────────────────────────────
    class LSTMForecaster(nn.Module):
        def __init__(self, n_feat):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, FORE_LSTM_HIDDEN, FORE_LSTM_LAYERS,
                                batch_first=True, dropout=FORE_LSTM_DROPOUT)
            self.fc   = nn.Linear(FORE_LSTM_HIDDEN, FORE_HORIZON)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def train_lstm(model, X_tr, y_tr, X_val, y_val):
        model   = model.to(DEVICE)
        opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        ds      = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        dl      = DataLoader(ds, batch_size=FORE_BATCH_SIZE, shuffle=True)
        best_val, best_state, wait = np.inf, None, 0
        for _ in range(FORE_EPOCHS):
            model.train()
            for Xb, yb in dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(Xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(torch.tensor(X_val).to(DEVICE)),
                             torch.tensor(y_val).to(DEVICE)).item()
            if vl < best_val:
                best_val  = vl
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 5:
                    break
        model.load_state_dict(best_state)
        return model

    def predict(model, X_te):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(X_te).to(DEVICE)).cpu().numpy()

    # ── Load clusters ─────────────────────────────────────────────────────────
    cluster_users = load_cluster_assignments(out_dir)
    if cluster_users:
        groups = clusters_to_groups(cluster_users)
    else:
        groups = {}
        for uid in proc_elec:
            sec = user_section_map.get(uid, "Unknown")
            groups.setdefault(sec, []).append(uid)

    print(f"\n  Training on {len(groups)} groups "
          f"({sum(len(v) for v in groups.values())} buildings) ...")

    all_results  = []
    sample_plots = []

    for grp_name, uids in groups.items():
        print(f"\n  [{grp_name}] ({len(uids)} buildings) ...")
        X, y, ew_mask, n_feat = aggregate_group_sequences(
            uids, proc_elec, ew_classified, user_location,
            FORE_LOOKBACK, FORE_HORIZON, FORE_MAX_WIN_PER_USER)

        if X is None or len(X) < 20:
            print("    [SKIP] Not enough data.")
            continue

        (X_tr, y_tr, ew_tr,
         X_val, y_val, _,
         X_te, y_te, ew_te) = train_val_test_split(
             X, y, ew_mask, FORE_TRAIN_FRAC, FORE_VAL_FRAC)

        if len(X_te) < 5:
            print("    [SKIP] Too few test samples.")
            continue

        import re
        grp_key = re.sub(r"[^\w]+", "_", grp_name[:20]).strip("_")
        model   = LSTMForecaster(n_feat=n_feat)
        model   = train_lstm(model, X_tr, y_tr, X_val, y_val)
        torch.save(model.state_dict(), FORE_OUT / f"lstm_{grp_key}.pt")

        pred = predict(model, X_te)
        met  = compute_metrics(y_te.flatten(), pred.flatten())
        print(f"    LSTM  MAE={met['MAE']:.4f}  RMSE={met['RMSE']:.4f}")

        mae_n, mae_e, deg = ew_split_metrics(y_te, pred, ew_te)
        all_results.append({
            "Group": grp_name, "n_buildings": len(uids),
            "Model": "LSTM", **met,
            "MAE_normal": mae_n, "MAE_ew": mae_e,
            "EW_degradation_%": deg,
        })
        if sample_plots is not None and len(sample_plots) < 6:
            idx_p = min(5, len(X_te)-1)
            sample_plots.append({
                "grp": grp_name, "y_true": y_te[idx_p],
                "y_pred": pred[idx_p], "is_ew": bool(ew_te[idx_p]),
            })

    if not all_results:
        print("  [ERROR] No results produced.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_excel(FORE_OUT / "forecast_metrics.xlsx", index=False)
    logger.info("Saved forecast_metrics.xlsx")

    # ── Figures ───────────────────────────────────────────────────────────────
    hours_x   = np.arange(FORE_HORIZON) / 4
    COLORS_M  = {"LSTM": "steelblue"}

    # f1: Forecast examples
    n_p   = min(6, len(sample_plots))
    nrows = int(np.ceil(n_p / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4*nrows))
    axes_flat = np.array(axes).flatten()
    for i, sp in enumerate(sample_plots[:n_p]):
        ax = axes_flat[i]
        ax.plot(hours_x, sp["y_true"], color="black", lw=2.0, label="Actual")
        ax.plot(hours_x, sp["y_pred"], color="steelblue", lw=1.8,
                ls="--", label="LSTM")
        ew_str = " [EW]" if sp["is_ew"] else ""
        ax.set_title(f"{sp['grp'][:35]}{ew_str}", fontsize=8)
        ax.set_xlabel("Hours ahead", fontsize=7)
        ax.set_ylabel("Norm. load", fontsize=7)
        ax.legend(fontsize=7); ax.grid(ls="--", alpha=0.4)
    for i in range(n_p, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle("24-Hour Load Forecasts by Cluster — LSTM",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FORE_OUT / "f1_forecast_examples.png", dpi=150); plt.close()

    # f2: MAE/RMSE by cluster
    grps = results_df["Group"].unique()
    x    = np.arange(len(grps)); bw = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        sub  = results_df.set_index("Group")
        vals = [sub.loc[g, metric] if g in sub.index else np.nan for g in grps]
        ax.bar(x, vals, width=bw, color="steelblue", alpha=0.85,
               edgecolor="k", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([g[:25] for g in grps], rotation=45,
                           ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Cluster")
        ax.grid(ls="--", alpha=0.4, axis="y")
    plt.suptitle("Forecasting Error by Load Behaviour Cluster",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FORE_OUT / "f2_error_metrics.png", dpi=150); plt.close()

    # f3: EW impact
    ew_valid = results_df.dropna(subset=["MAE_ew"])
    if not ew_valid.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        x_ew = np.arange(len(ew_valid))
        axes[0].bar(x_ew - bw/4, ew_valid["MAE_normal"], width=bw/2,
                    color="steelblue", alpha=0.7, label="Normal")
        axes[0].bar(x_ew + bw/4, ew_valid["MAE_ew"], width=bw/2,
                    color="tomato", alpha=0.7, label="EW")
        axes[0].set_xticks(x_ew)
        axes[0].set_xticklabels([g[:20] for g in ew_valid["Group"]],
                                rotation=45, ha="right", fontsize=7)
        axes[0].set_title("MAE: Normal vs EW days")
        axes[0].legend(fontsize=8); axes[0].grid(ls="--", alpha=0.4, axis="y")
        axes[1].bar(x_ew, ew_valid["EW_degradation_%"],
                    color="darkorange", alpha=0.85, edgecolor="k", lw=0.3)
        axes[1].axhline(0, color="black", lw=1.0)
        axes[1].set_xticks(x_ew)
        axes[1].set_xticklabels([g[:20] for g in ew_valid["Group"]],
                                rotation=45, ha="right", fontsize=7)
        axes[1].set_title("EW Degradation % by Cluster")
        axes[1].grid(ls="--", alpha=0.4, axis="y")
        plt.suptitle("Impact of Extreme Weather on LSTM Forecast Accuracy",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FORE_OUT / "f3_ew_impact.png", dpi=150); plt.close()

    # f4: EW heatmap
    if not ew_valid.empty:
        pivot = results_df.pivot_table(
            index="Group", columns="Model",
            values="EW_degradation_%", aggfunc="mean")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, max(5, len(pivot)*0.5)))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                        center=0, ax=ax, linewidths=0.5)
            ax.set_title("EW Degradation (%) by Cluster\n"
                         "(positive = worse under EW)", fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, rotation=0)
            plt.tight_layout()
            plt.savefig(FORE_OUT / "f4_ew_heatmap.png", dpi=150); plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("  FORECASTING SUMMARY")
    print("=" * 60)
    print(results_df[["Group","MAE","RMSE","EW_degradation_%"]].round(4)
          .to_string(index=False))
    print(f"\n  Overall MAE: {results_df['MAE'].mean():.4f}")
    print(f"  EW degrades most in:")
    worst = (results_df.dropna(subset=["EW_degradation_%"])
             .nlargest(3, "EW_degradation_%")[["Group","EW_degradation_%"]])
    for _, row in worst.iterrows():
        print(f"    {row['Group'][:45]}  +{row['EW_degradation_%']:.1f}%")
    print(f"\n[PHASE 6 DONE]  Outputs in: {FORE_OUT}")
