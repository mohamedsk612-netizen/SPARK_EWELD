"""
spark/attention.py
------------------
Phase 7: Attention-LSTM with EW-conditioned output heads.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from spark.config import (FORE_LOOKBACK, FORE_HORIZON, FORE_TRAIN_FRAC,
                           FORE_VAL_FRAC, FORE_LSTM_HIDDEN, FORE_LSTM_LAYERS,
                           FORE_LSTM_DROPOUT, FORE_BATCH_SIZE,
                           FORE_MAX_WIN_PER_USER,
                           ATT_EPOCHS, ATT_LR, ATT_WEIGHT_DECAY,
                           ATT_PATIENCE, ATT_HUBER_DELTA)
from spark.utils  import (load_cluster_assignments, clusters_to_groups,
                           aggregate_group_sequences, train_val_test_split,
                           compute_metrics, ew_split_metrics)

logger = logging.getLogger(__name__)


def run_attention_lstm_pipeline(proc_elec: dict, ew_classified: dict,
                                 user_location: dict, user_section_map: dict,
                                 out_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 7: Attention-LSTM Forecasting")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("  [SKIP] PyTorch not installed.")
        return

    import matplotlib.pyplot as plt

    ATT_OUT = out_dir / "attention_lstm"
    ATT_OUT.mkdir(parents=True, exist_ok=True)
    print(f"  Device: {DEVICE}")

    # ── Attention-LSTM model ──────────────────────────────────────────────────
    class AttentionLSTM(nn.Module):
        """
        LSTM with Bahdanau-style temporal attention + EW-conditioned heads.
        The attention layer scores each hidden state so the model can focus
        on the most relevant historical timesteps.
        Two separate output heads (normal / EW) are blended by the EW flag.
        """
        def __init__(self, n_feat):
            super().__init__()
            self.lstm      = nn.LSTM(n_feat, FORE_LSTM_HIDDEN, FORE_LSTM_LAYERS,
                                     batch_first=True,
                                     dropout=FORE_LSTM_DROPOUT)
            self.attn_w    = nn.Linear(FORE_LSTM_HIDDEN, FORE_LSTM_HIDDEN)
            self.attn_v    = nn.Linear(FORE_LSTM_HIDDEN, 1, bias=False)
            self.fc_normal = nn.Linear(FORE_LSTM_HIDDEN, FORE_HORIZON)
            self.fc_ew     = nn.Linear(FORE_LSTM_HIDDEN, FORE_HORIZON)
            self.dropout   = nn.Dropout(FORE_LSTM_DROPOUT)

        def attention(self, lstm_out):
            score  = torch.tanh(self.attn_w(lstm_out))       # (B, T, H)
            score  = self.attn_v(score).squeeze(-1)           # (B, T)
            weight = F.softmax(score, dim=1).unsqueeze(2)     # (B, T, 1)
            context = (lstm_out * weight).sum(dim=1)           # (B, H)
            return context, weight.squeeze(2)

        def forward(self, x, ew_flag=None):
            out, _ = self.lstm(x)
            ctx, attn_w = self.attention(out)
            ctx    = self.dropout(ctx)
            if ew_flag is not None:
                ew_f = ew_flag.float().unsqueeze(1)           # (B, 1)
                pred = (self.fc_normal(ctx) * (1 - ew_f) +
                        self.fc_ew(ctx)     * ew_f)
            else:
                pred = self.fc_normal(ctx)
            return pred, attn_w

    # ── Training helper ───────────────────────────────────────────────────────
    def train_attention_lstm(model, X_tr, y_tr, ew_tr, X_val, y_val, ew_val):
        model   = model.to(DEVICE)
        opt     = torch.optim.Adam(model.parameters(), lr=ATT_LR,
                                    weight_decay=ATT_WEIGHT_DECAY)
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      opt, patience=3, factor=0.5)
        loss_fn = nn.HuberLoss(delta=ATT_HUBER_DELTA)
        ds  = TensorDataset(torch.tensor(X_tr),
                            torch.tensor(y_tr),
                            torch.tensor(ew_tr.astype(np.float32)))
        dl  = DataLoader(ds, batch_size=FORE_BATCH_SIZE, shuffle=True)
        best_val, best_state, wait = np.inf, None, 0

        for _ in range(ATT_EPOCHS):
            model.train()
            for Xb, yb, ewb in dl:
                Xb  = Xb.to(DEVICE)
                yb  = yb.to(DEVICE)
                ewb = ewb.to(DEVICE).bool()
                opt.zero_grad()
                pred, _ = model(Xb, ewb)
                loss_fn(pred, yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                Xv  = torch.tensor(X_val).to(DEVICE)
                yv  = torch.tensor(y_val).to(DEVICE)
                ewv = torch.tensor(ew_val.astype(np.float32)).to(DEVICE).bool()
                vl  = loss_fn(model(Xv, ewv)[0], yv).item()
            sched.step(vl)

            if vl < best_val:
                best_val   = vl
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= ATT_PATIENCE:
                    break

        model.load_state_dict(best_state)
        return model

    # ── Load clusters ─────────────────────────────────────────────────────────
    cluster_users = load_cluster_assignments(out_dir)
    if cluster_users:
        groups = clusters_to_groups(cluster_users)
    else:
        groups = {}
        for uid in proc_elec:
            sec = user_section_map.get(uid, "Unknown")
            groups.setdefault(sec, []).append(uid)

    print(f"\n  Training on {len(groups)} groups ...")

    all_results  = []
    attn_maps    = {}
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
         X_val, y_val, ew_val,
         X_te, y_te, ew_te) = train_val_test_split(
             X, y, ew_mask, FORE_TRAIN_FRAC, FORE_VAL_FRAC)

        if len(X_te) < 5:
            print("    [SKIP] Too few test samples.")
            continue

        import re
        grp_key = re.sub(r"[^\w]+", "_", grp_name[:20]).strip("_")
        model   = AttentionLSTM(n_feat=n_feat)
        model   = train_attention_lstm(model, X_tr, y_tr, ew_tr,
                                        X_val, y_val, ew_val)
        torch.save(model.state_dict(), ATT_OUT / f"att_lstm_{grp_key}.pt")

        # Predict + collect attention weights
        model.eval()
        with torch.no_grad():
            Xte_t  = torch.tensor(X_te).to(DEVICE)
            ewte_t = torch.tensor(ew_te.astype(np.float32)).to(DEVICE).bool()
            pred, attn_w = model(Xte_t, ewte_t)
            pred   = pred.cpu().numpy()
            attn_w = attn_w.cpu().numpy()   # (N_te, LOOKBACK)

        attn_maps[grp_name] = attn_w.mean(axis=0)  # mean attention per slot

        met = compute_metrics(y_te.flatten(), pred.flatten())
        print(f"    Attention-LSTM  MAE={met['MAE']:.4f}  "
              f"RMSE={met['RMSE']:.4f}")

        mae_n, mae_e, deg = ew_split_metrics(y_te, pred, ew_te)
        all_results.append({
            "Group": grp_name, "n_buildings": len(uids),
            **met,
            "MAE_normal": mae_n, "MAE_ew": mae_e,
            "EW_degradation_%": deg,
        })

        if len(sample_plots) < 6:
            idx_p = min(5, len(X_te) - 1)
            sample_plots.append({
                "grp":    grp_name,
                "y_true": y_te[idx_p],
                "y_pred": pred[idx_p],
                "is_ew":  bool(ew_te[idx_p]),
            })

    if not all_results:
        print("  [ERROR] No results produced.")
        return

    results_df = pd.DataFrame(all_results)

    # Compare with Phase 6 LSTM if available
    fore_path = out_dir / "forecasting" / "forecast_metrics.xlsx"
    if fore_path.exists():
        p6  = pd.read_excel(fore_path)
        p6_lstm = p6[p6["Model"] == "LSTM"].set_index("Group")["MAE"]
        results_df["Phase6_LSTM_MAE"] = results_df["Group"].map(p6_lstm)
        results_df["Improvement_%"]   = (
            (results_df["Phase6_LSTM_MAE"] - results_df["MAE"])
            / (results_df["Phase6_LSTM_MAE"] + 1e-10) * 100)

    results_df.to_excel(ATT_OUT / "attention_lstm_metrics.xlsx", index=False)
    logger.info("Saved attention_lstm_metrics.xlsx")

    # ── Figures ───────────────────────────────────────────────────────────────
    hours_x = np.arange(FORE_HORIZON) / 4

    # p7f1: Forecast examples
    n_p   = min(6, len(sample_plots))
    nrows = max(1, int(np.ceil(n_p / 2)))
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4*nrows))
    axes_flat = np.array(axes).flatten()
    for i, sp in enumerate(sample_plots[:n_p]):
        ax = axes_flat[i]
        ax.plot(hours_x, sp["y_true"], color="black", lw=2.0, label="Actual")
        ax.plot(hours_x, sp["y_pred"], color="steelblue", lw=1.8,
                ls="--", label="Attention-LSTM")
        ew_str = " [EW]" if sp["is_ew"] else ""
        ax.set_title(f"{sp['grp'][:35]}{ew_str}", fontsize=8)
        ax.set_xlabel("Hours ahead", fontsize=7)
        ax.set_ylabel("Norm. load", fontsize=7)
        ax.legend(fontsize=7); ax.grid(ls="--", alpha=0.4)
    for i in range(n_p, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle("Attention-LSTM: 24-Hour Forecasts by Cluster",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ATT_OUT / "p7f1_forecasts.png", dpi=150); plt.close()

    # p7f2: Attention weights heatmap
    if attn_maps:
        fig, ax = plt.subplots(figsize=(14, max(5, len(attn_maps)*0.55)))
        attn_mat   = np.array([attn_maps[g] for g in attn_maps])
        hours_back = np.arange(-FORE_LOOKBACK, 0) / 4
        im = ax.imshow(attn_mat, aspect="auto", cmap="YlOrRd",
                       extent=[hours_back[0], 0,
                                len(attn_maps)-0.5, -0.5])
        ax.set_yticks(range(len(attn_maps)))
        ax.set_yticklabels([g[:30] for g in attn_maps], fontsize=7)
        ax.set_xlabel("Hours in lookback window (0 = most recent)")
        ax.set_title("Mean Attention Weights by Cluster\n"
                     "(bright = model focuses here most)", fontsize=11)
        plt.colorbar(im, ax=ax, label="Attention weight")
        plt.tight_layout()
        plt.savefig(ATT_OUT / "p7f2_attention_heatmap.png", dpi=150)
        plt.close()

    # p7f3: MAE comparison Phase 6 vs Phase 7
    if "Phase6_LSTM_MAE" in results_df.columns:
        bw  = 0.35
        x   = np.arange(len(results_df))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - bw/2, results_df["Phase6_LSTM_MAE"], width=bw,
               color="steelblue", alpha=0.8, label="Phase 6 LSTM",
               edgecolor="k", lw=0.3)
        ax.bar(x + bw/2, results_df["MAE"], width=bw,
               color="crimson", alpha=0.8, label="Phase 7 Attention-LSTM",
               edgecolor="k", lw=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([g[:20] for g in results_df["Group"]],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("MAE (normalised)")
        ax.set_title("Phase 6 LSTM vs Phase 7 Attention-LSTM — MAE by Cluster")
        ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.4, axis="y")
        plt.tight_layout()
        plt.savefig(ATT_OUT / "p7f3_improvement.png", dpi=150); plt.close()

    # p7f4: EW impact
    ew_valid = results_df.dropna(subset=["MAE_ew"])
    if not ew_valid.empty:
        bw  = 0.35
        x   = np.arange(len(ew_valid))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - bw/2, ew_valid["MAE_normal"], width=bw,
               color="steelblue", alpha=0.7, label="Normal days")
        ax.bar(x + bw/2, ew_valid["MAE_ew"], width=bw,
               color="tomato", alpha=0.7, label="EW days")
        ax.set_xticks(x)
        ax.set_xticklabels([g[:20] for g in ew_valid["Group"]],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("MAE")
        ax.set_title("Attention-LSTM: Normal vs EW Day Accuracy")
        ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.4, axis="y")
        plt.tight_layout()
        plt.savefig(ATT_OUT / "p7f4_ew_impact.png", dpi=150); plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("  PHASE 7 SUMMARY — Attention-LSTM")
    print("=" * 60)
    cols = ["Group","MAE","RMSE","EW_degradation_%"]
    if "Improvement_%" in results_df.columns:
        cols.append("Improvement_%")
    print(results_df[cols].round(4).to_string(index=False))
    print(f"\n  Mean MAE: {results_df['MAE'].mean():.4f}")
    if "Improvement_%" in results_df.columns:
        print(f"  Mean improvement over Phase 6: "
              f"{results_df['Improvement_%'].mean():.2f}%")
    print(f"\n[PHASE 7 DONE]  Outputs in: {ATT_OUT}")
    for f in ["p7f1_forecasts.png", "p7f2_attention_heatmap.png",
              "p7f3_improvement.png", "p7f4_ew_impact.png",
              "attention_lstm_metrics.xlsx"]:
        print(f"    {f}")
