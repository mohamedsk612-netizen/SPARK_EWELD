"""
spark/clustering.py
-------------------
Phase 5: Industrial building clustering pipeline.
K-means + Agglomerative + DBSCAN with Gap Statistic / Elbow optimisation,
cluster refinement, t-SNE, CatBoost + SHAP explainability.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)


def run_clustering_pipeline(proc_elec: dict, ew_classified: dict,
                             user_location: dict, user_section_map: dict,
                             out_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 5: Clustering Pipeline")
    print("=" * 60)

    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                  calinski_harabasz_score, silhouette_samples,
                                  classification_report)
    from sklearn.manifold import TSNE
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns

    from spark.config import (EW_COLS, CLUSTER_K_MAX, CLUSTER_GAP_B,
                               CLUSTER_DBSCAN_SIL, CLUSTER_MIN_SPLIT,
                               SLOT_LABELS)

    CLUST_OUT = out_dir / "clustering"
    CLUST_OUT.mkdir(parents=True, exist_ok=True)
    COLORS = plt.cm.tab10.colors

    # ── 5a: Build mean 96-slot daily profiles ────────────────────────────────
    print("\n[5a] Building mean 96-slot daily profiles ...")
    records = []
    for uid, df in proc_elec.items():
        val = df["Value"].dropna()
        if len(val) == 0:
            continue
        slot_idx  = val.index.hour * 4 + val.index.minute // 15
        slot_mean = val.groupby(slot_idx).mean().reindex(range(96))
        row = {"building_id": uid}
        row.update({SLOT_LABELS[s]: (slot_mean.iloc[s]
                                     if s < len(slot_mean) else np.nan)
                    for s in range(96)})
        records.append(row)

    ts_raw = (pd.DataFrame(records)
              .set_index("building_id")[SLOT_LABELS].dropna())
    print(f"  Profile matrix: {ts_raw.shape}  (buildings × 96 slots)")

    # ── 5b: Feature engineering ───────────────────────────────────────────────
    print("\n[5b] Feature engineering ...")
    row_max = ts_raw.max(axis=1).replace(0, 1.0)
    ts_norm = ts_raw.div(row_max, axis=0)
    features = ts_norm.copy()

    # Statistical features
    features["feat_mean"] = ts_norm.mean(axis=1)
    features["feat_std"]  = ts_norm.std(axis=1)
    features["feat_min"]  = ts_norm.min(axis=1)
    features["feat_max"]  = ts_norm.max(axis=1)
    features["feat_p25"]  = ts_norm.quantile(0.25, axis=1)
    features["feat_p50"]  = ts_norm.quantile(0.50, axis=1)
    features["feat_p75"]  = ts_norm.quantile(0.75, axis=1)

    # Peak period features
    def slots_between(start, end):
        return [s for s in SLOT_LABELS if start <= s[:5] <= end]

    features["peak_early_morning"] = ts_norm[slots_between("05:00","09:45")].mean(axis=1)
    features["peak_morning"]       = ts_norm[slots_between("10:00","13:45")].mean(axis=1)
    features["peak_noon"]          = ts_norm[slots_between("14:00","16:45")].mean(axis=1)
    features["peak_evening"]       = ts_norm[slots_between("17:00","20:45")].mean(axis=1)
    features["peak_night"]         = ts_norm[slots_between("21:00","23:45")].mean(axis=1)
    features["peak_late_night"]    = ts_norm[slots_between("00:00","04:45")].mean(axis=1)

    # EW exposure features
    print("  Adding EW exposure features ...")
    ew_rows = []
    for uid in ts_norm.index:
        city  = user_location.get(uid)
        row   = {"building_id": uid}
        elec  = proc_elec.get(uid)
        if city and city in ew_classified and elec is not None and len(elec)>0:
            ew_df      = ew_classified[city]
            common_idx = ew_df.index.intersection(elec.index)
            n_total    = max(len(common_idx), 1)
            for ew in EW_COLS:
                row[f"ew_frac_{ew}"] = (ew_df.loc[common_idx, ew].sum() / n_total
                                         if ew in ew_df.columns and len(common_idx)>0
                                         else 0.0)
        else:
            for ew in EW_COLS:
                row[f"ew_frac_{ew}"] = 0.0
        ew_rows.append(row)

    ew_feat_df = (pd.DataFrame(ew_rows).set_index("building_id")
                  .reindex(ts_norm.index).fillna(0.0))
    features = pd.concat([features, ew_feat_df], axis=1)

    # Section one-hot
    sec_series  = pd.Series(user_section_map).reindex(ts_norm.index).fillna("Unknown")
    sec_dummies = pd.get_dummies(sec_series, prefix="sec").reindex(
                      ts_norm.index).fillna(0)
    features    = pd.concat([features, sec_dummies], axis=1).dropna()
    features    = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X         = features.values.astype(np.float64)
    n_samples = X.shape[0]
    print(f"  Feature matrix: {X.shape}")

    # ── 5c: Cluster count optimisation ───────────────────────────────────────
    print("\n[5c] Optimising cluster count ...")
    K_MAX   = min(n_samples - 1, CLUSTER_K_MAX)
    K_RANGE = list(range(2, K_MAX + 1))

    def eval_metrics(X, labels):
        n_u = len(set(labels)) - (1 if -1 in labels else 0)
        if n_u < 2: return np.nan, np.nan, np.nan
        return (davies_bouldin_score(X, labels),
                calinski_harabasz_score(X, labels),
                silhouette_score(X, labels))

    res = {"kmeans": {"dbi":[],"chi":[],"sil":[],"labels":{}},
           "agg":    {"dbi":[],"chi":[],"sil":[],"labels":{}}}

    for k in K_RANGE:
        for algo, obj in [
            ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
            ("agg",    AgglomerativeClustering(n_clusters=k, linkage="ward"))
        ]:
            lbl = obj.fit_predict(X)
            res[algo]["labels"][k] = lbl
            d, c, s = eval_metrics(X, lbl)
            res[algo]["dbi"].append(d)
            res[algo]["chi"].append(c)
            res[algo]["sil"].append(s)
        if k % 5 == 0:
            print(f"  k={k} done")

    # DBSCAN
    dbscan_res = []
    for eps in np.arange(0.5, 5.0, 0.25):
        lbl = DBSCAN(eps=eps, min_samples=2).fit_predict(X)
        n_c = len(set(lbl)) - (1 if -1 in lbl else 0)
        if n_c >= 2:
            mask = lbl != -1
            Xe, le = (X[mask], lbl[mask]) if mask.any() else (X, lbl)
            d, c, s = eval_metrics(Xe, le)
            dbscan_res.append({"eps":eps,"n_clusters":n_c,
                               "dbi":d,"chi":c,"sil":s,"labels":lbl})
    best_db = (max(dbscan_res, key=lambda r: r["sil"])
               if dbscan_res else None)
    if best_db:
        print(f"  Best DBSCAN: eps={best_db['eps']:.2f}, "
              f"k={best_db['n_clusters']}, SIL={best_db['sil']:.3f}")

    # Gap Statistic
    print("  Gap Statistic ...")
    rng_g = np.random.RandomState(42)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    gap_ks = list(range(2, min(16, K_MAX+1)))
    gap_vals, gap_sks = [], []

    def Wk(X, lbl):
        return sum(np.sum((X[lbl==c] - X[lbl==c].mean(0))**2)
                   for c in np.unique(lbl) if (lbl==c).sum() > 1)

    for k in gap_ks:
        lbl    = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(X)
        log_Wk = np.log(Wk(X, lbl) + 1e-10)
        refs   = []
        for _ in range(CLUSTER_GAP_B):
            Xr  = rng_g.uniform(mins, maxs, X.shape)
            lr  = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(Xr)
            refs.append(np.log(Wk(Xr, lr) + 1e-10))
        refs = np.array(refs)
        gap_vals.append(refs.mean() - log_Wk)
        gap_sks.append(np.sqrt(refs.var(ddof=1) * 1.1))
    gap_vals, gap_sks = np.array(gap_vals), np.array(gap_sks)
    gap_k = gap_ks[next((i for i in range(len(gap_vals)-1)
                         if gap_vals[i] >= gap_vals[i+1]-gap_sks[i+1]),
                        int(np.argmax(gap_vals)))]

    # Elbow
    inertias = np.array([KMeans(n_clusters=k, random_state=42, n_init=5)
                         .fit(X).inertia_ for k in K_RANGE])
    elbow_k  = K_RANGE[int(np.argmax(np.abs(np.diff(np.diff(inertias))))) + 1]
    print(f"  Gap k={gap_k}  Elbow k={elbow_k}")

    # Vote
    votes = []
    for algo in ["kmeans","agg"]:
        votes += [K_RANGE[int(np.argmax(res[algo]["sil"]))],
                  K_RANGE[int(np.argmin(res[algo]["dbi"]))],
                  K_RANGE[int(np.argmax(res[algo]["chi"]))]]
    if best_db: votes.append(best_db["n_clusters"])
    votes += [gap_k, gap_k, elbow_k]
    OPTIMAL_K = Counter(votes).most_common(1)[0][0]
    if best_db and best_db["sil"] > CLUSTER_DBSCAN_SIL:
        OPTIMAL_K = best_db["n_clusters"]
        print(f"  DBSCAN override → k={OPTIMAL_K}")
    if OPTIMAL_K > n_samples // 3:
        OPTIMAL_K = max(3, n_samples // 10)
    print(f"  OPTIMAL_K = {OPTIMAL_K}")

    # Optimisation figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for algo, name, col in [("kmeans","K-means","steelblue"),
                             ("agg","Agglomerative","darkorange")]:
        axes[0].plot(K_RANGE, res[algo]["sil"], marker="o", color=col,
                     lw=1.5, ms=4, label=name)
    axes[0].axvline(OPTIMAL_K, color="red", ls="--", lw=1.5,
                    label=f"k={OPTIMAL_K}")
    axes[0].set(xlabel="k", ylabel="Silhouette", title="Silhouette Score")
    axes[0].legend(fontsize=8); axes[0].grid(ls="--", alpha=0.4)

    axes[1].plot(K_RANGE, inertias, marker="o", color="steelblue", lw=1.5, ms=4)
    axes[1].axvline(elbow_k,  color="green", ls="--", lw=1.5,
                    label=f"Elbow k={elbow_k}")
    axes[1].axvline(OPTIMAL_K,color="red",   ls="--", lw=1.5,
                    label=f"k={OPTIMAL_K}")
    axes[1].set(xlabel="k", ylabel="Inertia", title="Elbow Method")
    axes[1].legend(fontsize=8); axes[1].grid(ls="--", alpha=0.4)

    axes[2].plot(gap_ks, gap_vals, marker="o", color="steelblue", lw=1.5, ms=4)
    axes[2].fill_between(gap_ks, gap_vals-gap_sks, gap_vals+gap_sks,
                         alpha=0.2, color="steelblue")
    axes[2].axvline(gap_k,    color="green", ls="--", lw=1.5,
                    label=f"Gap k={gap_k}")
    axes[2].axvline(OPTIMAL_K,color="red",   ls="--", lw=1.5,
                    label=f"k={OPTIMAL_K}")
    axes[2].set(xlabel="k", ylabel="Gap Statistic", title="Gap Statistic")
    axes[2].legend(fontsize=8); axes[2].grid(ls="--", alpha=0.4)

    plt.suptitle("Cluster Count Optimisation — SPARK",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(CLUST_OUT / "c2_k_optimisation.png", dpi=150,
                bbox_inches="tight"); plt.close()
    print("  Saved c2_k_optimisation.png")

    # ── 5d: Clustering + refinement ───────────────────────────────────────────
    print(f"\n[5d] Clustering k={OPTIMAL_K} + refinement ...")
    km_labels  = res["kmeans"]["labels"][OPTIMAL_K].astype(int)
    agg_labels = res["agg"]["labels"][OPTIMAL_K]

    sil_samp = silhouette_samples(X, km_labels)
    sorted_c = sorted({c: sil_samp[km_labels==c].mean()
                       for c in np.unique(km_labels)},
                      key=lambda c: sil_samp[km_labels==c].mean())
    new_id   = int(km_labels.max()) + 1
    for worst in sorted_c[:2]:
        mask = km_labels == worst
        if mask.sum() < CLUSTER_MIN_SPLIT: continue
        sub  = KMeans(n_clusters=2, random_state=42,
                      n_init=10).fit_predict(X[mask])
        if (sub==0).sum()<2 or (sub==1).sum()<2: continue
        km_labels[mask] = np.where(sub==0, worst, new_id)
        print(f"  Split C{worst} → C{worst}+C{new_id}"); new_id += 1

    unique_lbl = sorted(np.unique(km_labels))
    lmap       = {old:new for new,old in enumerate(unique_lbl)}
    km_labels  = np.array([lmap[l] for l in km_labels])
    REFINED_K  = len(unique_lbl)
    print(f"  Refined k={REFINED_K}")

    ts_arr    = ts_norm.loc[features.index].values
    time_cols = SLOT_LABELS

    # Cluster labelling
    def label_cluster(avg, mean_val, all_means):
        pi = int(np.argmax(avg)); ph = int(time_cols[pi][:2])
        cp = avg.copy(); cp[max(0,pi-4):pi+5] = 0
        sh = int(time_cols[int(np.argmax(cp))][:2])
        thirds = sorted(all_means)
        lth = thirds[len(thirds)//3]
        hth = thirds[2*len(thirds)//3]
        tier = ("Low"    if mean_val <= lth else
                "High"   if mean_val >= hth else "Medium")
        if 5<=ph<=9:   return f"{tier}-Activity Early Shift"
        if 10<=ph<=13: return f"{tier}-Activity Mid-Morning Ops"
        if 14<=ph<=16: return f"{tier}-Activity Afternoon Ops"
        if 17<=ph<=18:
            if 6<=sh<=10 and cp.max()>0.3*avg.max():
                return "Dual-Shift (Morning & Evening)"
            return f"{tier}-Activity Early Evening Ops"
        if 19<=ph<=20: return f"{tier}-Activity Evening Ops"
        if ph==21:     return f"{tier}-Activity Late Evening Ops"
        if ph>=22 or ph<=4: return "24/7 Night Operations"
        return "Flat / Continuous Operations"

    all_m         = [float(ts_arr[km_labels==c].mean()) for c in range(REFINED_K)]
    cluster_names = {}; seen = {}
    for c in range(REFINED_K):
        name = label_cluster(ts_arr[km_labels==c].mean(0), all_m[c], all_m)
        if name in seen:
            o  = seen[name]
            po = time_cols[int(np.argmax(ts_arr[km_labels==o].mean(0)))][:5]
            pc = time_cols[int(np.argmax(ts_arr[km_labels==c].mean(0)))][:5]
            cluster_names[o] = f"{name} (peak {po})"
            cluster_names[c] = f"{name} (peak {pc})"
            seen[cluster_names[o]] = o; seen[cluster_names[c]] = c
        else:
            cluster_names[c] = name; seen[name] = c

    print("  Cluster labels:")
    for c, name in cluster_names.items():
        print(f"    C{c:2d} [{(km_labels==c).sum():3d} bldgs]  {name}")

    # Save assignments CSV
    building_ids = list(features.index)
    pd.DataFrame({
        "building_name": building_ids,
        "cluster_id":    km_labels,
        "cluster_label": [cluster_names[c] for c in km_labels],
        "section": [user_section_map.get(u,"Unknown") for u in building_ids],
        "city":    [user_location.get(u,"Unknown")    for u in building_ids],
    }).sort_values(["cluster_id","building_name"]).reset_index(drop=True)\
      .to_csv(CLUST_OUT / "building_cluster_assignments.csv", index=False)
    print("  Saved building_cluster_assignments.csv")

    # ── 5e: Visualisations ────────────────────────────────────────────────────
    print("\n[5e] Visualisations ...")
    tick_step = 4

    # c1: avg daily profiles
    fig, ax = plt.subplots(figsize=(14, 5))
    for uid in list(ts_raw.index[:50]):
        ax.plot(range(96), ts_raw.loc[uid].values,
                color="#cc5500", alpha=0.25, lw=0.5)
    ax.plot(range(96), ts_raw.mean().values, color="navy", lw=2.5,
            label="Grand mean")
    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([SLOT_LABELS[s] for s in range(0, 96, 4)],
                       rotation=45, fontsize=6)
    ax.set(xlabel="Time of Day", ylabel="Energy Consumption (kWh)",
           title="Mean Daily Load Profiles — All Industrial Buildings")
    ax.legend(); plt.tight_layout()
    plt.savefig(CLUST_OUT / "c1_avg_daily_profiles.png", dpi=150); plt.close()

    # c3: similarity matrix
    mat = np.zeros((REFINED_K, OPTIMAL_K), dtype=int)
    for a, b in zip(km_labels, agg_labels):
        if 0<=a<REFINED_K and 0<=b<OPTIMAL_K: mat[a,b] += 1
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(pd.DataFrame(mat), annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title(f"K-means (k={REFINED_K}) vs Agglomerative (k={OPTIMAL_K})")
    ax.set_xlabel("Agglomerative"); ax.set_ylabel("K-means")
    plt.tight_layout()
    plt.savefig(CLUST_OUT / "c3_similarity_matrix.png", dpi=150); plt.close()

    # c4: t-SNE
    perp = min(30, n_samples-1)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perp,
                max_iter=500, learning_rate="auto",
                init="pca").fit_transform(X)
    fig, ax = plt.subplots(figsize=(10,8))
    for c in range(REFINED_K):
        mask = km_labels==c
        ax.scatter(X_2d[mask,0], X_2d[mask,1],
                   color=COLORS[c%len(COLORS)], s=100, alpha=0.85,
                   edgecolors="k", linewidths=0.5,
                   label=f"C{c}: {cluster_names[c]} (n={mask.sum()})")
    ax.set_title(f"t-SNE — Industrial Buildings (k={REFINED_K})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.grid(ls="--", alpha=0.4); plt.tight_layout()
    plt.savefig(CLUST_OUT / "c4_tsne.png", dpi=150); plt.close()

    # c5: cluster medians
    fig, ax = plt.subplots(figsize=(15,6))
    for i in range(len(ts_arr)):
        ax.plot(range(96), ts_arr[i],
                color=COLORS[km_labels[i]%len(COLORS)], alpha=0.12, lw=0.5)
    for c in range(REFINED_K):
        mask = km_labels==c
        ax.plot(range(96), np.median(ts_arr[mask], axis=0),
                color=COLORS[c%len(COLORS)], lw=3.0, ls="--",
                label=f"C{c}: {cluster_names[c]} (n={mask.sum()})", zorder=5)
    ax.set_xticks(range(0, 96, tick_step))
    ax.set_xticklabels([SLOT_LABELS[s] for s in range(0,96,tick_step)],
                       rotation=45, fontsize=7)
    ax.set(xlabel="Time of Day", ylabel="Normalised Consumption",
           title=f"Daily Load Profiles — Cluster Medians (k={REFINED_K})")
    ax.legend(fontsize=6, ncol=2, framealpha=0.85)
    ax.grid(ls="--", alpha=0.35); plt.tight_layout()
    plt.savefig(CLUST_OUT / "c5_cluster_medians.png", dpi=150); plt.close()

    # c6: individual cluster plots
    ncols = 2; nrows = int(np.ceil(REFINED_K/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5*nrows))
    axes_flat = np.array(axes).flatten()
    for c in range(REFINED_K):
        ax = axes_flat[c]; mask = km_labels==c
        for row in ts_arr[mask]:
            ax.plot(range(96), row, color="#f0a040", alpha=0.20, lw=0.6)
        ax.plot(range(96), ts_arr[mask].mean(0), color="royalblue",
                lw=2.5, ls="--", label="Mean", zorder=5)
        ax.plot(range(96), np.median(ts_arr[mask],0), color="crimson",
                lw=2.5, label="Median", zorder=6)
        ax.set_title(f'C{c} — "{cluster_names[c]}" (n={mask.sum()})',
                     fontsize=8, fontweight="bold")
        ax.set_xticks(range(0, 96, tick_step*2))
        ax.set_xticklabels([SLOT_LABELS[s] for s in range(0,96,tick_step*2)],
                           rotation=45, fontsize=6)
        ax.legend(fontsize=7); ax.grid(ls="--", alpha=0.35)
    for c in range(REFINED_K, len(axes_flat)): axes_flat[c].set_visible(False)
    fig.suptitle(f"Industrial Building Cluster Profiles (k={REFINED_K})",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(CLUST_OUT / "c6_cluster_individual.png", dpi=150); plt.close()

    # c7: EW exposure by cluster
    ew_cols_p = [f"ew_frac_EW{i}" for i in range(1,21)
                 if f"ew_frac_EW{i}" in features.columns]
    if ew_cols_p:
        ew_arr = features[ew_cols_p].values
        fig, ax = plt.subplots(figsize=(14,5))
        x = np.arange(len(ew_cols_p)); bw = 0.8/REFINED_K
        for c in range(REFINED_K):
            mask = km_labels==c
            ax.bar(x + c*bw, ew_arr[mask].mean(0), width=bw,
                   color=COLORS[c%len(COLORS)], alpha=0.8,
                   label=f"C{c}: {cluster_names[c][:25]}")
        ax.set_xticks(x + bw*(REFINED_K-1)/2)
        ax.set_xticklabels([c.replace("ew_frac_","") for c in ew_cols_p],
                           rotation=45, fontsize=8)
        ax.set(xlabel="EW Type",
               ylabel="Mean Fraction of Time Under EW",
               title="Extreme Weather Exposure by Cluster")
        ax.legend(fontsize=6, ncol=2, framealpha=0.85)
        plt.tight_layout()
        plt.savefig(CLUST_OUT / "c7_ew_exposure_by_cluster.png", dpi=150)
        plt.close()
        print("  Saved c7_ew_exposure_by_cluster.png")

    # ── 5f: CatBoost + SHAP ───────────────────────────────────────────────────
    print("\n[5f] CatBoost + SHAP ...")
    try:
        from catboost import CatBoostClassifier, Pool
        import shap

        X_df = pd.DataFrame(X, columns=features.columns)
        y    = km_labels
        can_strat = (pd.Series(y).value_counts() >= 2).all()
        test_sz   = min(0.20, max(0.10, 1-(REFINED_K*2)/n_samples))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_df, y, test_size=test_sz, random_state=42,
            stratify=y if can_strat else None)

        model = CatBoostClassifier(iterations=500, learning_rate=0.05,
                                    depth=6, random_seed=42,
                                    verbose=0, thread_count=-1)
        model.fit(X_tr, y_tr, eval_set=(X_te,y_te),
                  early_stopping_rounds=30)

        y_pred = model.predict(X_te).flatten().astype(int)
        print("\n  CatBoost Classification Report:")
        print(classification_report(y_te, y_pred))

        feat_names = model.feature_names_
        X_samp     = X_te.sample(min(200,len(X_te)), random_state=42)[feat_names]
        y_samp     = y_te[X_te.index.get_indexer(X_samp.index)]
        shap_raw   = model.get_feature_importance(
            data=Pool(X_samp, label=y_samp),
            type="ShapValues", verbose=False)
        n_f        = len(feat_names)
        shap_vals  = ([shap_raw[:,c,:n_f] for c in range(shap_raw.shape[1])]
                      if shap_raw.ndim==3 else [shap_raw[:,:n_f]])
        dom_c      = int(np.argmax([np.abs(sv).mean() for sv in shap_vals]))

        shap.summary_plot(shap_vals[dom_c], X_samp.values,
                          feature_names=feat_names, max_display=20, show=False)
        plt.title(f"SHAP Summary — Dominant Class C{dom_c}")
        plt.tight_layout()
        plt.savefig(CLUST_OUT / "c8_shap_summary.png", dpi=150,
                    bbox_inches="tight"); plt.close()

        ncols = 2; nrows = int(np.ceil(REFINED_K/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
        axes_flat = np.array(axes).flatten()
        for c in range(REFINED_K):
            sv  = shap_vals[c if c<len(shap_vals) else 0]
            top = (pd.Series(np.abs(sv).mean(0), index=feat_names)
                   .sort_values(ascending=False).head(10).sort_values())
            top.plot(kind="barh", ax=axes_flat[c],
                     color=COLORS[c%len(COLORS)],
                     edgecolor="k", linewidth=0.4)
            axes_flat[c].set_title(f"SHAP Top-10 — C{c}: "
                                    f"{cluster_names[c][:40]}", fontsize=8)
            axes_flat[c].set_xlabel("Mean |SHAP value|")
        for c in range(REFINED_K, len(axes_flat)):
            axes_flat[c].set_visible(False)
        plt.tight_layout()
        plt.savefig(CLUST_OUT / "c9_shap_per_cluster.png", dpi=150)
        plt.close()
        print("  Saved c8_shap_summary.png  c9_shap_per_cluster.png")

    except ImportError:
        print("  [SKIP] catboost/shap not installed — "
              "run: pip install catboost shap")

    print(f"\n[CLUSTERING DONE] k={OPTIMAL_K} → refined k={REFINED_K}")
    print(f"  Outputs in: {CLUST_OUT}")
    for f in ["c1_avg_daily_profiles.png","c2_k_optimisation.png",
              "c3_similarity_matrix.png", "c4_tsne.png",
              "c5_cluster_medians.png",   "c6_cluster_individual.png",
              "c7_ew_exposure_by_cluster.png",
              "c8_shap_summary.png",      "c9_shap_per_cluster.png",
              "building_cluster_assignments.csv"]:
        print(f"    {f}")
