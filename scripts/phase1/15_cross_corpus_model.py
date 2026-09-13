# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 15 — Leave-One-Corpus-Out (LOCO) Modelling
#
# Outer loop: Leave-One-Corpus-Out (Train IPVS -> Test KCL; Train KCL -> Test IPVS).
# Inner loop: 3-fold StratifiedGroupKFold on `neg_log_loss` within training corpus only.
#
# **Estimators:**
# 1. Logistic Regression (L2 / ElasticNet)
# 2. Regularized XGBoost (shallow trees)
# 3. Random Forest (Phase 0 continuity baseline)
#
# **Gate B:** LOCO AUROC lower CI bound exceeds the F0-sex baseline.
#
# **Output:** `results/phase1/loco_results.parquet`, `results/phase1/within_vs_cross_gap.parquet`,
# `figures/phase1/phase1_fig04_cross_corpus_roc.png`.

# %%
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src import metrics as M  # noqa: E402
from src import splits as S  # noqa: E402
from src import viz as V  # noqa: E402

V.use_house_style()

PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = PROCESSED / "features"
RESULTS = ROOT / "results" / "phase1"
FIGURES = ROOT / "figures" / "phase1"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

KEY_COLS = ["corpus", "subject_id", "group", "label", "task", "language", "age", "sex", "severity_hy", "severity_u2", "severity_u3", "path"]

# %% [markdown]
# ## Load Feature Matrices

# %%
tiers = {
    "tier_a": pd.read_parquet(FEATURE_DIR / "harmonized__tier_a.parquet"),
    "tier_b": pd.read_parquet(FEATURE_DIR / "harmonized__tier_b.parquet"),
    "tier_ab": pd.read_parquet(FEATURE_DIR / "harmonized__tier_ab.parquet"),
    "tier_c": pd.read_parquet(FEATURE_DIR / "harmonized__tier_c.parquet"),
}

# Define models and tuning grids
MODELS = {
    "Logistic_L2": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Logistic_ElasticNet": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=2000, random_state=42))]),
        {"clf__C": [0.05, 0.2, 1.0], "clf__l1_ratio": [0.3, 0.7]},
    ),
    "XGBoost": (
        Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1))]),
        {"clf__max_depth": [2, 3], "clf__n_estimators": [50, 100], "clf__reg_alpha": [0.1, 1.0], "clf__learning_rate": [0.05, 0.1]},
    ),
    "RandomForest": (
        Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1))]),
        {"clf__max_depth": [3, 5], "clf__min_samples_leaf": [2, 4]},
    ),
}

# %% [markdown]
# ## Run Leave-One-Corpus-Out (LOCO)

# %%
loco_rows = []
loco_predictions = {}

for tier_name, frame in tiers.items():
    meta = frame[KEY_COLS]
    X_mat = frame.iloc[:, len(KEY_COLS):].to_numpy()
    y_arr = frame["label"].to_numpy()
    groups = meta["subject_id"].to_numpy()
    corpus_arr = meta["corpus"].to_numpy()

    print(f"\n--- LOCO Evaluation for {tier_name.upper()} ({X_mat.shape[1]} features) ---")

    for model_name, (pipe, param_grid) in MODELS.items():
        # Outer loop: 2 folds (IPVS -> KCL, KCL -> IPVS)
        test_probs = np.full(len(y_arr), np.nan)

        for repeat, train_idx, test_idx in S.iter_splits(meta, y_arr, "D_leave_one_corpus_out", n_repeats=1):
            train_corpus = corpus_arr[train_idx][0]
            test_corpus = corpus_arr[test_idx][0]

            # Inner 3-fold StratifiedGroupKFold on training corpus only
            inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=inner_cv.split(X_mat[train_idx], y_arr[train_idx], groups=groups[train_idx]),
                scoring="neg_log_loss",
                n_jobs=-1,
            )
            grid.fit(X_mat[train_idx], y_arr[train_idx])
            best_model = grid.best_estimator_

            # Predict on unseen held-out corpus
            proba = best_model.predict_proba(X_mat[test_idx])[:, 1]
            test_probs[test_idx] = proba

        # Aggregate to participant level
        part = M.aggregate_to_participant(meta["subject_id"].to_numpy(), y_arr, test_probs)
        point, lo, hi = M.bootstrap_auroc_ci(part["y_true"].to_numpy(), part["y_score"].to_numpy(), n_boot=2000, seed=42)

        # Per-corpus breakdown
        part_ipvs = part[part["subject_id"].str.startswith("IPVS")]
        part_kcl = part[part["subject_id"].str.startswith("KCL")]
        auc_ipvs = M.safe_auroc(part_ipvs["y_true"].to_numpy(), part_ipvs["y_score"].to_numpy())
        auc_kcl = M.safe_auroc(part_kcl["y_true"].to_numpy(), part_kcl["y_score"].to_numpy())

        print(f"  {model_name:20s}: LOCO AUROC = {point:.3f} [{lo:.3f}, {hi:.3f}] | Test IPVS: {auc_ipvs:.3f} | Test KCL: {auc_kcl:.3f}")

        loco_rows.append(
            {
                "tier": tier_name,
                "model": model_name,
                "loco_auroc": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "auc_test_ipvs": auc_ipvs,
                "auc_test_kcl": auc_kcl,
                "n_participants": len(part),
            }
        )
        loco_predictions[(tier_name, model_name)] = part

loco_df = pd.DataFrame(loco_rows)
loco_df.to_parquet(RESULTS / "loco_results.parquet", index=False)

# %% [markdown]
# ## Within-Corpus vs Cross-Corpus Performance Gap

# %%
gap_rows = []
for tier_name, frame in tiers.items():
    meta = frame[KEY_COLS]
    X_mat = frame.iloc[:, len(KEY_COLS):].to_numpy()
    y_arr = frame["label"].to_numpy()

    for corpus in ("ipvs", "mdvr_kcl"):
        c_mask = (meta["corpus"] == corpus).to_numpy()
        X_c = X_mat[c_mask]
        y_c = y_arr[c_mask]
        meta_c = meta.iloc[c_mask].reset_index(drop=True)

        # Standard within-corpus 5x5 cross-validation
        def lr_factory(s):
            return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=s))])

        res = E.run_arm(pd.DataFrame(X_c), y_c, meta_c, "C_subject_key", n_splits=5, n_repeats=5, seed=42, estimator_factory=lr_factory)
        within_auc = res["auroc_participant"][0]

        # Corresponding cross-corpus test AUC
        loco_sub = loco_df[(loco_df["tier"] == tier_name) & (loco_df["model"] == "Logistic_L2")].iloc[0]
        cross_auc = loco_sub["auc_test_ipvs"] if corpus == "ipvs" else loco_sub["auc_test_kcl"]

        gap_rows.append(
            {
                "tier": tier_name,
                "corpus": corpus,
                "within_corpus_auroc": within_auc,
                "cross_corpus_auroc": cross_auc,
                "generalization_gap": within_auc - cross_auc,
            }
        )

gap_df = pd.DataFrame(gap_rows)
print("\nWithin-Corpus vs Cross-Corpus Generalization Gap:")
print(gap_df.round(3).to_string(index=False))
gap_df.to_parquet(RESULTS / "within_vs_cross_gap.parquet", index=False)

# %% [markdown]
# ## Figure 4 — Cross-Corpus ROC Curves & Gate B Check

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Tier AB models
for model_name, style in (("Logistic_L2", "-"), ("Logistic_ElasticNet", "--"), ("XGBoost", "-."), ("RandomForest", ":")):
    part = loco_predictions[("tier_ab", model_name)]
    fpr, tpr, _ = roc_curve(part["y_true"], part["y_score"])
    sub = loco_df[(loco_df["tier"] == "tier_ab") & (loco_df["model"] == model_name)].iloc[0]
    axes[0].plot(fpr, tpr, style, linewidth=2, label=f"{model_name} (AUROC {sub['loco_auroc']:.2f})")

axes[0].plot([0, 1], [0, 1], "k--", label="Chance (0.50)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("(a) LOCO ROC Curves (Tier AB)", fontsize=10.5, fontweight="bold")
axes[0].legend(fontsize=8, loc="lower right")
axes[0].set_aspect("equal")

# Barplot comparison of Tiers
tier_order = ["tier_a", "tier_b", "tier_ab", "tier_c"]
tier_labels = ["Tier A\n(Physiology)", "Tier B\n(CMVN-MFCC)", "Tier AB\n(Combined)", "Tier C\n(eGeMAPS)"]
lr_sub = loco_df[loco_df["model"] == "Logistic_L2"].set_index("tier").loc[tier_order]

x = np.arange(len(tier_order))
bars = axes[1].bar(x, lr_sub["loco_auroc"], width=0.45, color="#0072B2", edgecolor="white",
                  yerr=[lr_sub["loco_auroc"] - lr_sub["ci_lo"], lr_sub["ci_hi"] - lr_sub["loco_auroc"]], capsize=4)
axes[1].bar_label(bars, fmt="%.3f", padding=5, fontsize=9, fontweight="bold")

# F0 sex-baseline threshold for Gate B
f0_baseline = 0.65
axes[1].axhline(f0_baseline, color="#D55E00", linestyle=":", linewidth=1.4, label=f"F0 Sex Baseline ({f0_baseline:.2f})")
axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Chance (0.50)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(tier_labels, fontsize=9)
axes[1].set_ylabel("LOCO AUROC (95% CI)")
axes[1].set_ylim(0.4, 1.05)
axes[1].set_title("(b) Feature Tier Performance (Logistic L2)", fontsize=10.5, fontweight="bold")
axes[1].legend(fontsize=8, loc="upper right")

fig.suptitle("Leave-One-Corpus-Out Generalization (IPVS <-> MDVR-KCL)", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = FIGURES / "phase1_fig04_cross_corpus_roc.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "phase1_fig04_cross_corpus_roc.pdf", bbox_inches="tight")
print(f"Saved {fig_path}")
plt.close(fig)

print("\nStage 15 completed successfully.")
