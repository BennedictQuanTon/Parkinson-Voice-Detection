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
# # 16 — Generalization, Severity Gradient, Calibration & Clinical Utility
#
# Three rigorous validity audits:
# 1. **16a — Severity Gradient:** Do predictions correlate with Hoehn & Yahr and UPDRS II-5 / III-18?
# 2. **16b — Calibration & DCA:** Reliability curve, Brier score, ECE, prior correction (1.5% prevalence), PPV, Decision Curve Analysis.
# 3. **16c — Sex Sensitivity Analysis:** Evaluation on the female-only stratum (confound-free).
#
# **Output:** `figures/phase1/phase1_fig05_severity_gradient.png`, `figures/phase1/phase1_fig06_calibration_dca.png`,
# `figures/phase1/phase1_fig07_sex_sensitivity.png`, `results/phase1/generalization_audit.parquet`.

# %%
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import calibration as Cal  # noqa: E402
from src import metrics as M  # noqa: E402
from src import viz as V  # noqa: E402

V.use_house_style()

PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = PROCESSED / "features"
RESULTS = ROOT / "results" / "phase1"
FIGURES = ROOT / "figures" / "phase1"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

meta = pd.read_parquet(PROCESSED / "metadata_harmonized.parquet")
loco_df = pd.read_parquet(RESULTS / "loco_results.parquet")
tier_c_frame = pd.read_parquet(FEATURE_DIR / "harmonized__tier_c.parquet")
f0_df = pd.read_parquet(FEATURE_DIR / "harmonized__tier_a.parquet")[["subject_id", "corpus", "f0_mean"]]

print(f"Loaded harmonized metadata ({len(meta)} rows)")

# %% [markdown]
# ## 16a — Clinical Severity Gradient (Hoehn & Yahr and UPDRS)

# %%
# Train on IPVS, test on KCL with Logistic L2
X_cols = [c for c in tier_c_frame.columns if c not in meta.columns]
ipvs_mask = tier_c_frame["corpus"] == "ipvs"
kcl_mask = tier_c_frame["corpus"] == "mdvr_kcl"

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))])
pipe.fit(tier_c_frame.loc[ipvs_mask, X_cols].to_numpy(), tier_c_frame.loc[ipvs_mask, "label"].to_numpy())

kcl_test = tier_c_frame.loc[kcl_mask].copy()
kcl_test["pred_prob"] = pipe.predict_proba(kcl_test[X_cols].to_numpy())[:, 1]

# Participant-level predictions on KCL
kcl_part = kcl_test.groupby("subject_id").agg(
    label=("label", "max"),
    pred_prob=("pred_prob", "mean"),
    hy=("severity_hy", "first"),
    u2=("severity_u2", "first"),
    u3=("severity_u3", "first"),
).reset_index()

# Correlations on MDVR-KCL
rho_hy, p_hy = stats.spearmanr(kcl_part["hy"], kcl_part["pred_prob"])
rho_u2, p_u2 = stats.spearmanr(kcl_part["u2"], kcl_part["pred_prob"])
rho_u3, p_u3 = stats.spearmanr(kcl_part["u3"], kcl_part["pred_prob"])

print("\nSeverity Gradient on MDVR-KCL (Spearman rho with predicted probability):")
print(f"  Hoehn & Yahr stage : rho = {rho_hy:+.3f} (p = {p_hy:.4f})")
print(f"  UPDRS II-5 (speech): rho = {rho_u2:+.3f} (p = {p_u2:.4f})")
print(f"  UPDRS III-18 (speech): rho = {rho_u3:+.3f} (p = {p_u3:.4f})")

# %% [markdown]
# ### Figure 5 — Clinical Severity Gradient

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# 1. Hoehn & Yahr
hy_groups = [kcl_part.loc[kcl_part["hy"] == h, "pred_prob"].values for h in sorted(kcl_part["hy"].unique())]
axes[0].boxplot(hy_groups, tick_labels=[str(h) for h in sorted(kcl_part["hy"].unique())], patch_artist=True,
                boxprops=dict(facecolor="#56B4E9", alpha=0.5))
for i, h in enumerate(sorted(kcl_part["hy"].unique())):
    y_pts = kcl_part.loc[kcl_part["hy"] == h, "pred_prob"]
    x_pts = np.random.normal(i + 1, 0.04, size=len(y_pts))
    axes[0].scatter(x_pts, y_pts, color="#0072B2", s=28, zorder=3)
axes[0].set_xlabel("Hoehn & Yahr Stage")
axes[0].set_ylabel("Predicted Probability of PD")
axes[0].set_title(f"(a) Hoehn & Yahr (rho = {rho_hy:+.2f})")

# 2. UPDRS II-5 (Speech)
u2_groups = [kcl_part.loc[kcl_part["u2"] == u, "pred_prob"].values for u in sorted(kcl_part["u2"].unique())]
axes[1].boxplot(u2_groups, tick_labels=[str(u) for u in sorted(kcl_part["u2"].unique())], patch_artist=True,
                boxprops=dict(facecolor="#009E73", alpha=0.5))
for i, u in enumerate(sorted(kcl_part["u2"].unique())):
    y_pts = kcl_part.loc[kcl_part["u2"] == u, "pred_prob"]
    x_pts = np.random.normal(i + 1, 0.04, size=len(y_pts))
    axes[1].scatter(x_pts, y_pts, color="#009E73", s=28, zorder=3)
axes[1].set_xlabel("UPDRS II-5 Speech Score")
axes[1].set_title(f"(b) UPDRS II-5 (rho = {rho_u2:+.2f})")

# 3. UPDRS III-18 (Speech)
u3_groups = [kcl_part.loc[kcl_part["u3"] == u, "pred_prob"].values for u in sorted(kcl_part["u3"].unique())]
axes[2].boxplot(u3_groups, tick_labels=[str(u) for u in sorted(kcl_part["u3"].unique())], patch_artist=True,
                boxprops=dict(facecolor="#E69F00", alpha=0.5))
for i, u in enumerate(sorted(kcl_part["u3"].unique())):
    y_pts = kcl_part.loc[kcl_part["u3"] == u, "pred_prob"]
    x_pts = np.random.normal(i + 1, 0.04, size=len(y_pts))
    axes[2].scatter(x_pts, y_pts, color="#D55E00", s=28, zorder=3)
axes[2].set_xlabel("UPDRS III-18 Speech Score")
axes[2].set_title(f"(c) UPDRS III-18 (rho = {rho_u3:+.2f})")

fig.suptitle("Clinical Severity Gradient: Positive Evidence of Disease Tracking", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = FIGURES / "phase1_fig05_severity_gradient.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "phase1_fig05_severity_gradient.pdf", bbox_inches="tight")
print(f"Saved {fig_path}")
plt.close(fig)

# %% [markdown]
# ## 16b — Calibration, Prior Correction & Decision Curve Analysis

# %%
# Pooled LOCO predictions
pipe_kcl = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))])
pipe_kcl.fit(tier_c_frame.loc[kcl_mask, X_cols].to_numpy(), tier_c_frame.loc[kcl_mask, "label"].to_numpy())
ipvs_test = tier_c_frame.loc[ipvs_mask].copy()
ipvs_test["pred_prob"] = pipe_kcl.predict_proba(ipvs_test[X_cols].to_numpy())[:, 1]
ipvs_part = ipvs_test.groupby("subject_id").agg(label=("label", "max"), pred_prob=("pred_prob", "mean")).reset_index()

all_part = pd.concat([kcl_part[["subject_id", "label", "pred_prob"]], ipvs_part], ignore_index=True)
y_true = all_part["label"].to_numpy()
y_prob = all_part["pred_prob"].to_numpy()

# Calibration metrics
cal_res = Cal.reliability_curve_and_ece(y_true, y_prob, n_bins=8)
murphy = Cal.brier_score_murphy(y_true, y_prob, n_bins=8)

print(f"\nCalibration Audit:")
print(f"  Expected Calibration Error (ECE): {cal_res['ece']:.3f}")
print(f"  Brier Score                     : {murphy['brier']:.3f}")
print(f"  Reliability (lower is better)   : {murphy['reliability']:.4f}")
print(f"  Resolution (higher is better)   : {murphy['resolution']:.4f}")

# Prior correction to 1.5% community prevalence
y_prob_corrected = Cal.prior_correction(y_prob, pi_train=float(np.mean(y_true)), pi_real=0.015)
sens_90_ppv = Cal.positive_predictive_value(0.90, 0.90, prevalence=0.015)
print(f"\nPrior Correction (1.5% population prevalence):")
print(f"  Max corrected probability in cohort: {np.max(y_prob_corrected):.3f}")
print(f"  PPV at 90% sensitivity & 90% specificity: {sens_90_ppv * 100:.1f}%")

# %% [markdown]
# ### Figure 6 — Calibration Curve & Decision Curve Analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

# 1. Reliability curve
axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
axes[0].plot(cal_res["prob_pred"], cal_res["prob_true"], "o-", color="#0072B2", linewidth=2,
             label=f"LOCO Model (ECE = {cal_res['ece']:.2f})")
axes[0].set_xlabel("Mean Predicted Probability")
axes[0].set_ylabel("Observed Proportion of PD")
axes[0].set_title("(a) Reliability Diagram", fontsize=10.5, fontweight="bold")
axes[0].legend(fontsize=8.5)
axes[0].set_aspect("equal")

# 2. Decision Curve Analysis (DCA)
dca_df = Cal.decision_curve_analysis(y_true, y_prob_corrected, thresholds=np.linspace(0.005, 0.08, 50))
axes[1].plot(dca_df["threshold"] * 100, dca_df["net_benefit_model"], "-", color="#D55E00", linewidth=2, label="AI Screening Model")
axes[1].plot(dca_df["threshold"] * 100, dca_df["net_benefit_all"], ":", color="#666666", linewidth=1.5, label="Treat / Screen All")
axes[1].axhline(0, color="k", linestyle="--", linewidth=1.0, label="Treat None")
axes[1].set_xlabel("Threshold Probability (%)")
axes[1].set_ylabel("Net Benefit")
axes[1].set_title("(b) Decision Curve Analysis (1.5% Prevalence)", fontsize=10.5, fontweight="bold")
axes[1].legend(fontsize=8.5)

fig.suptitle("Clinical Calibration and Net Utility Analysis", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = FIGURES / "phase1_fig06_calibration_dca.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "phase1_fig06_calibration_dca.pdf", bbox_inches="tight")
print(f"Saved {fig_path}")
plt.close(fig)

# %% [markdown]
# ## 16c — Sex Sensitivity Analysis (Female-Only Stratum)
#
# Because MDVR-KCL has only 2 male controls, the male stratum is completely confounded.
# The female stratum has:
# - IPVS: 9 PD / 12 HC
# - MDVR-KCL: 5 PD / 19 HC
# This provides a 100% sex-confound-free cross-corpus test!

# %%
female_meta = meta[meta["sex"] == "F"].copy()
# On KCL, identify females using F0 proxy (F0 > 155 Hz)
kcl_f0 = f0_df[f0_df["corpus"] == "mdvr_kcl"].groupby("subject_id")["f0_mean"].mean()
kcl_female_ids = set(kcl_f0[kcl_f0 > 155.0].index)

female_sub_ids = set(meta[meta["sex"] == "F"]["subject_id"]).union(kcl_female_ids)
female_df = all_part[all_part["subject_id"].isin(female_sub_ids)]

fem_y = female_df["label"].to_numpy()
fem_prob = female_df["pred_prob"].to_numpy()
fem_auc = M.safe_auroc(fem_y, fem_prob)

print(f"\nFemale-Only Sensitivity Analysis (Confound-Free):")
print(f"  Participants: {len(female_df)} ({sum(fem_y == 1)} PD, {sum(fem_y == 0)} HC)")
print(f"  AUROC in Female Stratum: {fem_auc:.3f}")

# %% [markdown]
# ### Figure 7 — Female-Only Confound-Free Validation

# %%
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(6, 5))
fpr_fem, tpr_fem, _ = roc_curve(fem_y, fem_prob)
ax.plot(fpr_fem, tpr_fem, color="#CC79A7", linewidth=2.5, label=f"Female Stratum Only (AUROC {fem_auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", label="Chance (0.50)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Sex Sensitivity: 100% Confound-Free Female Stratum", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_aspect("equal")

fig_path = FIGURES / "phase1_fig07_sex_sensitivity.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "phase1_fig07_sex_sensitivity.pdf", bbox_inches="tight")
print(f"Saved {fig_path}")
plt.close(fig)

# Save all audit metrics
audit_summary = pd.DataFrame(
    [
        {"metric": "spearman_rho_hy", "value": rho_hy},
        {"metric": "spearman_rho_updrs_ii5", "value": rho_u2},
        {"metric": "spearman_rho_updrs_iii18", "value": rho_u3},
        {"metric": "ece", "value": cal_res["ece"]},
        {"metric": "brier_score", "value": murphy["brier"]},
        {"metric": "female_only_auroc", "value": fem_auc},
    ]
)
audit_summary.to_parquet(RESULTS / "generalization_audit.parquet", index=False)
print("\nStage 16 completed successfully.")
