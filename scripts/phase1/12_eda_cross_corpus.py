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
# # 12 — Per-corpus confound audit and domain shift analysis
#
# Characterize the acoustic environment and confounds of each corpus independently:
# 1. Silence-only and channel-statistics controls on MDVR-KCL vs IPVS.
# 2. F0-only baseline as a proxy for the severe sex confound on MDVR-KCL.
# 3. Acoustic domain shift (bandwidth, reverberation, noise floor, levels).
#
# **Output:** `figures/phase1/phase1_fig01_domain_shift.png`, `figures/phase1/phase1_fig02_per_corpus_audit.png`, `results/phase1/per_corpus_audit.parquet`.

# %%
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402
from src import metrics as M  # noqa: E402
from src import viz as V  # noqa: E402

V.use_house_style()

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "phase1"
FIGURES = ROOT / "figures" / "phase1"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

meta = pd.read_parquet(PROCESSED / "metadata_harmonized.parquet")
print(f"Loaded {len(meta)} recordings from metadata_harmonized.parquet")

# %% [markdown]
# ## Extract channel, silence, and F0 features for both corpora

# %%
print("Extracting channel features...")
ch_features = F.extract(meta, "channel", verbose=False)
ch_df = pd.concat([meta.loc[ch_features.index, ["corpus", "subject_id", "group", "label"]], ch_features], axis=1)

print("Extracting silence features...")
sil_features = F.extract(meta, "silence", verbose=False)
sil_df = pd.concat([meta.loc[sil_features.index, ["corpus", "subject_id", "group", "label"]], sil_features], axis=1)

print("Extracting F0 sex proxy features...")
f0_features = F.extract(meta, "f0_sex_proxy", verbose=False)
f0_df = pd.concat([meta.loc[f0_features.index, ["corpus", "subject_id", "group", "label"]], f0_features], axis=1)

# %% [markdown]
# ## Evaluate per-corpus controls (Participant-Level AUROC)

# %%
audit_rows = []

for corpus in ("ipvs", "mdvr_kcl"):
    # 1. Silence control (best single sil_mfcc or silence_seconds)
    s_sub = sil_df[sil_df["corpus"] == corpus]
    sil_cols = [c for c in sil_features.columns if c in s_sub.columns]
    best_sil_auc = 0.5
    for c in sil_cols:
        part = s_sub.groupby("subject_id").agg(label=("label", "max"), score=(c, "mean"))
        if part["label"].nunique() == 2:
            auc = M.safe_auroc(part["label"].to_numpy(), part["score"].to_numpy())
            auc = max(auc, 1.0 - auc)
            if auc > best_sil_auc:
                best_sil_auc = auc

    # 2. Channel control (best single channel feature)
    c_sub = ch_df[ch_df["corpus"] == corpus]
    ch_cols = [c for c in ch_features.columns if c in c_sub.columns]
    best_ch_auc = 0.5
    for c in ch_cols:
        part = c_sub.groupby("subject_id").agg(label=("label", "max"), score=(c, "mean"))
        if part["label"].nunique() == 2:
            auc = M.safe_auroc(part["label"].to_numpy(), part["score"].to_numpy())
            auc = max(auc, 1.0 - auc)
            if auc > best_ch_auc:
                best_ch_auc = auc

    # 3. F0 sex-proxy baseline
    f_sub = f0_df[f0_df["corpus"] == corpus]
    part_f0 = f_sub.groupby("subject_id").agg(label=("label", "max"), score=("f0_mean", "mean"))
    f0_auc = 0.5
    if part_f0["label"].nunique() == 2:
        auc = M.safe_auroc(part_f0["label"].to_numpy(), part_f0["score"].to_numpy())
        f0_auc = max(auc, 1.0 - auc)

    audit_rows.append(
        {
            "corpus": corpus,
            "silence_best_auroc": best_sil_auc,
            "channel_best_auroc": best_ch_auc,
            "f0_sex_proxy_auroc": f0_auc,
            "n_participants": meta[meta["corpus"] == corpus]["subject_id"].nunique(),
        }
    )

audit_df = pd.DataFrame(audit_rows)
print("\nPer-Corpus Audit Results (Participant-Level AUROC):")
print(audit_df.round(3).to_string(index=False))
audit_df.to_parquet(RESULTS / "per_corpus_audit.parquet", index=False)

# %% [markdown]
# ## Figure 1 — Acoustic Domain Shift between IPVS and MDVR-KCL

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

# 1. Noise floor comparison (ch_silence_db)
for corpus, col in (("ipvs", "#0072B2"), ("mdvr_kcl", "#D55E00")):
    vals = ch_df[ch_df["corpus"] == corpus]["ch_silence_db"].dropna()
    axes[0].hist(vals, bins=15, alpha=0.6, color=col, label=corpus.upper(), edgecolor="white")
axes[0].set_xlabel("Silence RMS (dB)")
axes[0].set_ylabel("Recordings")
axes[0].set_title("(a) Noise floor shift")
axes[0].legend(fontsize=8)

# 2. SNR comparison (ch_snr_db)
for corpus, col in (("ipvs", "#0072B2"), ("mdvr_kcl", "#D55E00")):
    vals = ch_df[ch_df["corpus"] == corpus]["ch_snr_db"].dropna()
    axes[1].hist(vals, bins=15, alpha=0.6, color=col, label=corpus.upper(), edgecolor="white")
axes[1].set_xlabel("Signal-to-Noise Ratio (dB)")
axes[1].set_title("(b) Dynamic range / SNR")
axes[1].legend(fontsize=8)

# 3. Spectral Tilt / Low Frequency Ratio (< 625 Hz)
for corpus, col in (("ipvs", "#0072B2"), ("mdvr_kcl", "#D55E00")):
    vals = ch_df[ch_df["corpus"] == corpus]["ch_lf_ratio"].dropna()
    axes[2].hist(vals, bins=15, alpha=0.6, color=col, label=corpus.upper(), edgecolor="white")
axes[2].set_xlabel("LF energy ratio (< 625 Hz)")
axes[2].set_title("(c) Spectral tilt difference")
axes[2].legend(fontsize=8)

# 4. F0 Distribution (Sex proxy)
for corpus, col in (("ipvs", "#0072B2"), ("mdvr_kcl", "#D55E00")):
    vals = f0_df[f0_df["corpus"] == corpus]["f0_mean"].dropna()
    axes[3].hist(vals, bins=15, alpha=0.6, color=col, label=corpus.upper(), edgecolor="white")
axes[3].set_xlabel("Mean F0 (Hz)")
axes[3].set_title("(d) F0 distribution (Sex proxy)")
axes[3].legend(fontsize=8)

fig.suptitle("Acoustic Domain Shift: IPVS vs MDVR-KCL", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
fig_path = FIGURES / "phase1_fig01_domain_shift.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "phase1_fig01_domain_shift.pdf", bbox_inches="tight")
print(f"Saved {fig_path}")
plt.close(fig)

# %% [markdown]
# ## Figure 2 — Per-Corpus Confound Profile

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(audit_df))
width = 0.25

bars1 = ax.bar(x - width, audit_df["silence_best_auroc"], width, label="Silence control", color="#D55E00")
bars2 = ax.bar(x, audit_df["channel_best_auroc"], width, label="Channel stats control", color="#E69F00")
bars3 = ax.bar(x + width, audit_df["f0_sex_proxy_auroc"], width, label="F0 sex-proxy baseline", color="#0072B2")

ax.bar_label(bars1, fmt="%.2f", fontsize=8.5, padding=2)
ax.bar_label(bars2, fmt="%.2f", fontsize=8.5, padding=2)
ax.bar_label(bars3, fmt="%.2f", fontsize=8.5, padding=2)

ax.axhline(0.5, color="black", linestyle="--", linewidth=1.1, label="Chance level (0.50)")
ax.set_xticks(x)
ax.set_xticklabels([c.upper() for c in audit_df["corpus"]], fontsize=10, fontweight="bold")
ax.set_ylabel("AUROC")
ax.set_ylim(0.4, 1.08)
ax.set_title("Per-Corpus Confound Profiles: The Two Opposite Hazards", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")

fig_path = FIGURES / "phase1_fig02_per_corpus_audit.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "phase1_fig02_per_corpus_audit.pdf", bbox_inches="tight")
print(f"Saved {fig_path}")
plt.close(fig)

print("\nStage 12 completed successfully.")
