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
# # 13 — Channel Invariance: Enforcing Gate A
#
# Validate that Cepstral Mean & Variance Normalization (CMVN) and channel normalization
# neutralize the stationary acquisition confound on IPVS.
#
# **Success Criterion (Gate A):** Silence-only AUROC on IPVS drops from 1.000 to <= 0.65
# under a participant-disjoint cross-validation split (arm C).
#
# **Output:** `figures/phase1/phase1_fig03_gate_a_invariance.png`, `results/phase1/gate_a_verification.parquet`.

# %%
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import librosa

ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src import metrics as M  # noqa: E402
from src import normalize as N  # noqa: E402
from src import splits as S  # noqa: E402
from src import viz as V  # noqa: E402

V.use_house_style()

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "phase1"
FIGURES = ROOT / "figures" / "phase1"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

meta = pd.read_parquet(PROCESSED / "metadata_harmonized.parquet")
ipvs_meta = meta[meta["corpus"] == "ipvs"].reset_index(drop=True)
print(f"Loaded {len(ipvs_meta)} IPVS read passage recordings from {ipvs_meta['subject_id'].nunique()} participants")

# %% [markdown]
# ## Extract Raw Silence vs Normalized Silence Features

# %%
print("Extracting raw silence and CMVN-normalized silence features on IPVS...")

raw_silence_rows = []
cmvn_silence_rows = []
valid_indices = []

for idx, path in enumerate(ipvs_meta["path"]):
    signal, _ = librosa.load(path, sr=16000, mono=True)
    intervals = librosa.effects.split(signal, top_db=30)
    mask = np.ones(len(signal), dtype=bool)
    for start, end in intervals:
        mask[start:end] = False
    silence = signal[mask]

    if len(silence) < 16000 // 4:  # At least 250ms
        continue

    # 1. Raw MFCC on silence
    mfcc_raw = librosa.feature.mfcc(y=silence, sr=16000, n_mfcc=20)
    raw_feats = {}
    for i in range(20):
        raw_feats[f"sil_mfcc{i:02d}_mean"] = float(np.mean(mfcc_raw[i]))
        raw_feats[f"sil_mfcc{i:02d}_std"] = float(np.std(mfcc_raw[i]))
    raw_silence_rows.append(raw_feats)

    # 2. CMVN-normalized features on silence (Delta & Delta-Delta)
    cmvn_feats = N.extract_cmvn_mfcc(silence, sr=16000, n_mfcc=20)
    cmvn_silence_rows.append(cmvn_feats)
    valid_indices.append(idx)

sub_meta = ipvs_meta.iloc[valid_indices].reset_index(drop=True)
X_raw = pd.DataFrame(raw_silence_rows)
X_cmvn = pd.DataFrame(cmvn_silence_rows)
y = sub_meta["label"].to_numpy()

print(f"Evaluated {len(sub_meta)} recordings with valid non-speech regions")

# %% [markdown]
# ## Cross-validate Silence AUROC under Arm C (Participant-Disjoint)

# %%
res_raw = E.run_arm(X_raw, y, sub_meta, "C_subject_key", n_splits=5, n_repeats=5, seed=42)
res_cmvn = E.run_arm(X_cmvn, y, sub_meta, "C_subject_key", n_splits=5, n_repeats=5, seed=42)

raw_auc, raw_lo, raw_hi = res_raw["auroc_participant"]
cmvn_auc, cmvn_lo, cmvn_hi = res_cmvn["auroc_participant"]

print("\nGate A Verification (Participant-Level Silence AUROC):")
print(f"  Raw Silence MFCC      : {raw_auc:.3f} [{raw_lo:.3f}, {raw_hi:.3f}] (Severe confound)")
print(f"  CMVN Normalized Silence: {cmvn_auc:.3f} [{cmvn_lo:.3f}, {cmvn_hi:.3f}]")

gate_a_passed = cmvn_auc <= 0.65
print(f"\nGate A Status: {'PASSED' if gate_a_passed else 'FAILED'} (AUROC <= 0.65 threshold)")

# Save verification results
verif_df = pd.DataFrame(
    [
        {"condition": "Raw silence MFCC", "auroc": raw_auc, "ci_lo": raw_lo, "ci_hi": raw_hi, "status": "Confounded"},
        {"condition": "CMVN normalized silence", "auroc": cmvn_auc, "ci_lo": cmvn_lo, "ci_hi": cmvn_hi, "status": "Passed" if gate_a_passed else "Failed"},
    ]
)
verif_df.to_parquet(RESULTS / "gate_a_verification.parquet", index=False)

# %% [markdown]
# ## Figure 3 — Gate A Verification Plot

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))

labels = ["Raw Silence MFCC\n(Phase 0 baseline)", "CMVN Normalized Silence\n(Phase 1 invariant)"]
aucs = [raw_auc, cmvn_auc]
err_lo = [raw_auc - raw_lo, cmvn_auc - cmvn_lo]
err_hi = [raw_hi - raw_auc, cmvn_hi - cmvn_auc]
colors = ["#D55E00", "#009E73" if gate_a_passed else "#E69F00"]

bars = ax.bar(labels, aucs, yerr=[err_lo, err_hi], color=colors, width=0.45, capsize=4, edgecolor="white")
ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10, fontweight="bold")

ax.axhline(0.5, color="black", linestyle="--", linewidth=1.1, label="Chance level (0.50)")
ax.axhline(0.65, color="#D55E00", linestyle=":", linewidth=1.3, label="Gate A Ceiling (0.65)")

ax.set_ylabel("Participant-level AUROC (95% CI)")
ax.set_ylim(0.4, 1.08)
ax.set_title("Gate A: Neutralizing the Channel Shortcut via CMVN", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")

fig_path = FIGURES / "phase1_fig03_gate_a_invariance.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "phase1_fig03_gate_a_invariance.pdf", bbox_inches="tight")
print(f"Saved {fig_path}")
plt.close(fig)

print("\nStage 13 completed successfully.")
