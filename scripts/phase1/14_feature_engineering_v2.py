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
# # 14 — Feature Extraction: Four Multi-Tier Sets
#
# Extract channel-robust, clinically grounded features across both corpora:
# - **Tier A:** Physiological & perturbation measures via Praat/Parselmouth (jitter, shimmer, HNR, F0, pause statistics).
# - **Tier B:** CMVN-MFCC (40 delta & delta-delta features, mathematically invariant to channel tilt).
# - **Tier C:** eGeMAPSv02 (88 standard paralinguistic features) with absolute energy parameters flagged.
# - **Tier AB:** Combined physiological + dynamic acoustic representation.
#
# **Output:** `data/processed/features/harmonized__{tier}.parquet`.

# %%
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = PROCESSED / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

meta = pd.read_parquet(PROCESSED / "metadata_harmonized.parquet")
print(f"Extracting features for {len(meta)} harmonized recordings across {meta['subject_id'].nunique()} participants")

KEY_COLS = ["corpus", "subject_id", "group", "label", "task", "language", "age", "sex", "severity_hy", "severity_u2", "severity_u3", "path"]

# %% [markdown]
# ## Extract Tier A: Physiological, Perturbation & Pause Features

# %%
print("\nExtracting Tier A (Parselmouth jitter/shimmer/HNR, F0, pause rhythm)...")
start_t = time.time()
tier_a_df = F.extract(meta, "tier_a", verbose=True)
tier_a_combined = pd.concat([meta.loc[tier_a_df.index, KEY_COLS], tier_a_df], axis=1)
tier_a_path = FEATURE_DIR / "harmonized__tier_a.parquet"
tier_a_combined.to_parquet(tier_a_path, index=False)
print(f"Tier A saved: {tier_a_combined.shape} in {time.time() - start_t:.1f}s -> {tier_a_path.name}")

# %% [markdown]
# ## Extract Tier B: CMVN-MFCC (Channel Invariant)

# %%
print("\nExtracting Tier B (CMVN-MFCC delta & acceleration features)...")
start_t = time.time()
tier_b_df = F.extract(meta, "tier_b", verbose=True)
tier_b_combined = pd.concat([meta.loc[tier_b_df.index, KEY_COLS], tier_b_df], axis=1)
tier_b_path = FEATURE_DIR / "harmonized__tier_b.parquet"
tier_b_combined.to_parquet(tier_b_path, index=False)
print(f"Tier B saved: {tier_b_combined.shape} in {time.time() - start_t:.1f}s -> {tier_b_path.name}")

# %% [markdown]
# ## Extract Tier C: eGeMAPSv02

# %%
print("\nExtracting Tier C (openSMILE eGeMAPSv02 functionals)...")
start_t = time.time()
tier_c_df = F.extract(meta, "egemaps", verbose=True)
tier_c_combined = pd.concat([meta.loc[tier_c_df.index, KEY_COLS], tier_c_df], axis=1)
tier_c_path = FEATURE_DIR / "harmonized__tier_c.parquet"
tier_c_combined.to_parquet(tier_c_path, index=False)
print(f"Tier C saved: {tier_c_combined.shape} in {time.time() - start_t:.1f}s -> {tier_c_path.name}")

# %% [markdown]
# ## Combine Tier A + Tier B (The Primary Robust Feature Set)

# %%
common_idx = tier_a_df.index.intersection(tier_b_df.index)
tier_ab_feats = pd.concat([tier_a_df.loc[common_idx], tier_b_df.loc[common_idx]], axis=1)
tier_ab_combined = pd.concat([meta.loc[common_idx, KEY_COLS], tier_ab_feats], axis=1)
tier_ab_path = FEATURE_DIR / "harmonized__tier_ab.parquet"
tier_ab_combined.to_parquet(tier_ab_path, index=False)
print(f"\nTier AB combined: {tier_ab_combined.shape} -> {tier_ab_path.name}")

# %% [markdown]
# ## Sanity Checks on Extracted Tables

# %%
for name, path in (
    ("Tier A", tier_a_path),
    ("Tier B", tier_b_path),
    ("Tier C", tier_c_path),
    ("Tier AB", tier_ab_path),
):
    frame = pd.read_parquet(path)
    feats = frame.iloc[:, len(KEY_COLS):]
    assert not feats.isna().any().any(), f"{name} contains NaN values"
    assert (feats.nunique() > 1).all(), f"{name} contains constant columns"
    assert frame["label"].nunique() == 2, f"{name} does not have both classes"
    print(f"PASS: {name} (N={len(frame)} recordings, D={feats.shape[1]} features)")

print("\nStage 14 completed successfully.")
