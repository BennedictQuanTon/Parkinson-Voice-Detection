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
# # 11 — Harmonize corpora into a unified schema
#
# Combine IPVS and MDVR-KCL into a single, standardized schema restricted to
# the shared task: the phonetically balanced read passage.
#
# **Input:** `data/raw/ipvs/` (and `data/processed/metadata.parquet`), `data/raw/mdvr_kcl/`.
# **Output:** `data/processed/metadata_harmonized.parquet`.

# %%
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import corpora as C  # noqa: E402

IPVS_DIR = ROOT / "data" / "raw" / "ipvs"
KCL_DIR = ROOT / "data" / "raw" / "mdvr_kcl"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
IPVS_META = PROCESSED / "metadata.parquet"

print(f"IPVS dir : {IPVS_DIR}")
print(f"KCL dir  : {KCL_DIR}")

# %% [markdown]
# ## Load both corpora

# %%
ipvs_df = C.load_ipvs(IPVS_DIR, metadata_parquet=IPVS_META if IPVS_META.exists() else None)
kcl_df, unparsed_kcl = C.load_mdvr_kcl(KCL_DIR)

print(f"IPVS total records : {len(ipvs_df)} across {ipvs_df['subject_id'].nunique()} participants")
print(f"KCL total records  : {len(kcl_df)} across {kcl_df['subject_id'].nunique()} participants")
assert not unparsed_kcl, f"Unparsed KCL files: {unparsed_kcl}"

# %% [markdown]
# ## Filter to shared task: Read Passage

# %%
ipvs_read = ipvs_df[ipvs_df["task"] == "read_passage"].copy()
kcl_read = kcl_df[kcl_df["task"] == "read_passage"].copy()

print(f"\nRead passage recordings:")
print(f"  IPVS: {len(ipvs_read)} recordings ({ipvs_read['subject_id'].nunique()} participants)")
print(f"  KCL : {len(kcl_read)} recordings ({kcl_read['subject_id'].nunique()} participants)")

harmonized = pd.concat([ipvs_read, kcl_read], ignore_index=True)

# %% [markdown]
# ## Schema verification & missingness audit

# %%
print("\nHarmonized dataset overview:")
breakdown = harmonized.groupby(["corpus", "group"]).agg(
    participants=("subject_id", "nunique"),
    recordings=("path", "count"),
    has_age=("age", lambda s: s.notna().sum()),
    has_sex=("sex", lambda s: s.notna().sum()),
    has_hy=("severity_hy", lambda s: s.notna().sum()),
    has_updrs=("severity_u2", lambda s: s.notna().sum()),
)
print(breakdown.to_string())

# Verification assertions
assert harmonized["label"].nunique() == 2
assert set(harmonized["corpus"]) == {"ipvs", "mdvr_kcl"}
assert harmonized.groupby("subject_id")["label"].nunique().max() == 1, "Subject with conflicting labels"
assert harmonized.groupby("subject_id")["corpus"].nunique().max() == 1, "Subject in multiple corpora"

# %% [markdown]
# ## Save harmonized tables

# %%
out_path = PROCESSED / "metadata_harmonized.parquet"
harmonized.to_parquet(out_path, index=False)
print(f"\nSaved {out_path} ({len(harmonized)} rows, {len(harmonized.columns)} columns)")

public_path = PROCESSED / "metadata_harmonized_public.parquet"
public_cols = [c for c in harmonized.columns if c not in ("path", "folder_path")]
harmonized[public_cols].to_parquet(public_path, index=False)
print(f"Saved public redacted table to {public_path}")

print("\nStage 11 completed successfully.")
