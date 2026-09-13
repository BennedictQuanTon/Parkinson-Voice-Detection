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
# # 03 — Feature extraction
#
# Two feature sets on two tasks, cached to parquet so the modelling notebook is
# cheap to re-run.
#
# | feature set | size | why |
# |---|---|---|
# | `mfcc` | 40 (20 means + 20 SDs) | closest match to the representation used by the reference study |
# | `egemaps` | 88 | eGeMAPSv02, the standard interpretable paralinguistic baseline |
#
# | task | recordings | why |
# |---|---|---|
# | `read_text` | ~94 | where the reference result sits, and where the repeated-reading structure lives |
# | `vowel_a` | ~99 | sustained phonation, the only task where perturbation measures are defined |
#
# A third set, `silence`, is extracted from the **non-speech** regions of the read
# passage. It carries no voice, so any discriminative power it has comes from the
# recording channel. It is the control that makes the batch confound measurable.
#
# All audio is resampled to 16 kHz on load, so the 44.1 kHz files are not
# trivially separable by rate — the question is whether the channel difference
# survives resampling.

# %%
import sys
import time
from pathlib import Path

import pandas as pd

# Walk up to the repository root so this runs identically from notebooks/,
# scripts/, or the repository root itself.
ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = PROCESSED / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

meta = pd.read_parquet(PROCESSED / "metadata.parquet")
cohort = meta[meta["cohort_full"]].reset_index(drop=True)

TASKS = ["read_text", "vowel_a"]
print(f"cohort: {len(cohort)} recordings, {cohort['subject_id'].nunique()} participants")
for task in TASKS:
    subset = cohort[cohort["task"] == task]
    print(
        f"  {task:10s} {len(subset):3d} recordings, "
        f"{subset['subject_id'].nunique()} participants"
    )

# %% [markdown]
# ## Extract
#
# The index of every features frame is aligned to the row index of the metadata
# subset it came from, so features can never be silently paired with the wrong
# recording. That alignment is asserted, not assumed.

# %%
FEATURE_SETS = {
    "read_text": ["mfcc", "egemaps", "silence", "channel"],
    "vowel_a": ["mfcc", "egemaps", "channel"],
}

manifest = []
for task, kinds in FEATURE_SETS.items():
    subset = cohort[cohort["task"] == task].reset_index(drop=True)
    for kind in kinds:
        out_path = FEATURE_DIR / f"{task}__{kind}.parquet"
        if out_path.exists():
            frame = pd.read_parquet(out_path)
            print(f"{task}/{kind}: cached, {frame.shape}")
        else:
            print(f"{task}/{kind}: extracting from {len(subset)} recordings...")
            start = time.time()
            frame = F.extract(subset, kind)
            # Carry the keys needed downstream; features alone are not enough.
            keys = subset.loc[
                frame.index,
                ["subject_id", "group", "label", "task", "task_rep", "session_index",
                 "folder_path", "age", "sex", "samplerate", "batch", "duration",
                 "cohort_batch_matched"],
            ]
            frame = pd.concat([keys, frame], axis=1)
            frame.to_parquet(out_path, index=False)
            print(f"  {frame.shape} in {time.time() - start:.1f}s -> {out_path.name}")

        n_feat = frame.shape[1] - 13
        manifest.append(
            {
                "task": task,
                "feature_set": kind,
                "recordings": len(frame),
                "features": n_feat,
                "participants": frame["subject_id"].nunique(),
                "file": out_path.name,
            }
        )

manifest = pd.DataFrame(manifest)
print()
print(manifest.to_string(index=False))

# %% [markdown]
# ## Sanity checks
#
# Cheap, and each one catches a class of bug that would otherwise show up as a
# suspiciously good result.

# %%
problems = []
for row in manifest.itertuples():
    frame = pd.read_parquet(FEATURE_DIR / row.file)
    feature_cols = frame.columns[13:]
    numeric = frame[feature_cols]

    if numeric.isna().any().any():
        problems.append(f"{row.file}: contains NaN")
    if (numeric.nunique() <= 1).any():
        constant = numeric.columns[numeric.nunique() <= 1].tolist()
        problems.append(f"{row.file}: constant columns {constant}")
    if frame[["subject_id", "task", "task_rep", "session_index"]].duplicated().any():
        problems.append(f"{row.file}: duplicate (participant, task, rep, session) rows")
    if frame["label"].nunique() != 2:
        problems.append(f"{row.file}: only one class present")
    if frame.groupby("subject_id")["label"].nunique().gt(1).any():
        problems.append(f"{row.file}: a participant carries both labels")

if problems:
    for problem in problems:
        print("PROBLEM:", problem)
    raise AssertionError(f"{len(problems)} feature-table problem(s); see above")
print("all feature tables pass: no NaN, no constant column, no duplicate row,")
print("both classes present, one label per participant")

# %% [markdown]
# ## Is any single feature suspiciously good on its own?
#
# A feature that separates the groups almost perfectly by itself is more likely to
# be measuring the recording setup than the disease. Worth knowing before it turns
# up inside a model.

# %%
from sklearn.metrics import roc_auc_score  # noqa: E402

rows = []
for row in manifest.itertuples():
    frame = pd.read_parquet(FEATURE_DIR / row.file)
    for cohort_name, subset in (
        ("full", frame),
        ("batch_matched", frame[frame["cohort_batch_matched"]]),
    ):
        y = subset["label"].to_numpy()
        for column in subset.columns[13:]:
            auc = roc_auc_score(y, subset[column])
            rows.append(
                {
                    "cohort": cohort_name,
                    "task": row.task,
                    "feature_set": row.feature_set,
                    "feature": column,
                    # Direction-free: 0.02 is as informative as 0.98.
                    "auroc": max(auc, 1 - auc),
                }
            )

univariate = pd.DataFrame(rows).sort_values("auroc", ascending=False)
univariate.to_parquet(PROCESSED / "univariate_auroc.parquet", index=False)

print("Strongest single feature per table, both cohorts:")
best = (
    univariate.sort_values("auroc", ascending=False)
    .groupby(["cohort", "task", "feature_set"])
    .head(1)
    .pivot_table(index=["task", "feature_set"], columns="cohort",
                 values="auroc", aggfunc="max")
    .round(3)
)
print(best.to_string())

# %% [markdown]
# Two things to read off that table.
#
# **The `silence` row reaches 1.000.** Silence contains no speech. A single MFCC
# coefficient of the non-speech portion of the recording separates the groups
# perfectly, which means the groups differ in their recording conditions, not
# only in their voices.
#
# **Restricting to the batch-matched cohort does not help.** Within the 2017
# campaign, at a single sample rate, silence still separates the groups perfectly.
# So the confound is not the sample rate; it is a deeper acquisition difference
# that the sample rate merely hinted at.

# %%
n_high = (
    univariate[univariate["auroc"] > 0.9]
    .groupby(["cohort", "task", "feature_set"])
    .size()
    .rename("features_above_0.90")
    .reset_index()
    .pivot_table(index=["task", "feature_set"], columns="cohort",
                 values="features_above_0.90", fill_value=0)
)
print("Number of individual features exceeding AUROC 0.90 on their own:")
print(n_high.to_string())

print("\nChannel statistics, which contain no phonetic information at all:")
print(
    univariate[univariate["feature_set"] == "channel"]
    .pivot_table(index="feature", columns=["task", "cohort"], values="auroc")
    .round(3)
    .to_string()
)

# %% [markdown]
# The sustained vowel is markedly less contaminated than the read passage, which
# is consistent with the passage recordings containing long stretches of
# background over which the channel signature accumulates. It is still not clean.

# %%
manifest.to_parquet(PROCESSED / "feature_manifest.parquet", index=False)
print("wrote data/processed/feature_manifest.parquet")
