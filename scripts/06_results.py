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
# # 06 — Results
#
# Assembles the final tables and figures, and places them next to the published
# result on the same corpus.
#
# ## What was asked
#
# **RQ1.** How much of the reported performance on this corpus is produced by
# splitting at recording level rather than at participant level?
#
# **RQ2.** How much is produced by acquisition conditions rather than by voice?
#
# ## What was found
#
# RQ2 dominates, and it changes what RQ1 can even be asked of.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Walk up to the repository root so this runs identically from notebooks/,
# scripts/, or the repository root itself.
ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import viz as V  # noqa: E402

V.use_house_style()
pd.set_option("display.width", 190)
pd.set_option("display.max_rows", 200)

RESULTS = ROOT / "results"
grid = pd.read_parquet(RESULTS / "grid.parquet")
effect = pd.read_parquet(RESULTS / "leakage_effect.parquet")
controls = pd.read_parquet(RESULTS / "negative_controls.parquet")
meta = pd.read_parquet(ROOT / "data" / "processed" / "metadata.parquet")

# %% [markdown]
# ## Table 1 — Cohort
#
# The count that matters is participants, not recordings. Published descriptions of
# this corpus quote "28 patients" from the directory count; three of those
# directories are second sessions of participants already present.

# %%
cohort = meta[meta["cohort_full"]]
table1 = (
    cohort.drop_duplicates("subject_id")
    .groupby("group")
    .agg(
        participants=("subject_id", "size"),
        age_mean=("age", "mean"),
        age_sd=("age", "std"),
        age_min=("age", "min"),
        age_max=("age", "max"),
        male=("sex", lambda s: int((s == "M").sum())),
    )
    .round(1)
)
table1["recordings"] = cohort.groupby("group").size()
table1["directories"] = cohort.groupby("group")["folder_path"].nunique()
print(table1.to_string())
print(
    f"\nPD: {table1.loc['PD', 'directories']} directories resolve to "
    f"{table1.loc['PD', 'participants']} participants "
    "(three were recorded in two sessions each)."
)
table1.to_csv(RESULTS / "table1_cohort.csv")

# %% [markdown]
# ## Table 2 — Main results
#
# Participant-level AUROC with a bootstrap interval resampled by participant, for
# every task, feature set, cohort and split scheme.

# %%
table2 = grid.copy()
table2["AUROC (95% CI)"] = table2.apply(
    lambda r: f"{r['auroc_participant']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
    axis=1,
)
pivot = table2.pivot_table(
    index=["task", "feature_set", "cohort"],
    columns="arm",
    values="AUROC (95% CI)",
    aggfunc="first",
)
print(pivot.to_string())
pivot.to_csv(RESULTS / "table2_main_results.csv")

# %% [markdown]
# ## Table 3 — Comparison with the published result on the same corpus
#
# Klempir, Krupicka & Krupil (2024), *Sensors* 24(17):5520, evaluate the same two
# free corpora with a random forest and MFCC-style features, reporting accuracy
# around 0.85–0.95 within the Italian corpus. The comparison below is deliberately
# not framed as "we beat them": our numbers are *higher*, and that is the problem.

# %%
best_ours = grid[
    (grid["arm"] == "C_subject_key")
    & (grid["task"] == "read_text")
    & (grid["feature_set"] == "mfcc")
]
ours_full = best_ours[best_ours["cohort"] == "full"].iloc[0]
ours_matched = best_ours[best_ours["cohort"] == "batch_matched"].iloc[0]
silence_matched = grid[
    (grid["arm"] == "C_subject_key")
    & (grid["task"] == "read_text")
    & (grid["feature_set"] == "silence")
    & (grid["cohort"] == "batch_matched")
].iloc[0]

table3 = pd.DataFrame(
    [
        {
            "study": "Klempir et al. 2024 (Sensors 24:5520)",
            "corpus": "IPVS (Italian)",
            "participants": "28 PD / 22 HC (as described)",
            "split": "5-fold CV; grouping not stated",
            "features": "MFCC + prosody",
            "model": "random forest",
            "reported": "accuracy 0.85-0.95",
            "negative controls": "none reported",
        },
        {
            "study": "This work, arm A (recording-level split)",
            "corpus": "IPVS, read passage",
            "participants": f"{int(ours_full['n_participants'])} total",
            "split": "recording-level, 5x5 CV",
            "features": "MFCC (40)",
            "model": "random forest (100)",
            "reported": f"AUROC {grid[(grid.arm=='A_recording') & (grid.task=='read_text') & (grid.feature_set=='mfcc') & (grid.cohort=='full')].iloc[0]['auroc_participant']:.3f}",
            "negative controls": "fail",
        },
        {
            "study": "This work, arm C (participant-level split)",
            "corpus": "IPVS, read passage",
            "participants": f"{int(ours_full['n_participants'])} total",
            "split": "participant-level, 5x5 CV",
            "features": "MFCC (40)",
            "model": "random forest (100)",
            "reported": f"AUROC {ours_full['auroc_participant']:.3f} [{ours_full['ci_lo']:.3f}, {ours_full['ci_hi']:.3f}]",
            "negative controls": "fail",
        },
        {
            "study": "This work, batch-matched cohort",
            "corpus": "IPVS 2017 only, 16 kHz",
            "participants": f"{int(ours_matched['n_participants'])} total",
            "split": "participant-level, 5x5 CV",
            "features": "MFCC (40)",
            "model": "random forest (100)",
            "reported": f"AUROC {ours_matched['auroc_participant']:.3f} [{ours_matched['ci_lo']:.3f}, {ours_matched['ci_hi']:.3f}]",
            "negative controls": "fail",
        },
        {
            "study": "This work, SILENCE ONLY (control)",
            "corpus": "IPVS 2017 only, 16 kHz",
            "participants": f"{int(silence_matched['n_participants'])} total",
            "split": "participant-level, 5x5 CV",
            "features": "MFCC of non-speech regions",
            "model": "random forest (100)",
            "reported": f"AUROC {silence_matched['auroc_participant']:.3f} [{silence_matched['ci_lo']:.3f}, {silence_matched['ci_hi']:.3f}]",
            "negative controls": "this IS the control",
        },
    ]
)
print(table3.to_string(index=False))
table3.to_csv(RESULTS / "table3_comparison.csv", index=False)

# %% [markdown]
# The final row is the whole result. A model shown only the parts of the recording
# where nobody is speaking reproduces the performance of every speech-based model
# in the table, under a correct participant-level split, on the cohort where the
# sample rate is constant.
#
# Our arm-C number, 0.99, is *higher* than the published accuracy. Under the
# ordinary reading of a leaderboard that would be an improvement. Given the
# silence control, it is instead a measure of how strong the confound is.

# %% [markdown]
# ## Figure 12 — Summary
#
# One figure for the write-up: what the model scores, and what four things that
# should not work score alongside it.

# %%
fig = plt.figure(figsize=(13.5, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.32)
ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# Left: speech vs controls, correct split only.
arm_c = grid[(grid["arm"] == "C_subject_key")]
order = ["silence", "channel", "mfcc", "egemaps"]
labels = {"silence": "silence only", "channel": "channel stats",
          "mfcc": "MFCC", "egemaps": "eGeMAPSv02"}
positions, heights, colours, ticklabels = [], [], [], []
pos = 0
for task in ("read_text", "vowel_a"):
    for feature in order:
        row = arm_c[
            (arm_c["task"] == task)
            & (arm_c["feature_set"] == feature)
            & (arm_c["cohort"] == "batch_matched")
        ]
        if not len(row):
            continue
        row = row.iloc[0]
        positions.append(pos)
        heights.append(row["auroc_participant"])
        colours.append("#D55E00" if feature in ("silence", "channel") else "#0072B2")
        ticklabels.append(f"{labels[feature]}")
        pos += 1
    pos += 0.8

bars = ax0.bar(positions, heights, 0.72, color=colours)
ax0.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=2)
ax0.axhline(0.5, color="black", linestyle="--", linewidth=1.1)
ax0.set_xticks(positions)
ax0.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=8.5)
ax0.set_ylabel("Participant-level AUROC")
ax0.set_ylim(0.4, 1.08)
ax0.set_title("Batch-matched cohort, correct split", fontsize=10.5)
n_read = len(order)
ax0.text(1.5, 1.055, "read passage", ha="center", fontsize=9, style="italic")
ax0.text(6.3, 1.055, "sustained vowel", ha="center", fontsize=9, style="italic")
handles = [
    plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
    plt.Rectangle((0, 0), 1, 1, color="#D55E00"),
]
ax0.legend(handles, ["speech features", "controls with no speech"], fontsize=8.5,
           loc="lower left")

# Right: the four controls.
control_order = [
    ("silence only, no speech (batch_matched)", "Silence only\n(no speech)"),
    ("label permutation null mean (A_recording)", "Shuffled labels,\nrecording-level split"),
    ("age + sex only (batch_matched)", "Age + sex only"),
    ("label permutation null mean (C_subject_key)", "Shuffled labels,\ncorrect split"),
]
rows = []
for key, label in control_order:
    row = controls[controls["control"] == key]
    if len(row):
        row = row.iloc[0]
        rows.append({"label": label, "auroc": row["auroc"], "lo": row["ci_lo"],
                     "hi": row["ci_hi"]})
cdf = pd.DataFrame(rows)[::-1].reset_index(drop=True)
ypos = np.arange(len(cdf))
ax1.barh(
    ypos, cdf["auroc"],
    xerr=[cdf["auroc"] - cdf["lo"], cdf["hi"] - cdf["auroc"]],
    color=["#009E73" if v < 0.7 else "#D55E00" for v in cdf["auroc"]],
    height=0.6, capsize=4, error_kw={"linewidth": 1.1, "ecolor": "#333333"},
)
label_x = cdf["hi"].max() + 0.04
for y, value in zip(ypos, cdf["auroc"]):
    ax1.text(label_x, y, f"{value:.3f}", va="center", fontsize=9.5)
ax1.axvline(0.5, color="black", linestyle="--", linewidth=1.1)
ax1.set_yticks(ypos)
ax1.set_yticklabels(cdf["label"], fontsize=9)
ax1.set_xlim(0, 1.2)
ax1.set_xlabel("Participant-level AUROC (95% CI)")
ax1.set_title("What should not work, but does", fontsize=10.5)
ax1.grid(axis="y", visible=False)

fig.suptitle(
    "IPVS separates patients from controls by recording condition, not by voice",
    fontsize=13, fontweight="bold", y=1.0,
)
V.save_fig(
    fig, "fig12_summary",
    "Summary. Under a correct participant-level split on a single-sample-rate "
    "cohort, a model given only non-speech audio matches models given speech, and "
    "a recording-level split reports AUROC 0.88 for randomly assigned labels.",
)
plt.show()

# %% [markdown]
# ## Answers
#
# **RQ2 — acquisition.** On the read passage, a random forest given only the
# non-speech regions reaches participant-level AUROC 1.000 (95% CI 1.000–1.000)
# under a participant-disjoint split, on the cohort restricted to a single campaign
# and a single sample rate. Eight gross channel statistics containing no phonetic
# information reach 0.897. Patients and controls in this corpus were recorded under
# systematically different conditions, and that difference is sufficient to
# separate them perfectly. No performance figure computed on this corpus can be
# attributed to voice.
#
# **RQ1 — splitting.** With the real labels, the split scheme makes almost no
# difference, because every arm is already at the ceiling the confound provides
# (Δ AUROC = 0.000 for MFCC and eGeMAPS). The cost of a recording-level split is
# visible only once the real signal is removed: under participant-level label
# shuffling, a recording-level split reports AUROC 0.88 while a participant-level
# split reports 0.45. So a recording-level split on this corpus can manufacture
# 0.88 from noise, and the reason our main comparison shows no difference is a
# ceiling effect, not harmlessness.
#
# ## What may and may not be claimed
#
# May be claimed:
#
# - The IPVS corpus contains an acquisition confound that is by itself sufficient
#   to separate the diagnostic groups perfectly.
# - Consequently, classification performance reported on this corpus — including
#   ours — is not evidence of voice-based Parkinson's detection.
# - A recording-level split on a corpus with repeated readings can report AUROC
#   ≈0.88 for randomly assigned labels.
#
# May **not** be claimed:
#
# - Any statement about detecting Parkinson's disease from voice. This study
#   produces no such estimate, and the corpus cannot support one.
# - That the published work on this corpus made an error of method. What is shown
#   is that its reported numbers are not identifiable from a confound it did not
#   test for, because the necessary negative controls were not run — by anyone,
#   including us, until now.
# - Any clinical or screening implication whatsoever.
#
# ## What would be needed to answer the original question
#
# A corpus where patients and controls are recorded on the same equipment in the
# same setting, with the acquisition metadata released, and where a silence-only
# control is reported alongside the headline number. Candidates requiring a data
# use agreement are listed in `docs/NEXT_STEPS.md`.

# %%
summary = pd.DataFrame(
    [
        {"quantity": "Participants (PD / eHC)", "value": "25 / 22"},
        {"quantity": "Directories mistaken for participants", "value": "28 / 22"},
        {"quantity": "Silence-only AUROC, correct split, batch-matched",
         "value": f"{silence_matched['auroc_participant']:.3f}"},
        {"quantity": "MFCC AUROC, correct split, batch-matched",
         "value": f"{ours_matched['auroc_participant']:.3f}"},
        {"quantity": "Channel-statistics-only AUROC",
         "value": f"{grid[(grid.arm=='C_subject_key') & (grid.task=='read_text') & (grid.feature_set=='channel') & (grid.cohort=='batch_matched')].iloc[0]['auroc_participant']:.3f}"},
        {"quantity": "Age+sex-only AUROC",
         "value": f"{controls[controls.control=='age + sex only (batch_matched)'].iloc[0]['auroc']:.3f}"},
        {"quantity": "Permutation null, correct split",
         "value": f"{controls[controls.control=='label permutation null mean (C_subject_key)'].iloc[0]['auroc']:.3f}"},
        {"quantity": "Permutation null, recording-level split",
         "value": f"{controls[controls.control=='label permutation null mean (A_recording)'].iloc[0]['auroc']:.3f}"},
    ]
)
print(summary.to_string(index=False))
summary.to_csv(RESULTS / "table4_summary.csv", index=False)
print("\nwrote results/table1..table4")
