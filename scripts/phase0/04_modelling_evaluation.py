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
# # 04 — Modelling and evaluation
#
# Modelling and evaluation are one notebook because the evaluation *is* the outer
# cross-validation loop; splitting them would mean either fitting twice or passing
# fitted folds between files.
#
# One estimator, `RandomForestClassifier(n_estimators=100)` at library defaults,
# is used everywhere. It is not tuned. Tuning would confound the comparison, and
# the question here is not how high the number can go — it is what the number
# means.
#
# ## The grid
#
# | axis | values |
# |---|---|
# | split arm | `A_recording` (no grouping), `B_folder_path` (grouped by directory), `C_subject_key` (grouped by verified participant) |
# | cohort | `full`, `batch_matched` (2017 campaign only, single sample rate) |
# | task | `read_text`, `vowel_a` |
# | feature set | `mfcc`, `egemaps`, `silence`, `channel` |
#
# `silence` and `channel` are the controls. Neither contains speech, so any
# performance they reach is attributable to acquisition rather than to voice.

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

from src import experiment as E  # noqa: E402
from src import splits as S  # noqa: E402
from src import viz as V  # noqa: E402

V.use_house_style()
pd.set_option("display.width", 180)
pd.set_option("display.max_rows", 120)

FEATURE_DIR = ROOT / "data" / "processed" / "features"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CELLS = [
    ("read_text", "mfcc", "full"),
    ("read_text", "mfcc", "batch_matched"),
    ("read_text", "egemaps", "full"),
    ("read_text", "egemaps", "batch_matched"),
    ("read_text", "silence", "full"),
    ("read_text", "silence", "batch_matched"),
    ("read_text", "channel", "full"),
    ("read_text", "channel", "batch_matched"),
    ("vowel_a", "mfcc", "full"),
    ("vowel_a", "mfcc", "batch_matched"),
    ("vowel_a", "egemaps", "full"),
    ("vowel_a", "egemaps", "batch_matched"),
    ("vowel_a", "channel", "full"),
    ("vowel_a", "channel", "batch_matched"),
]
print(f"{len(CELLS)} cells x {len(S.SCHEMES)} arms = {len(CELLS) * len(S.SCHEMES)} runs")
for arm, (column, label) in S.SCHEMES.items():
    print(f"  {arm:15s} group key = {str(column):12s} {label}")

# %% [markdown]
# ## Verify the arms differ in the way they are supposed to
#
# Before reading any AUROC, confirm that arm C really is participant-disjoint and
# that arms A and B really do leak. If this check is wrong, nothing below means
# anything.

# %%
X, y, meta = E.load_features(FEATURE_DIR, "read_text", "mfcc", "full")
leak_rows = []
for arm in S.SCHEMES:
    report = S.leakage_report(meta, y, arm)
    leak_rows.append(
        {
            "arm": arm,
            "folds": len(report),
            "folds_with_leakage": int((report["n_leaked_participants"] > 0).sum()),
            "mean_leaked_participants": report["n_leaked_participants"].mean(),
            "mean_leaked_test_recordings": report["n_leaked_recordings"].mean(),
            "pct_test_recordings_leaked": 100
            * report["n_leaked_recordings"].sum()
            / report["n_test_recordings"].sum(),
        }
    )
leak_table = pd.DataFrame(leak_rows).set_index("arm").round(2)
print(leak_table.to_string())

assert leak_table.loc["C_subject_key", "folds_with_leakage"] == 0
assert leak_table.loc["A_recording", "folds_with_leakage"] > 0
print("\nArm C is participant-disjoint. Arms A and B are not.")

# %% [markdown]
# Arm A places one of a participant's readings in training while scoring the
# other in almost every fold. Arm B leaks only through the three participants
# recorded in two campaigns — a much smaller effect, and the realistic mistake:
# it looks like a participant-level split.

# %% [markdown]
# ## Run the grid

# %%
grid, store = E.run_grid(FEATURE_DIR, CELLS, n_splits=5, n_repeats=5, seed=42)
grid.to_parquet(RESULTS / "grid.parquet", index=False)

main = grid.pivot_table(
    index=["task", "feature_set", "cohort"], columns="arm", values="auroc_participant"
).round(3)
print("Participant-level AUROC by split arm:")
print(main.to_string())

# %% [markdown]
# ## Figure 6 — The headline result
#
# Read this figure by comparing the controls (`silence`, `channel`) against the
# speech features. If a control performs as well as speech, the model is not
# listening to the voice.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

FEATURE_ORDER = ["silence", "channel", "mfcc", "egemaps"]
FEATURE_LABEL = {
    "silence": "silence only\n(no speech)",
    "channel": "channel stats\n(8 numbers)",
    "mfcc": "MFCC\n(speech)",
    "egemaps": "eGeMAPSv02\n(speech)",
}

for ax, cohort in zip(axes, ("full", "batch_matched")):
    subset = grid[(grid["cohort"] == cohort) & (grid["task"] == "read_text")]
    present = [f for f in FEATURE_ORDER if f in set(subset["feature_set"])]
    x = np.arange(len(present))
    width = 0.26
    for i, arm in enumerate(S.SCHEMES):
        vals, los, his = [], [], []
        for feature in present:
            row = subset[(subset["feature_set"] == feature) & (subset["arm"] == arm)]
            vals.append(row["auroc_participant"].iloc[0])
            los.append(row["auroc_participant"].iloc[0] - row["ci_lo"].iloc[0])
            his.append(row["ci_hi"].iloc[0] - row["auroc_participant"].iloc[0])
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            yerr=[los, his],
            capsize=3,
            color=V.ARM_COLOURS[arm],
            label=arm.replace("_", " "),
            error_kw={"linewidth": 1, "ecolor": "#444444"},
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.text(len(present) - 0.55, 0.515, "chance", fontsize=8, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABEL[f] for f in present], fontsize=8.5)
    n_part = subset["n_participants"].max()
    ax.set_title(f"{cohort}  (n = {n_part} participants)")
    ax.set_ylim(0.35, 1.06)

axes[0].set_ylabel("Participant-level AUROC (95% bootstrap CI)")
axes[0].legend(title="split arm", fontsize=8, title_fontsize=8, loc="lower left")
fig.suptitle(
    "Read passage: the silence-only control matches the speech features",
    fontsize=12.5,
    fontweight="bold",
    y=1.0,
)
V.save_fig(
    fig,
    "fig06_main_result",
    "Participant-level AUROC on the read passage. A model given only the "
    "non-speech portions of the recordings performs as well as one given the "
    "speech, on both cohorts and under all three split schemes.",
)
plt.show()

# %% [markdown]
# ## Figure 7 — Same comparison on the sustained vowel
#
# The vowel task is shorter and almost entirely phonation, so it offers less
# background for a channel signature to accumulate in.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, cohort in zip(axes, ("full", "batch_matched")):
    subset = grid[(grid["cohort"] == cohort) & (grid["task"] == "vowel_a")]
    present = [f for f in FEATURE_ORDER if f in set(subset["feature_set"])]
    x = np.arange(len(present))
    for i, arm in enumerate(S.SCHEMES):
        vals = [
            subset[(subset["feature_set"] == f) & (subset["arm"] == arm)][
                "auroc_participant"
            ].iloc[0]
            for f in present
        ]
        errs = [
            [
                v - subset[(subset["feature_set"] == f) & (subset["arm"] == arm)][
                    "ci_lo"
                ].iloc[0]
                for f, v in zip(present, vals)
            ],
            [
                subset[(subset["feature_set"] == f) & (subset["arm"] == arm)][
                    "ci_hi"
                ].iloc[0] - v
                for f, v in zip(present, vals)
            ],
        ]
        bars = ax.bar(
            x + (i - 1) * 0.26, vals, 0.26, yerr=errs, capsize=3,
            color=V.ARM_COLOURS[arm], label=arm.replace("_", " "),
            error_kw={"linewidth": 1, "ecolor": "#444444"},
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABEL[f] for f in present], fontsize=8.5)
    ax.set_title(f"{cohort}  (n = {subset['n_participants'].max()} participants)")
    ax.set_ylim(0.35, 1.06)
axes[0].set_ylabel("Participant-level AUROC (95% CI)")
axes[0].legend(title="split arm", fontsize=8, title_fontsize=8, loc="lower left")
fig.suptitle("Sustained vowel /a/", fontsize=12.5, fontweight="bold", y=1.01)
V.save_fig(fig, "fig07_vowel_result",
           "Participant-level AUROC on the sustained vowel task.")
plt.show()

# %% [markdown]
# ## How much does the split scheme itself change the answer?
#
# Paired bootstrap on the participants common to both arms, so the difference is
# attributable to the split and not to a different sample.

# %%
effect = E.leakage_effect(grid, store)
effect.to_parquet(RESULTS / "leakage_effect.parquet", index=False)
print("Delta AUROC relative to the correct participant-level split:")
print(
    effect.pivot_table(
        index=["task", "feature_set", "cohort"], columns="arm", values="delta_auroc"
    )
    .round(3)
    .to_string()
)
print("\nFull table with intervals:")
print(effect.round(3).to_string(index=False))

# %% [markdown]
# ## Figure 8 — Training diagnostics
#
# Random forests have no per-epoch loss curve, so the diagnostics that matter here
# are the ones that reveal optimism and whether more data would help: the gap
# between training and held-out performance in every fold, out-of-bag error
# against forest size, and a learning curve whose x axis counts participants
# rather than recordings.
#
# These are shown on the `channel` feature set. With MFCC every quantity is
# pinned at 1.000 and the panels are unreadable — which is itself the finding, so
# panel (c) overlays both.

# %%
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

DIAG_TASK, DIAG_COHORT = "read_text", "batch_matched"
X, y, meta = E.load_features(FEATURE_DIR, DIAG_TASK, "channel", DIAG_COHORT)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

# (a) train vs held-out AUROC per fold, per arm. Distance below the diagonal is
# the optimism a single train-test split would have hidden.
for arm in S.SCHEMES:
    per_fold = store[(DIAG_TASK, "channel", DIAG_COHORT, arm)]["per_fold"]
    axes[0].scatter(
        per_fold["auroc_train_recording"],
        per_fold["auroc_test_recording"],
        s=38,
        alpha=0.8,
        color=V.ARM_COLOURS[arm],
        label=arm.replace("_", " "),
        edgecolor="white",
        linewidth=0.5,
    )
axes[0].plot([0.5, 1.02], [0.5, 1.02], color="black", linestyle="--", linewidth=1,
             label="no optimism")
axes[0].set_xlabel("AUROC on the training folds")
axes[0].set_ylabel("AUROC on the held-out fold")
axes[0].set_title("(a) Optimism per fold (channel features)")
axes[0].legend(fontsize=7.5, loc="lower left")

# (b) out-of-bag error against forest size: the forest's own convergence check.
sizes = [10, 25, 50, 75, 100, 150, 200, 300, 500]
oob = {}
for feature_set in ("channel", "mfcc"):
    X_f, y_f, _ = E.load_features(FEATURE_DIR, DIAG_TASK, feature_set, DIAG_COHORT)
    errors = []
    for n in sizes:
        forest = RandomForestClassifier(
            n_estimators=n, oob_score=True, bootstrap=True, random_state=42, n_jobs=-1
        )
        forest.fit(X_f.to_numpy(), y_f)
        errors.append(1 - forest.oob_score_)
    oob[feature_set] = errors
    axes[1].plot(
        sizes, errors, marker="o", linewidth=1.8,
        color="#0072B2" if feature_set == "channel" else "#D55E00",
        label=feature_set,
    )
axes[1].axvline(100, color="#666666", linestyle=":", linewidth=1.5,
                label="n_estimators used = 100")
axes[1].set_xlabel("Number of trees")
axes[1].set_ylabel("Out-of-bag error")
axes[1].set_title("(b) Forest size is not the limitation")
axes[1].legend(fontsize=8)

# (c) learning curve by participant count, both feature sets.
curves = {}
for feature_set, colour in (("channel", "#0072B2"), ("mfcc", "#D55E00")):
    X_f, y_f, meta_f = E.load_features(FEATURE_DIR, DIAG_TASK, feature_set, DIAG_COHORT)
    curve = E.learning_curve_by_participant(X_f, y_f, meta_f, n_repeats=3)
    curve["feature_set"] = feature_set
    curves[feature_set] = curve
    agg = curve.groupby("n_train_participants")["auroc_test"].agg(["mean", "std"])
    xs = agg.index.to_numpy()
    axes[2].plot(xs, agg["mean"], marker="o", color=colour, linewidth=1.8,
                 label=f"{feature_set}, held-out")
    axes[2].fill_between(
        xs, agg["mean"] - agg["std"], agg["mean"] + agg["std"], color=colour, alpha=0.18
    )
curve_all = pd.concat(curves.values(), ignore_index=True)
curve_all.to_parquet(RESULTS / "learning_curve.parquet", index=False)
axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
axes[2].set_ylim(0.4, 1.05)
axes[2].set_xlabel("Participants in the training set")
axes[2].set_ylabel("Held-out AUROC")
axes[2].set_title("(c) Learning curve, by participant")
axes[2].legend(fontsize=8, loc="lower right")

fig.suptitle(
    f"Training diagnostics: {DIAG_TASK}, {DIAG_COHORT} cohort, participant-level split",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)
V.save_fig(
    fig,
    "fig08_training_diagnostics",
    "Per-fold optimism, out-of-bag error against forest size, and a learning "
    "curve whose x axis counts participants rather than recordings. MFCC is "
    "already saturated with twelve training participants.",
)
plt.show()

print("Out-of-bag error by forest size:")
print(pd.DataFrame(oob, index=sizes).round(4).to_string())
print("\nLearning curve, held-out AUROC:")
print(
    curve_all.pivot_table(
        index="n_train_participants", columns="feature_set", values="auroc_test"
    )
    .round(3)
    .to_string()
)

# %% [markdown]
# ## Figure 9 — ROC curves
#
# Out-of-fold predictions pooled across repeats, aggregated to one score per
# participant.

# %%
from sklearn.metrics import roc_curve  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
panels = [
    ("read_text", "batch_matched", "Read passage, batch-matched cohort"),
    ("vowel_a", "batch_matched", "Sustained vowel, batch-matched cohort"),
]
for ax, (task, cohort, title) in zip(axes, panels):
    for feature_set, style in (("mfcc", "-"), ("silence", ":"), ("channel", "--")):
        key = (task, feature_set, cohort, "C_subject_key")
        if key not in store:
            continue
        part = store[key]["participant"]
        fpr, tpr, _ = roc_curve(part["y_true"], part["y_score"])
        auc = store[key]["auroc_participant"][0]
        ax.plot(fpr, tpr, style, linewidth=2,
                label=f"{feature_set} (AUROC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
fig.suptitle(
    "Participant-level ROC, correct split (arm C)",
    fontsize=12, fontweight="bold", y=1.0,
)
V.save_fig(fig, "fig09_roc", "Participant-level ROC curves under the correct split.")
plt.show()

# %%
grid.round(4).to_csv(RESULTS / "grid.csv", index=False)
print("wrote results/grid.parquet, results/grid.csv, results/leakage_effect.parquet")
