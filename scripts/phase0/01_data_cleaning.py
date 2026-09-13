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
# # 01 — Data cleaning: building a participant key that actually holds
#
# The IPVS corpus does not ship a participant identifier. It ships one directory
# per recording session, named with the participant's real first name and
# surname initial, and filenames that encode task, a scrambled name, birth year,
# sex and timestamp.
#
# This notebook decides what a "participant" is. Everything downstream depends on
# that decision, and getting it wrong fails silently, so the checks here are
# deliberately loud.
#
# **Output:** `data/processed/metadata.parquet` (pseudonymised) and a data quality
# report.

# %%
import sys
from pathlib import Path

import pandas as pd

# Walk up to the repository root so this runs identically from notebooks/,
# scripts/, or the repository root itself.
ROOT = Path.cwd().resolve()
while not (ROOT / "requirements.txt").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import parsing as P  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

IPVS = ROOT / "data" / "raw" / "ipvs"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
print("corpus root:", IPVS)

# %% [markdown]
# ## Parse every filename
#
# Anything that fails to parse is printed in full. A recording that is silently
# dropped here is a recording missing from the analysis for no recorded reason.

# %%
meta, unparsed = P.build_metadata(IPVS)
meta = P.add_audio_properties(meta)

print(f"parsed   : {len(meta)} recordings")
print(f"unparsed : {len(unparsed)}")
for path in unparsed:
    print("   UNPARSED:", path.relative_to(IPVS))

assert not unparsed, (
    f"{len(unparsed)} filenames did not match the schema. Fix the regex in "
    "src/parsing.py rather than dropping them."
)

# %% [markdown]
# ## What the three candidate participant keys disagree about
#
# Three keys are available, and they do not agree:
#
# - `folder_path` — the full directory. Looks participant-level.
# - `folder_name` — the directory basename.
# - `subject_key` — sorted name-code signature + birth year + sex.

# %%
counts = (
    meta.groupby("group")
    .agg(
        recordings=("path", "size"),
        by_subject_key=("subject_key", "nunique"),
        by_folder_path=("folder_path", "nunique"),
        by_folder_name=("folder_name", "nunique"),
    )
    .loc[["PD", "eHC", "yHC"]]
)
counts

# %% [markdown]
# ### Hazard 1 — one participant, two directories
#
# These participants were recorded in two sessions months apart and filed under
# two different severity bins. Grouping by `folder_path` treats each session as
# a separate person, so their own recordings end up on both sides of a split:
# leakage that looks like a clean participant-level split.

# %%
collisions = P.key_collisions(meta)
split_people = collisions["one_person_split_across_folders"]

if len(split_people):
    detail = (
        meta[meta["subject_key"].isin(split_people["subject_key"])]
        .groupby(["group", "subject_id", "subject_key", "folder_path"])
        .agg(recordings=("path", "size"), age=("age", "first"), sex=("sex", "first"))
        .reset_index()
        .drop(columns="subject_key")
    )
    print(f"{len(split_people)} participant(s) split across directories:\n")
    print(P.anonymise_directories(detail).to_string(index=False))
else:
    print("none")

# %% [markdown]
# ### Hazard 2 — two participants, one directory name
#
# Different people who share a first name and surname initial. Grouping by
# `folder_name` merges them into a single "participant", which corrupts the
# label-to-person mapping.

# %%
merged_names = collisions["two_people_sharing_folder_name"]

if len(merged_names):
    detail = (
        meta[meta["folder_name"].isin(merged_names["folder_name"])]
        .groupby(["group", "folder_name", "subject_id"])
        .agg(
            recordings=("path", "size"),
            age=("age", "first"),
            sex=("sex", "first"),
            sessions=("session_date", "nunique"),
        )
        .reset_index()
    )
    print(f"{len(merged_names)} directory name(s) covering more than one person:\n")
    print(P.anonymise_directories(detail).to_string(index=False))
else:
    print("none")

# %% [markdown]
# ### Hazard 3 — hand-typed birth years contain typos
#
# The birth year is not derived from anything; it was typed. A single wrong digit
# in one filename out of a session would fork that participant in two if the key
# included the raw year, inventing a participant. The year is therefore resolved
# by majority vote within participant, and every disagreement is listed here
# rather than corrected silently.

# %%
typos = collisions["birth_year_typos"]
if len(typos):
    print(f"{len(typos)} filename(s) disagree with their participant's modal year:\n")
    print(P.anonymise_directories(typos).to_string(index=False))
else:
    print("none")

# %% [markdown]
# ### Hazard 4 — the young control filenames are not usable as identifiers
#
# If the name code collapses many young controls onto one key, the code is not
# unique for that group. Checked explicitly rather than assumed, because the
# same code would otherwise silently merge people.

# %%
yhc = meta[meta["group"] == "yHC"]
yhc_check = (
    yhc.groupby("subject_key")
    .agg(folders=("folder_name", "nunique"), recordings=("path", "size"))
    .sort_values("recordings", ascending=False)
    .reset_index(drop=True)
    .rename_axis("name_code")
)
print(yhc_check.to_string())
print(
    f"\nyHC: {yhc['folder_name'].nunique()} directories collapse onto "
    f"{yhc['subject_key'].nunique()} name-code key(s)."
)

# %% [markdown]
# ### Hazard 5 — acquisition batch is aligned with the label
#
# This is the most consequential finding in the notebook, and it needs no audio
# to establish: the corpus was recorded in two campaigns, and one of them
# contains patients only.

# %%
from sklearn.metrics import roc_auc_score  # noqa: E402

pd_ehc = meta[meta["group"].isin(["PD", "eHC"])]

print("Participants per acquisition batch:")
print(P.batch_confound_report(pd_ehc).to_string())
print("\nRecordings per batch and sample rate:")
print(
    pd_ehc.groupby(["batch", "samplerate", "group"])
    .size()
    .rename("recordings")
    .to_frame()
    .to_string()
)

rate_flag = (pd_ehc["samplerate"] == 44_100).astype(int)
auc_recording = roc_auc_score(pd_ehc["label"], rate_flag)
by_participant = pd_ehc.groupby("subject_id").agg(
    label=("label", "max"), rate=("samplerate", "max")
)
auc_participant = roc_auc_score(
    by_participant["label"], (by_participant["rate"] == 44_100).astype(int)
)

print(
    f"\nSample rate alone predicts the label:"
    f"\n  recording level   AUROC = {auc_recording:.3f}"
    f"\n  participant level AUROC = {auc_participant:.3f}"
)

# %% [markdown]
# The 2016 campaign recorded patients only, at 44.1 kHz. The 2017 campaign
# recorded the remaining patients and every control, at 16 kHz. So a value read
# straight out of the WAV header carries most of a coin-flip's worth of
# diagnostic information, and any systematic difference between the two campaigns
# — microphone, room, distance, gain, preprocessing — is available to a model as a
# shortcut to the label.
#
# The three participants recorded twice are exactly the participants who appear
# in both campaigns.

# %%
both = (
    pd_ehc.groupby("subject_id")["batch"]
    .nunique()
    .loc[lambda s: s > 1]
    .index.tolist()
)
print("Participants appearing in both campaigns:", both)

# %% [markdown]
# Two cohorts are therefore defined, and every result is reported on both:
#
# | cohort | definition | purpose |
# |---|---|---|
# | `full` | all PD + eHC | reproduces what published work on this corpus uses |
# | `batch_matched` | 2017 campaign only, 16 kHz | removes the batch confound |

# %%
meta["cohort_full"] = meta["group"].isin(["PD", "eHC"])
meta["cohort_batch_matched"] = meta["cohort_full"] & (meta["recording_year"] == 2017)

matched = meta[meta["cohort_batch_matched"]]
print("batch_matched cohort:")
print(
    matched.drop_duplicates("subject_id")
    .groupby("group")
    .agg(n=("subject_id", "size"), age_mean=("age", "mean"))
    .round(1)
    .to_string()
)
print(f"  recordings: {len(matched)}")
print(f"  sample rates present: {sorted(matched['samplerate'].unique())}")

# %% [markdown]
# ## Analysis cohort
#
# Young controls are excluded for three independent reasons, any one of which
# would be sufficient:
#
# 1. **Age.** They are 19–29 while the PD group is 40–80. A classifier with no
#    access to voice at all would separate the groups on age.
# 2. **Task coverage.** They only performed the passage and word tasks — no
#    sustained vowels, no diadochokinetic task.
# 3. **Identifiability.** Their name codes are not unique, so no reliable
#    participant key exists for them.

# %%
task_matrix = (
    meta.pivot_table(
        index="group", columns="task", values="path", aggfunc="count", fill_value=0
    )
    .loc[["PD", "eHC", "yHC"]]
)
print("Recordings per task:")
print(task_matrix.to_string())

cohort = meta[meta["group"].isin(["PD", "eHC"])].copy()
print(
    f"\nAnalysis cohort: {cohort['subject_id'].nunique()} participants "
    f"({cohort[cohort.group == 'PD']['subject_id'].nunique()} PD, "
    f"{cohort[cohort.group == 'eHC']['subject_id'].nunique()} eHC), "
    f"{len(cohort)} recordings"
)

# %% [markdown]
# ## Read-passage subset
#
# The passage task is where the comparison with the published result sits, and it
# is the task whose structure makes recording-level splits leak: the protocol
# asks for the passage to be read **twice**.

# %%
read_text = cohort[cohort["task"] == "read_text"]
per_participant = (
    read_text.groupby(["group", "subject_id"])
    .agg(readings=("path", "size"), sessions=("session_index", "nunique"))
    .reset_index()
)

print("Readings per participant:")
print(
    per_participant.groupby(["group", "readings"])
    .size()
    .rename("n_participants")
    .to_frame()
    .to_string()
)
print("\nParticipants with more than one session:")
multi = per_participant[per_participant["sessions"] > 1]
print(multi.to_string(index=False) if len(multi) else "  none")

# %% [markdown]
# Participants with four readings are the ones recorded twice. Under a
# recording-level split their own readings appear in both train and test; under a
# `folder_path` split they are treated as two people, which has the same effect.

# %%
demographics = (
    cohort.drop_duplicates("subject_id")
    .groupby("group")
    .agg(
        n=("subject_id", "size"),
        age_mean=("age", "mean"),
        age_sd=("age", "std"),
        age_min=("age", "min"),
        age_max=("age", "max"),
        pct_male=("sex", lambda s: 100 * (s == "M").mean()),
    )
    .round(1)
)
print("Demographics of the analysis cohort:")
print(demographics.to_string())

# %% [markdown]
# ## Write the pseudonymised table
#
# `metadata.parquet` keeps the resolved keys because the experiment needs them.
# `metadata_public.parquet` is the redacted version safe to share: no real name,
# no directory, no timestamp, no file path.

# %%
meta.to_parquet(PROCESSED / "metadata.parquet", index=False)
P.redact(meta).to_parquet(PROCESSED / "metadata_public.parquet", index=False)

print("wrote:")
for name in ("metadata.parquet", "metadata_public.parquet"):
    print(f"  data/processed/{name}")
print("\nColumns:", list(meta.columns))

# Preview the redacted table, not the internal one: this output is committed.
P.redact(meta).head(8)
