# Dataset version and provenance

## Corpus

**Italian Parkinson's Voice and Speech (IPVS)**
Dimauro, G. & Girardi, F. (2019). IEEE DataPort.
<https://doi.org/10.21227/aw6b-tg17> — CC BY 4.0.

Associated publication: Dimauro, G. et al. (2017). "Assessment of speech
intelligibility in Parkinson's disease using a speech-to-text system."
*IEEE Access* 5:22199–22208.

## Snapshot used

| property | value |
|---|---|
| retrieved | 2026-09 |
| total `.wav` files | 831 |
| size on disk | 812 MB |
| top-level directories | 3 |
| recording campaigns | 2 (2016, 2017) |
| sample rates present | 16 kHz (671 files), 44.1 kHz (160 files) |
| encoding | PCM_16, mono, throughout |

Directory contents as shipped:

```
28 People with Parkinson's disease/   28 directories, 437 wav, + FILE CODES.xlsx, TAB 5.xlsx
  1-5/  6-10/  11-16/  17-28/         (severity bins)
22 Elderly Healthy Control/           22 directories, 349 wav, + FILE CODES.xlsx, Tab 3.xlsx
15 Young Healthy Control/             15 directories,  45 wav, + FILE CODES.xlsx, 15 YHC.xlsx
```

## Filename schema

Decoded in `src/parsing.py`. Example:

```
B1APGANRET55F170320171104.wav
│ │        │ ││       │
│ │        │ ││       └── hhmm            11:04
│ │        │ │└────────── ddmmyyyy        17 Mar 2017
│ │        │ └─────────── sex             F
│ │        └───────────── birth year      1955
│ └────────────────────── scrambled name  APGANRET
└──────────────────────── task + index    B1
```

The date is `ddmmyy` in the 2016 campaign and `ddmmyyyy` in 2017. The scrambled
name is not stable across sessions, so its letters must be sorted before
comparison.

Task codes:

| code | task | repetitions |
|---|---|---|
| `B1`, `B2` | phonemically balanced passage, read aloud | 2 |
| `D1`, `D2` | diadochokinetic (`/pa/ /ta/`) | 2 |
| `VA`,`VE`,`VI`,`VO`,`VU` ×2 | sustained vowels | 2 each |
| `PR1` | words | 1 |
| `FB1` | phrases | 1 |

## Derived cohorts

| cohort | definition | PD | controls | recordings |
|---|---|---|---|---|
| `full` | all PD + elderly controls | 25 | 22 | 786 |
| `batch_matched` | 2017 campaign only, 16 kHz | 18 | 22 | 626 |

Young controls (45 recordings) are excluded from both; see `FINDINGS.md`.

## Integrity checks run on load

`01_data_cleaning` fails rather than proceeding if any of these is violated:

- every `.wav` filename matches the schema (currently 831/831)
- every participant carries exactly one diagnostic label
- no `(participant, task, repetition, session)` tuple appears twice
- the participant key is disjoint across cross-validation folds in arm C

## Privacy

The corpus is CC BY 4.0, but directory names are real participant first names with
surname initials and filenames encode birth year, sex and exact recording
timestamp. Combined with the diagnostic label this is identifiable health data.

This repository therefore:

- never commits audio (`.gitignore` blocks `data/` and `*.wav`)
- pseudonymises to `PD01`/`eHC01`/`yHC01`, ordered by a hash of the participant key
  so the mapping does not reveal alphabetical position
- writes `metadata_public.parquet` with name, path, directory and timestamp columns
  removed (`src/parsing.redact`, covered by a test)
- never writes a real name into a figure, table or log

## Datasets not used, and why

| corpus | status | reason |
|---|---|---|
| MDVR-KCL | not downloaded | deferred to Phase 1; single reading per participant makes it the natural replication set |
| NeuroVoz | requires DUA | request not submitted; needs institutional affiliation |
| EWA-DB | requires DUA | as above |
| UCI Parkinson's telemonitoring | rejected | 188/64 subjects with a sex confound and no raw audio |
| Bridge2AI Voice | rejected | no per-individual raw audio available |
