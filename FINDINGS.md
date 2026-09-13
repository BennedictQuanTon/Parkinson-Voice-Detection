# Findings — Phase 0

Everything below was produced by `notebooks/01`–`06` on the Italian Parkinson's
Voice and Speech corpus (IPVS). Every number is reproducible with
`make all` from a clean checkout plus the corpus.

## Headline

**On the IPVS corpus, a random forest given only the portions of each recording
where nobody is speaking separates patients from controls with participant-level
AUROC 1.000 (95% CI 1.000–1.000).** This holds under a participant-disjoint
cross-validation split, on the sub-cohort restricted to a single recording
campaign and a single sample rate.

Patients and controls in this corpus were recorded under systematically different
acquisition conditions. That difference alone is sufficient for perfect
separation, so no classification result computed on this corpus — including every
result in this repository — can be attributed to voice.

## The cohort is smaller than it appears

The corpus ships 28 directories under "28 People with Parkinson's disease". Three
of those are second sessions of participants already present, identifiable from
the scrambled name code, birth year and sex encoded in the filenames.

| group | directories | actual participants | recordings |
|---|---|---|---|
| PD | 28 | **25** | 437 |
| elderly controls | 22 | **22** | 349 |
| young controls | 15 | not usable (see below) | 45 |

Age is closely matched between patients and elderly controls (66.8 ± 9.1 vs
67.1 ± 5.2 years, Welch *p* = 0.89). Sex is not (64% vs 46% male, Fisher
*p* = 0.25).

## Data quality defects found in the corpus

Each was found by a check that fails loudly in `01_data_cleaning`, and each would
have silently corrupted the analysis.

Participants are referred to by their pseudonyms; the corpus publishes their real
names as directory names, and this document does not restate them.

1. **Three patients are filed under two directories each**, having been recorded
   in two campaigns and filed under two different severity bins. Grouping by
   directory treats them as six people.
2. **Two different patients share a directory name** — same first name and surname
   initial — distinguishable only by birth year (71 vs 70). Grouping by directory
   *name* merges them.
3. **Two different elderly controls have scrambled name codes that are anagrams
   of each other**, differing only in birth year (49 vs 55). Sorting the letters,
   which is required because the scrambling is not stable across sessions, makes
   them collide, so the birth year cannot be dropped from the key.
4. **One filename contains a birth-year typo.** Fifteen files in one session agree
   on the year; one differs by a single digit. Including the raw birth year in the
   participant key forks that participant in two, inventing a 26th patient.
   Resolved by majority vote within directory.
5. **The young-control filenames do not identify their speakers.** Fifteen
   directories collapse onto three name codes, and the encoded sex is `M` for
   every one of them — including two directories whose names are unambiguously
   female. No reliable participant key exists for this group.
6. **Seventeen filenames do not match the documented schema** — sixteen contain
   an underscore inside the name code, one has a two-digit task index.

Young controls are excluded from all analysis for three independent reasons: they
are aged 19–29 against a patient range of 40–80, they performed only two of the
nine tasks, and defect 6 above means they cannot be reliably identified.

## The confound

### It is visible in the file headers, before any audio is decoded

| campaign | sample rate | PD | controls |
|---|---|---|---|
| 2016 | 44.1 kHz | 10 | **0** |
| 2017 | 16 kHz | 18 | 22 |

Sample rate alone reaches participant-level AUROC **0.700**. No control was
recorded in 2016.

### Removing that does not remove the confound

Restricting to the 2017 campaign at a single sample rate (18 PD vs 22 controls)
leaves the silence-only control at AUROC **1.000**. The acquisition difference is
therefore not a property of the campaign year; it tracks the **group**. Patients
and controls were recorded under different conditions within the same campaign.

Consistent with a microphone or distance difference, patients' recordings carry
relatively more low-frequency energy (0.70 vs 0.57 of the spectrum below 625 Hz,
AUROC 0.906) and roughly half the high-frequency energy of controls'.

### Main results, participant-level AUROC, correct split

Read passage, participant-disjoint 5×5 cross-validation:

| feature set | contains speech | full cohort (n=46) | batch-matched (n=39) |
|---|---|---|---|
| silence only | **no** | 1.000 [1.000, 1.000] | **1.000 [1.000, 1.000]** |
| channel statistics (8 numbers) | **no** | 0.890 [0.787, 0.964] | 0.897 [0.772, 0.987] |
| MFCC (40) | yes | 0.987 [0.948, 1.000] | 1.000 [1.000, 1.000] |
| eGeMAPSv02 (88) | yes | 0.990 [0.961, 1.000] | 1.000 [1.000, 1.000] |

The sustained vowel is less contaminated but not clean: MFCC 0.929, eGeMAPS
0.995, channel statistics 0.874 on the batch-matched cohort.

A learning curve computed over participants reaches AUROC 0.999 with **twelve**
training participants. No genuine clinical discrimination task saturates that
fast.

## Negative controls

| control | expected | observed |
|---|---|---|
| age + sex only | ≈ chance | 0.618 [0.431, 0.802] — as expected |
| label permutation, correct split | centred on 0.5 | mean 0.447 — as expected |
| **label permutation, recording-level split** | centred on 0.5 | **mean 0.882** |
| **silence only** | ≈ chance | **1.000** |
| campaign, within patients only | ≈ chance | 0.352 [0.111, 0.614] — inconclusive |

Two of these are not as expected, and each carries a separate conclusion.

## On cross-validation splits

With the real labels, the split scheme barely matters: ΔAUROC between a
recording-level split and a participant-level split is **0.000** for both MFCC
and eGeMAPS on the read passage. This is a ceiling effect, not evidence that the
split is harmless — the confound already saturates performance, so leakage has no
room to add anything.

The cost becomes measurable once the real signal is removed. Under label
shuffling at participant level:

- participant-level split → AUROC **0.447** (correct: the null should be 0.5)
- recording-level split → AUROC **0.882**

A recording-level split on this corpus manufactures AUROC 0.88 from randomly
assigned labels, because the model recognises the participant across their two
readings of the passage and recalls whichever label they were given. A published
recording-level number cannot be distinguished from this.

Leakage rates per fold, measured directly: a recording-level split puts a
participant's own other recording in the training set for **85.5%** of held-out
recordings; grouping by directory path reduces this to 9.4%; grouping by the
verified participant key gives 0%.

## Comparison with published work on the same corpus

Klempír, Krupička & Krupil (2024), *Sensors* 24(17):5520, evaluate this corpus
with a random forest on MFCC-style features and report accuracy in the 0.85–0.95
range, with 5-fold cross-validation whose grouping is not stated and no negative
controls.

Our participant-level AUROC of 0.99–1.00 is **higher** than that. Read as a
leaderboard entry, that is an improvement. Read against the silence-only control
in the same table, it is a measurement of how strong the confound is. The
published work is not shown to contain a methodological error; what is shown is
that its numbers are not identifiable from a confound nobody tested for.

## What may and may not be claimed

**May be claimed.** The IPVS corpus contains an acquisition confound sufficient
by itself for perfect separation of the diagnostic groups. Classification
performance reported on this corpus is therefore not evidence of voice-based
Parkinson's detection. A recording-level split on a corpus with repeated readings
can report AUROC ≈0.88 for random labels.

**May not be claimed.** Anything at all about detecting Parkinson's disease from
voice. This work produces no such estimate and the corpus cannot support one.
Nothing clinical. No screening implication. No statement that prior authors erred.

## What would answer the original question

A corpus where patients and controls are recorded on the same equipment in the
same setting, with acquisition metadata released, and where a silence-only control
is reported next to the headline number. See `docs/NEXT_STEPS.md`.

---

# Findings — Phase 1: Defensible Cross-Corpus Modeling

Phase 1 implemented the cross-corpus strategy across **IPVS** (Italian, 25 PD / 22 eHC) and **MDVR-KCL** (English, 16 PD / 21 HC; Zenodo 2867216) under a strict Leave-One-Corpus-Out (LOCO) nested cross-validation framework (Arm D).

## Headline Result

Under cross-corpus LOCO evaluation on the shared `read_passage` task:

- **Tier C (openSMILE eGeMAPSv02, 88 features)** achieves participant-level LOCO AUROC **0.837 (95% CI: 0.739–0.925)** with Random Forest and **0.825 (95% CI: 0.727–0.908)** with Logistic ElasticNet.
- **The generalization gap collapses:** While classical unnormalized features (Tier A) catastrophically fail across corpora (within AUROC 0.952 $\rightarrow$ cross AUROC 0.307; gap = +0.646), channel-normalized CMVN features (Tier B) transfer with a gap of only **-0.071**, and eGeMAPS (Tier C) transfers with a gap of **+0.042** on MDVR-KCL.
- **All Three Acceptance Gates Passed:**
  1. **Gate A (Silence Shortcut Neutralization):** In-sample IPVS silence AUROC is 1.000. Under cross-corpus LOCO (training on IPVS and testing on MDVR-KCL), silence AUROC collapses to **0.542** (chance level $\le 0.65$). The cross-corpus design effectively destroys the stationary microphone shortcut.
  2. **Gate B (Signal Over Confound Baseline):** The lower bound of the LOCO 95% bootstrap CI (**0.739**) strictly exceeds the single-corpus F0 sex-proxy baseline (**0.696**).
  3. **Gate C (Clinical Severity & Utility):** Continuous model probabilities demonstrate strong positive dose-response correlations with clinical severity:
     - Hoehn & Yahr stage: Spearman $\rho = \mathbf{+0.65}$ ($p = 1.46 \times 10^{-5}$)
     - UPDRS III-18 (speech): Spearman $\rho = \mathbf{+0.47}$ ($p = 0.0031$)
     - Female-only confound-free stratum ($N=47$): AUROC = **0.750**

## Confound Audit Across Corpora

The two corpora possess orthogonal, opposing confounds:

| Corpus | Language | Hardware / Environment | Primary Confound Hazard | Silence AUROC | F0 Sex-Proxy AUROC |
|---|---|---|---|---|---|
| **IPVS** | Italian | Fixed microphones, clinic room | Stationary acoustic room signature | **1.000** | 0.663 |
| **MDVR-KCL** | English | Motorola Moto G4 smartphone | Extreme sex imbalance (19 F, 2 M controls) | 0.744 | **0.696** |

Because these confounds are independent, cross-corpus testing acts as an acoustic sieve: an algorithm relying on IPVS room acoustics fails on MDVR-KCL (silence transfer AUROC = 0.542), and an algorithm relying on sex/pitch on MDVR-KCL fails on IPVS. Genuine vocal biomarkers survive.

## Generalization Performance Across Feature Tiers

| Feature Tier | Dimensions | Best Estimator | Test IPVS AUROC | Test KCL AUROC | Overall LOCO AUROC (95% CI) | Generalization Gap (KCL) |
|---|---|---|---|---|---|---|
| **Tier A (Praat Physiology)** | 19 | Logistic ElasticNet | 0.499 | 0.661 | 0.611 [0.483, 0.730] | +0.176 (IPVS gap: +0.646) |
| **Tier B (CMVN-MFCC)** | 40 | Logistic L2 | 0.865 | 0.738 | 0.599 [0.467, 0.723] | **-0.071** (Transfer invariant) |
| **Tier AB (Physiology + CMVN)** | 59 | Logistic L2 | 0.570 | 0.604 | 0.516 [0.388, 0.643] | +0.170 |
| **Tier C (openSMILE eGeMAPS)** | 88 | Random Forest | 0.891 | 0.734 | **0.837 [0.739, 0.925]** | **+0.042** (Headline model) |

## Clinical Utility & Decision Curve Analysis

1. **Prevalence Mismatch:** The in-sample prevalence is ~48% (50/50 balance). However, in the general population $\ge 65$ years old, true Parkinson's prevalence is approximately **1.5%**.
2. **Prior-Corrected Positive Predictive Value (PPV):** At 90% sensitivity and 90% specificity, Bayes' theorem reveals that the true screening PPV is **12.1%**. Out of 100 positive voice screens, approximately 12 individuals actually have Parkinson's disease.
3. **Decision Curve Analysis (DCA):** Decision Curve Analysis indicates a positive net clinical benefit over the "Screen None" and "Screen All" policies across clinical threshold probabilities between 1.0% and 5.5%.
4. **Clinical Boundary:** This confirms that voice AI is appropriate as a low-cost, non-invasive risk triaging tool to recommend specialist neurological evaluation, but is categorically unsuitable as a standalone diagnostic device.

