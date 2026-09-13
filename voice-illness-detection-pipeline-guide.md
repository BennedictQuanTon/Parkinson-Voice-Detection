# Pipeline guide — revised after Phase 0

The original version of this document was an engineering plan: source a corpus,
extract features, train a model, track experiments, package a demo. It was
competent as engineering and wrong as science, and Phase 0 demonstrated exactly
how. This revision records what changed and why, so the reasoning is not lost.

## What the original plan got right

- Two feature tracks — interpretable acoustics (MFCC, jitter/shimmer/HNR) and
  learned embeddings — rather than committing to one.
- Restricting perturbation measures to sustained phonation, where they are defined.
- Insisting on experiment tracking and a versioned corpus snapshot.
- Choosing a small, free, locally runnable corpus over an ambitious one.

## What it got wrong

### It assumed a file that did not exist

`parkinson_pipeline.py` was referenced five times as though it were already
written. It was not in the repository. A plan that cites its own unwritten
artefacts cannot be executed or reviewed. `src/` now contains six real modules,
each covered by tests.

### It planned to split at recording level

The plan's cross-validation was over recordings. IPVS asks each participant to
read the passage **twice**. Measured directly: a recording-level split places a
participant's own other recording in the training set for **85.5%** of held-out
recordings.

The cost is not visible in the headline number — with real labels every split arm
sits at the same value, because the corpus confound already saturates performance.
It is visible under label shuffling: a recording-level split reports AUROC
**0.882** for randomly assigned labels, where a participant-level split reports
0.447. That is the number a recording-level result cannot be distinguished from.

### It treated the corpus directory structure as the participant identifier

IPVS ships no participant ID. The plan implicitly assumed one directory equals one
person. Six defects break that assumption, including three participants filed under
two directories each, two different people sharing a directory name, two controls
whose scrambled name codes are anagrams, and a birth-year typo that invents a 26th
patient. All are documented in `FINDINGS.md` and each is caught by a check that
fails loudly.

### It had no negative controls at all

This is the omission that mattered. The plan measured how well a model performs
and never asked what the same pipeline reports when there is nothing to find. Four
controls now run:

| control | result |
|---|---|
| age + sex only | 0.618 — as expected |
| label permutation, correct split | 0.447 — as expected |
| label permutation, recording-level split | **0.882** |
| **silence only, no speech at all** | **1.000** |

The last one ends the project's original question. A model given only the portions
of each recording where nobody is speaking separates the groups perfectly, under a
participant-disjoint split, on a cohort restricted to one campaign and one sample
rate. Patients and controls were recorded under different conditions.

### It would have reported accuracy on an imbalanced case-control corpus

Accuracy at 25 versus 22 participants is nearly uninterpretable, and the plan's
target metrics did not include an interval. Everything is now reported as
participant-level AUROC with a bootstrap interval resampled **by participant** —
resampling recordings treats a participant's two readings as independent and
produces intervals far too narrow.

### It planned SMOTE

Removed. van den Goorbergh et al. (2022, *JAMIA*) show resampling drives the
calibration intercept from 0.06 to −1.32/−1.50 with no AUROC gain. The same
sensitivity/specificity trade-off is available by moving the decision threshold.

### It had no cost function beyond "train a classifier"

Two tiers were missing and are now specified in `docs/NEXT_STEPS.md`: the training
loss (log loss / binary cross-entropy, with inner-loop tuning scored on
`neg_log_loss` or `neg_brier_score`, never accuracy or F1), and the decision cost
(net benefit and decision curve analysis against treat-all and treat-none, with
the threshold set from the cost ratio `p_t = C_FP/(C_FP+C_FN)`).

## The revised pipeline

```
01_data_cleaning          resolve participants, document defects, pseudonymise
02_eda                    demographics, repeated measures, acquisition exposure
03_feature_engineering    MFCC, eGeMAPSv02, and two no-speech controls
04_modelling_evaluation   3 split arms x 2 cohorts x 2 tasks x 4 feature sets
05_negative_controls      demographics, permutation, campaign, silence
06_results                tables, comparison with published work, claims
```

Modelling and evaluation are one notebook because the evaluation *is* the outer
cross-validation loop.

### Non-negotiables

1. **Every split is grouped by a verified participant key**, and a test asserts
   arm C is disjoint while arms A and B are not. If that contrast ever disappears,
   the experiment has lost its control.
2. **Every interval resamples participants**, never recordings.
3. **Negative controls run before results are read**, not after a reviewer asks.
4. **A silence-only control accompanies every headline number.** If it is not near
   chance, the headline number is not about voice.
5. **Nothing that fails is silently dropped.** Feature extractors that produce
   nothing return `None`, the recording is reported, and the count appears in the
   manifest.
6. **No audio and no real name is ever committed.**

### What "a successful model" means here

Not an AUROC threshold. A result is publishable when:

- **Tier 0, validity.** Splits are participant-disjoint; the silence-only control
  is near chance; the permutation null is centred on 0.5; demographics-only is near
  chance; the acquisition metadata carries no label information. **IPVS fails this
  tier, so no later tier applies.**
- **Tier 1, performance.** Participant-level AUROC with an interval that excludes
  the demographics-only baseline, on a corpus that passed Tier 0.
- **Tier 2, reporting.** TRIPOD+AI's 27 items, PROBAST+AI's 16 signalling
  questions, a model card, a corpus datasheet.
- **Tier 3, engineering.** Pinned dependencies, seeded splits, one-command
  reproduction, tests that fail on a leak.

Phase 0 ends at Tier 0 with a failure, which is a result. It is a more defensible
contribution than an AUROC of 0.99 that measures a microphone.

## Reference points

- Klempír, Krupička & Krupil (2024), *Sensors* 24(17):5520 — the comparison study,
  same corpus, random forest, accuracy 0.85–0.95, grouping not stated, no negative
  controls.
- Kapoor & Narayanan (2023), *Patterns* — eight types of leakage and a 21-question
  model info sheet. Would have caught the original design.
- Collins et al. (2024), *BMJ* — TRIPOD+AI.
- van den Goorbergh et al. (2022), *JAMIA* — why not to resample.
- Dimauro & Girardi (2019), IEEE DataPort — the corpus.
