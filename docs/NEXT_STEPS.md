# Next steps

Phase 0 answered a different question than the one it set out to answer. The
original question — can Parkinson's disease be detected from voice — is untouched,
because IPVS cannot address it. What follows is what would.

## 1. Replicate the silence control on other corpora

The finding is only as interesting as it is general. The same three-line control
(extract non-speech regions, train, report participant-level AUROC) should be run
on every corpus in this area. If it comes out at chance elsewhere, IPVS is an
outlier. If it does not, the field has a systematic problem.

Priority order, by cost:

| corpus | access | why it matters |
|---|---|---|
| **MDVR-KCL** | free download | one reading per participant, so no repeated-measure leakage; recorded in a single setting |
| **Italian, other tasks** | already local | sustained vowels are less contaminated (0.874 for channel statistics vs 0.897); worth characterising properly |
| **NeuroVoz** | DUA | 108 participants, Spanish, documented acquisition protocol |
| **EWA-DB** | DUA | Slovak, multiple conditions, largest of the four |

MDVR-KCL is the one to do next. It is free, it is small, and its single-reading
design means a recording-level split cannot leak — so it isolates the acquisition
question from the leakage question.

## 2. If MDVR-KCL is also confounded

Then the conclusion is about the field's data, not about one corpus, and the
write-up becomes a methods paper: *silence-only classification as a mandatory
negative control for clinical speech corpora*. That is a more useful contribution
than another classifier.

## 3. If MDVR-KCL is clean

Then a genuine estimate is possible, and the machinery in `src/` is already built
for it. What would need adding, none of which was worth doing while the corpus was
confounded:

- **Calibration**, not just discrimination: reliability curves, Brier score with
  Murphy decomposition, expected calibration error.
- **Prior correction** for the case-control design. Parkinson's prevalence above
  60 is 1–2%, while these corpora are near 50/50. Corrected log-odds:
  `logit_corrected = logit_raw + log(π_real/(1−π_real)) − log(π_train/(1−π_train))`.
- **Positive predictive value at realistic prevalence.** At 90% sensitivity and 90%
  specificity, PPV at 1% prevalence is 8.3%. Any screening claim must state this.
- **Decision curve analysis.** Net benefit `NB = TP/N − (FP/N)·(p_t/(1−p_t))`
  against treat-all and treat-none, over a range of threshold probabilities.
- **Nested cross-validation** if any hyperparameter is tuned: outer 5 folds to
  measure, inner 3 to tune, scoring on `neg_log_loss` or `neg_brier_score` — not
  accuracy or F1.
- **Frozen self-supervised embeddings** (wav2vec2, WavLM) as a stronger feature
  set. Deliberately omitted from Phase 0: adding representation power to a
  confounded corpus only produces a more confident wrong answer.

Do **not** use SMOTE or any resampling. van den Goorbergh et al. (2022, *JAMIA*)
show it drives the calibration intercept from 0.06 to −1.32/−1.50 with no AUROC
gain; the same sensitivity/specificity trade-off is available by moving the
threshold.

## 4. Reporting

If a performance claim is ever made, it should be written against:

- **TRIPOD+AI** (Collins et al., *BMJ* 2024) — 27 items for prediction model
  reporting.
- **PROBAST+AI** — 16 signalling questions on risk of bias.
- **STARD 2015** if framed as a diagnostic accuracy study.
- A **model card** and a **datasheet** for the corpus.

The leakage taxonomy in Kapoor & Narayanan (2023, *Patterns*) — eight types, with a
21-question model info sheet — is the checklist that would have caught this
project's original design.

## 5. Statistical power

At 25 versus 22 participants, the Hanley–McNeil interval around an AUROC of 0.85
is roughly ±0.11. Any corpus of this size can only detect large effects, and
should report the interval rather than the point estimate. A replication aiming to
distinguish AUROC 0.75 from 0.85 needs on the order of 100 participants per group.

## Data use agreements

Neither DUA was submitted. Both require an institutional affiliation and email.

- **NeuroVoz** — <https://github.com/BYO-UPM/Neurovoz_Database>. Restriction 4 of
  the agreement forbids redistributing the materials to third parties, which means
  the corpus cannot be committed to any repository, public or private.
- **EWA-DB** — separate agreement, same constraint.

Turnaround is typically two to six weeks and depends on a human reading the
request, so submit both before starting work that depends on either.
