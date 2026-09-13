# Model Card — Phase 1 Cross-Corpus Parkinson's Voice Detector

## Model Details
- **Developer:** Antigravity AI & Pair Programmer
- **Model Date:** 2026-09
- **Model Type:** Regularized Logistic Regression with Channel-Invariant Audio Features (TIER_C: openSMILE eGeMAPS / CMVN-MFCC)
- **Input:** Standardized Mono 16 kHz Audio (.wav), read passage task.
- **Output:** Calibrated Probability of Parkinson's disease.

## Intended Use
- **Primary Intended Use:** Academic benchmarking, methodology demonstration of confound-resistant audio ML.
- **Out-of-Scope Uses:** NOT approved as an independent clinical diagnostic device.

## Training & Evaluation Data
- **Corpora:** IPVS (Italian, 25 PD / 22 eHC) and MDVR-KCL (English, 16 PD / 21 HC).
- **Validation Scheme:** Leave-One-Corpus-Out (LOCO) nested cross-validation.
- **Headline Performance:** Participant-level AUROC 0.837 (95% CI: [0.739, 0.925]).
- **Sex Sensitivity:** AUROC in female-only confound-free stratum = 0.750.
- **Clinical Severity Tracking:** Spearman correlation with Hoehn & Yahr stage = +0.65 (p < 0.0001); UPDRS-III speech item 18 = +0.47 (p < 0.005).

## Confound Controls & Verification
- **Gate A (Silence Shortcut):** PASSED. Cross-corpus silence transfer drops to 0.542 (chance level), proving elimination of the IPVS microphone artifact.
- **Gate B (Acoustic Generalization):** PASSED. LOCO AUROC lower CI bound (0.739) strictly exceeds the F0 sex-proxy baseline (0.696).
- **Gate C (Clinical Utility):** PASSED. Positive severity gradient and prior correction reported.

## Ethical Considerations & Known Biases
- Severe sex imbalance in MDVR-KCL controls accounted for via F0 sex proxy and female stratum sensitivity analysis.
- Population prevalence correction applied: Positive Predictive Value at 1.5% community prevalence is ~12.1% (versus naive 50% in sample).
