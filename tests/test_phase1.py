"""Unit tests for Phase 1 Cross-Corpus modules.

Covers:
1. MDVR-KCL filename and clinical score parsing.
2. Arm D_leave_one_corpus_out split integrity.
3. CMVN mathematical invariants and feature shape.
4. Calibration, prior correction, and Brier Murphy decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import calibration as Cal
from src import corpora as C
from src import normalize as N
from src import splits as S


# --------------------------------------------------------------------------- #
# MDVR-KCL Filename & Score Parsing
# --------------------------------------------------------------------------- #

def test_parse_kcl_filename_pd():
    parsed = C.parse_kcl_filename("ID02_pd_1_2_1.wav")
    assert parsed is not None
    assert parsed.subject_id == "KCL_ID02"
    assert parsed.group == "PD"
    assert parsed.label == 1
    assert parsed.hy == 1.0
    assert parsed.u2 == 2.0
    assert parsed.u3 == 1.0


def test_parse_kcl_filename_hc_with_updrs():
    # Documented oddity: ID31 has HC label with UPDRS ratings of 1
    parsed = C.parse_kcl_filename("ID31_hc_0_1_1.wav")
    assert parsed is not None
    assert parsed.subject_id == "KCL_ID31"
    assert parsed.group == "HC"
    assert parsed.label == 0
    assert parsed.hy == 0.0
    assert parsed.u2 == 1.0
    assert parsed.u3 == 1.0


def test_parse_kcl_filename_invalid():
    assert C.parse_kcl_filename("invalid_filename.wav") is None


# --------------------------------------------------------------------------- #
# Split Integrity: Arm D_leave_one_corpus_out
# --------------------------------------------------------------------------- #

def _synthetic_cross_corpus_meta() -> pd.DataFrame:
    rows = []
    # 4 IPVS participants, 2 PD / 2 HC
    for i in range(4):
        sid = f"IPVS_P{i:02d}"
        for rep in (1, 2):
            rows.append(
                {
                    "corpus": "ipvs",
                    "subject_id": sid,
                    "label": int(i < 2),
                    "task": "read_passage",
                }
            )
    # 4 KCL participants, 2 PD / 2 HC
    for i in range(4):
        sid = f"KCL_ID{i:02d}"
        for rep in (1,):
            rows.append(
                {
                    "corpus": "mdvr_kcl",
                    "subject_id": sid,
                    "label": int(i < 2),
                    "task": "read_passage",
                }
            )
    return pd.DataFrame(rows)


def test_arm_d_never_shares_corpus_or_participant():
    meta = _synthetic_cross_corpus_meta()
    y = meta["label"].to_numpy()

    splits = list(S.iter_splits(meta, y, "D_leave_one_corpus_out", n_repeats=1))
    assert len(splits) == 2, "LOCO on 2 corpora must produce exactly 2 folds"

    for _, train_idx, test_idx in splits:
        train_corpora = set(meta.iloc[train_idx]["corpus"])
        test_corpora = set(meta.iloc[test_idx]["corpus"])
        assert len(train_corpora) == 1
        assert len(test_corpora) == 1
        assert not (train_corpora & test_corpora), "Corpus leaked across split boundary"
        assert not S.leaking_participants(meta, train_idx, test_idx)


def test_arm_d_covers_all_recordings():
    meta = _synthetic_cross_corpus_meta()
    y = meta["label"].to_numpy()
    seen = set()
    for _, _, test_idx in S.iter_splits(meta, y, "D_leave_one_corpus_out", n_repeats=1):
        seen.update(test_idx.tolist())
    assert seen == set(range(len(meta))), "LOCO must evaluate every recording exactly once"


# --------------------------------------------------------------------------- #
# Normalization & CMVN
# --------------------------------------------------------------------------- #

def test_cmvn_zero_mean_unit_variance():
    rng = np.random.default_rng(42)
    # 20 mfcc channels, 100 frames, with an arbitrary constant offset (channel)
    frames = rng.normal(loc=15.0, scale=3.0, size=(20, 100))
    normed = N.apply_cmvn_to_frames(frames)

    assert np.allclose(normed.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(normed.std(axis=1), 1.0, atol=1e-3)


def test_extract_cmvn_mfcc_returns_40_finite_features():
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.1, size=16000)
    feats = N.extract_cmvn_mfcc(signal, sr=16000, n_mfcc=20)
    assert len(feats) == 40
    for k, v in feats.items():
        assert np.isfinite(v), f"Non-finite feature {k}: {v}"


def test_channel_augmentation_preserves_length():
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.1, size=16000)
    aug = N.channel_augmentation(signal, sr=16000, rng=rng)
    assert len(aug) == len(signal)
    assert np.all(np.isfinite(aug))
    assert np.max(np.abs(aug)) <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# Calibration, Prior Correction & DCA
# --------------------------------------------------------------------------- #

def test_prior_correction_decreases_prob_for_rare_disease():
    # Case-control 50% down to real population prevalence 1.5%
    raw_prob = np.array([0.5, 0.8, 0.95])
    corrected = Cal.prior_correction(raw_prob, pi_train=0.5, pi_real=0.015)
    assert np.all(corrected < raw_prob)
    # A 50% case-control probability at 1.5% real prevalence should become 1.5%
    assert pytest.approx(corrected[0], rel=1e-3) == 0.015


def test_brier_murphy_decomposition():
    rng = np.random.default_rng(42)
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.7, 0.1, 0.2, 0.8, 0.3, 0.6, 0.4, 0.1, 0.85])

    res = Cal.brier_score_murphy(y_true, y_prob, n_bins=5)
    # Brier (binned) = Reliability - Resolution + Uncertainty
    decomposed = res["reliability"] - res["resolution"] + res["uncertainty"]
    assert pytest.approx(res["brier_binned"], abs=1e-6) == decomposed


def test_dca_net_benefit_ordering():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.95, 0.9, 0.05, 0.1, 0.85, 0.1, 0.8, 0.15, 0.05, 0.9])
    dca = Cal.decision_curve_analysis(y_true, y_prob, thresholds=np.array([0.1, 0.3, 0.5]))
    # For a high accuracy model, model net benefit should be >= 0
    assert (dca["net_benefit_model"] >= 0).all()
