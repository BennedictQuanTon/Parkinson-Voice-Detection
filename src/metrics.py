"""Evaluation at the level the clinical question is asked: the participant.

Two rules are enforced here rather than left to the caller, because getting
either wrong silently inflates the reported interval:

1. Participant-level scores aggregate the predicted probabilities of a
   participant's recordings before scoring, they do not average per-recording
   scores.
2. Bootstrap resampling draws participants, not recordings. Resampling
   recordings treats a participant's repeated readings as independent samples
   and produces intervals that are far too narrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def aggregate_to_participant(
    subject_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"subject_id": subject_ids, "y_true": y_true, "y_score": y_score}
    )
    out = frame.groupby("subject_id").agg(
        y_true=("y_true", "max"), y_score=("y_score", "mean")
    )
    if frame.groupby("subject_id")["y_true"].nunique().gt(1).any():
        raise ValueError("a participant carries more than one label")
    return out.reset_index()


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC, or NaN when a resample happens to contain a single class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(point_estimate, lower, upper)``.

    ``y_true``/``y_score`` must already be one row per participant.
    """
    rng = np.random.default_rng(seed)
    point = safe_auroc(y_true, y_score)
    n = len(y_true)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = safe_auroc(y_true[idx], y_score[idx])
    draws = draws[~np.isnan(draws)]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def bootstrap_paired_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap for ``AUROC(a) - AUROC(b)`` on the same participants.

    Used instead of DeLong because the two arms being compared differ in how
    predictions were produced, not just in the score column, and a paired
    resample of participants makes no parametric assumption about that.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = safe_auroc(y_true[idx], score_a[idx]) - safe_auroc(
            y_true[idx], score_b[idx]
        )
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    observed = safe_auroc(y_true, score_a) - safe_auroc(y_true, score_b)
    # Two-sided bootstrap p-value: how often the resampled delta crosses zero.
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": float(observed),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(min(p, 1.0)),
    }


def permutation_test(
    fit_predict,
    meta: pd.DataFrame,
    y: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Null distribution of participant-level AUROC under label shuffling.

    Labels are shuffled *at participant level*, so the null preserves the fact
    that a participant's recordings share a label. ``fit_predict`` must accept
    ``(meta, y)`` and return the observed participant-level ``(y_true, y_score)``.
    """
    rng = np.random.default_rng(seed)
    y_true_obs, y_score_obs = fit_predict(meta, y)
    observed = safe_auroc(y_true_obs, y_score_obs)

    participants = meta[["subject_id"]].drop_duplicates()
    participants = participants.merge(
        pd.DataFrame({"subject_id": meta["subject_id"], "label": y})
        .drop_duplicates("subject_id"),
        on="subject_id",
    )

    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = participants.copy()
        shuffled["label"] = rng.permutation(shuffled["label"].to_numpy())
        y_perm = (
            meta[["subject_id"]]
            .merge(shuffled, on="subject_id", how="left")["label"]
            .to_numpy()
        )
        yt, ys = fit_predict(meta, y_perm)
        null[i] = safe_auroc(yt, ys)

    null = null[~np.isnan(null)]
    # +1 correction so the p-value can never be exactly zero.
    p = (np.sum(null >= observed) + 1) / (len(null) + 1)
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float(p),
        "null": null,
    }
