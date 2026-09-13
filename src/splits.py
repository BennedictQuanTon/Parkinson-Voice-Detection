"""Cross-validation schemes for the leakage experiment.

The whole point of the study is that these three schemes give very different
numbers on identical features and an identical estimator, so they live in one
place and are covered by ``tests/test_no_leakage.py``.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

#: ``arm -> (metadata column used as the grouping key, human-readable label)``
SCHEMES: dict[str, tuple[str | None, str]] = {
    "A_recording": (
        None,
        "Recording-level StratifiedKFold (no grouping)",
    ),
    "B_folder_path": (
        "folder_path",
        "Grouped by directory path (splits two-session participants)",
    ),
    "C_subject_key": (
        "subject_id",
        "Grouped by verified participant key",
    ),
    "D_leave_one_corpus_out": (
        "corpus",
        "Leave-One-Corpus-Out (train on one corpus, test on the other)",
    ),
}


def get_groups(meta: pd.DataFrame, arm: str) -> np.ndarray | None:
    if arm not in SCHEMES:
        raise KeyError(f"unknown arm {arm!r}; expected one of {sorted(SCHEMES)}")
    column, _ = SCHEMES[arm]
    if column is None:
        return None
    if column not in meta.columns:
        raise KeyError(f"metadata is missing the {column!r} column needed by {arm!r}")
    return meta[column].to_numpy()


def iter_splits(
    meta: pd.DataFrame,
    y: np.ndarray,
    arm: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat_index, train_idx, test_idx)`` for the requested arm."""
    from sklearn.model_selection import LeaveOneGroupOut

    groups = get_groups(meta, arm)
    X_dummy = np.zeros((len(y), 1))
    
    if arm == "D_leave_one_corpus_out":
        logo = LeaveOneGroupOut()
        for repeat in range(n_repeats):
            for train_idx, test_idx in logo.split(X_dummy, y, groups=groups):
                yield repeat, train_idx, test_idx
        return

    for repeat in range(n_repeats):
        rs = seed + repeat
        if groups is None:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
            folds = splitter.split(X_dummy, y)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=rs
            )
            folds = splitter.split(X_dummy, y, groups=groups)
        for train_idx, test_idx in folds:
            yield repeat, train_idx, test_idx


def leaking_participants(
    meta: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> set[str]:
    """Participants (by verified key) present on both sides of a split."""
    train = set(meta.iloc[train_idx]["subject_id"])
    test = set(meta.iloc[test_idx]["subject_id"])
    return train & test


def leakage_report(
    meta: pd.DataFrame,
    y: np.ndarray,
    arm: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-fold count of participants that appear in both train and test."""
    rows = []
    for fold, (repeat, train_idx, test_idx) in enumerate(
        iter_splits(meta, y, arm, n_splits, n_repeats, seed)
    ):
        leaked = leaking_participants(meta, train_idx, test_idx)
        rows.append(
            {
                "arm": arm,
                "repeat": repeat,
                "fold": fold % n_splits,
                "n_test_recordings": len(test_idx),
                "n_leaked_participants": len(leaked),
                "n_leaked_recordings": int(
                    meta.iloc[test_idx]["subject_id"].isin(leaked).sum()
                ),
            }
        )
    return pd.DataFrame(rows)
