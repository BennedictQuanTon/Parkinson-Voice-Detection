"""The controlled comparison: identical features, identical estimator, three
different ways of deciding which recordings may share a fold.

Everything that could differ between arms other than the split is held fixed:
the same feature matrix, the same estimator class, the same seeds, the same
number of folds and repeats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from . import splits as S


def default_estimator(seed: int = 42) -> Pipeline:
    """RandomForest with library defaults, matching the reference study.

    Scaling is inside the pipeline so it is fitted on the training fold only.
    It is a no-op for a forest but keeps the pipeline reusable for the linear
    baselines in the negative controls.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100, criterion="gini", random_state=seed, n_jobs=-1
                ),
            ),
        ]
    )


def run_arm(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    arm: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    estimator_factory=default_estimator,
) -> dict[str, object]:
    """Cross-validate one arm and return out-of-fold predictions plus scores."""
    X_values = X.to_numpy()
    n = len(y)
    oof = np.full((n_repeats, n), np.nan)
    per_fold = []

    for fold_id, (repeat, train_idx, test_idx) in enumerate(
        S.iter_splits(meta, y, arm, n_splits, n_repeats, seed)
    ):
        model = estimator_factory(seed)
        model.fit(X_values[train_idx], y[train_idx])
        proba = model.predict_proba(X_values[test_idx])[:, 1]
        oof[repeat, test_idx] = proba

        train_proba = model.predict_proba(X_values[train_idx])[:, 1]
        leaked = S.leaking_participants(meta, train_idx, test_idx)
        per_fold.append(
            {
                "arm": arm,
                "repeat": repeat,
                "fold": fold_id % n_splits,
                "auroc_test_recording": M.safe_auroc(y[test_idx], proba),
                "auroc_train_recording": M.safe_auroc(y[train_idx], train_proba),
                "n_leaked_participants": len(leaked),
            }
        )

    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())
        raise RuntimeError(f"{arm}: {missing} recordings never landed in a test fold")

    # Average the per-repeat out-of-fold scores, then score once. Each repeat is
    # a complete partition, so this is an average of unbiased predictions rather
    # than a reuse of any single fold.
    oof_mean = oof.mean(axis=0)

    recording_auroc = M.bootstrap_auroc_ci(y, oof_mean, seed=seed)
    participant = M.aggregate_to_participant(
        meta["subject_id"].to_numpy(), y, oof_mean
    )
    participant_auroc = M.bootstrap_auroc_ci(
        participant["y_true"].to_numpy(), participant["y_score"].to_numpy(), seed=seed
    )

    per_fold_df = pd.DataFrame(per_fold)
    return {
        "arm": arm,
        "label": S.SCHEMES[arm][1],
        "oof_mean": oof_mean,
        "participant": participant,
        "auroc_recording": recording_auroc,
        "auroc_participant": participant_auroc,
        "per_fold": per_fold_df,
        "mean_leaked_participants": float(
            per_fold_df["n_leaked_participants"].mean()
        ),
    }


def summarise(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for arm, res in results.items():
        rec, rec_lo, rec_hi = res["auroc_recording"]
        par, par_lo, par_hi = res["auroc_participant"]
        rows.append(
            {
                "arm": arm,
                "scheme": res["label"],
                "auroc_recording": rec,
                "recording_ci": f"[{rec_lo:.3f}, {rec_hi:.3f}]",
                "auroc_participant": par,
                "participant_ci": f"[{par_lo:.3f}, {par_hi:.3f}]",
                "leaked_participants_per_fold": res["mean_leaked_participants"],
                "train_auroc_mean": res["per_fold"]["auroc_train_recording"].mean(),
                "test_auroc_fold_sd": res["per_fold"]["auroc_test_recording"].std(),
            }
        )
    return pd.DataFrame(rows).set_index("arm")


FEATURE_KEY_COLUMNS = 13


def load_features(
    feature_dir, task: str, feature_set: str, cohort: str = "full"
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return ``(X, y, meta)`` for one cell of the experiment grid."""
    from pathlib import Path

    frame = pd.read_parquet(Path(feature_dir) / f"{task}__{feature_set}.parquet")
    if cohort == "batch_matched":
        frame = frame[frame["cohort_batch_matched"]]
    elif cohort != "full":
        raise KeyError(f"unknown cohort {cohort!r}")
    frame = frame.reset_index(drop=True)
    meta = frame.iloc[:, :FEATURE_KEY_COLUMNS]
    X = frame.iloc[:, FEATURE_KEY_COLUMNS:]
    return X, frame["label"].to_numpy(), meta


def run_grid(
    feature_dir,
    cells: list[tuple[str, str, str]],
    arms: list[str] | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Run every ``(task, feature_set, cohort)`` cell across every split arm."""
    arms = arms or list(S.SCHEMES)
    rows, store = [], {}

    for task, feature_set, cohort in cells:
        X, y, meta = load_features(feature_dir, task, feature_set, cohort)
        for arm in arms:
            result = run_arm(X, y, meta, arm, n_splits, n_repeats, seed)
            store[(task, feature_set, cohort, arm)] = result
            point, lo, hi = result["auroc_participant"]
            rec, rec_lo, rec_hi = result["auroc_recording"]
            rows.append(
                {
                    "task": task,
                    "feature_set": feature_set,
                    "cohort": cohort,
                    "arm": arm,
                    "n_recordings": len(y),
                    "n_participants": meta["subject_id"].nunique(),
                    "auroc_participant": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "auroc_recording": rec,
                    "rec_ci_lo": rec_lo,
                    "rec_ci_hi": rec_hi,
                    "leaked_per_fold": result["mean_leaked_participants"],
                }
            )
    return pd.DataFrame(rows), store


def leakage_effect(
    grid: pd.DataFrame, store: dict, reference_arm: str = "C_subject_key"
) -> pd.DataFrame:
    """Paired difference between each arm and the correct arm, same participants."""
    rows = []
    keys = {(t, f, c) for t, f, c, _ in store}
    for task, feature_set, cohort in sorted(keys):
        ref = store[(task, feature_set, cohort, reference_arm)]
        ref_part = ref["participant"].set_index("subject_id")
        for arm in S.SCHEMES:
            if arm == reference_arm:
                continue
            other = store[(task, feature_set, cohort, arm)]["participant"].set_index(
                "subject_id"
            )
            common = ref_part.index.intersection(other.index)
            stats = M.bootstrap_paired_delta(
                ref_part.loc[common, "y_true"].to_numpy(),
                other.loc[common, "y_score"].to_numpy(),
                ref_part.loc[common, "y_score"].to_numpy(),
            )
            rows.append(
                {
                    "task": task,
                    "feature_set": feature_set,
                    "cohort": cohort,
                    "arm": arm,
                    "vs": reference_arm,
                    "delta_auroc": stats["delta"],
                    "ci_lo": stats["lo"],
                    "ci_hi": stats["hi"],
                    "p": stats["p"],
                }
            )
    return pd.DataFrame(rows)


def learning_curve_by_participant(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    fractions=(0.4, 0.55, 0.7, 0.85, 1.0),
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Learning curve where the x axis counts *participants*, not recordings.

    ``sklearn.model_selection.learning_curve`` subsamples rows, which would put
    one reading of a passage in the training subset and the other in the test
    fold. Here the training subset is drawn at participant level inside an
    already participant-disjoint fold.
    """
    X_values = X.to_numpy()
    subject_ids = meta["subject_id"].to_numpy()
    rows = []

    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X_values, y, groups=subject_ids)
        ):
            train_participants = np.unique(subject_ids[train_idx])
            for frac in fractions:
                k = max(4, int(round(frac * len(train_participants))))
                chosen = rng.choice(train_participants, size=k, replace=False)
                sub = train_idx[np.isin(subject_ids[train_idx], chosen)]
                if len(np.unique(y[sub])) < 2:
                    continue
                model = default_estimator(seed)
                model.fit(X_values[sub], y[sub])
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "fraction": frac,
                        "n_train_participants": k,
                        "auroc_train": M.safe_auroc(
                            y[sub], model.predict_proba(X_values[sub])[:, 1]
                        ),
                        "auroc_test": M.safe_auroc(
                            y[test_idx], model.predict_proba(X_values[test_idx])[:, 1]
                        ),
                    }
                )
    return pd.DataFrame(rows)
