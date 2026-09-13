"""Clinical calibration, prevalence prior-correction, and decision curve analysis.

Essential clinical metrics:
1. Reliability curves and Expected Calibration Error (ECE).
2. Brier Score with Murphy decomposition (Reliability, Resolution, Uncertainty).
3. Prior correction: Adjusting case-control log-odds to realistic population prevalence (1-2%).
4. Positive Predictive Value (PPV) under realistic prevalence.
5. Decision Curve Analysis (DCA): Net benefit vs Treat-All and Treat-None.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve_and_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    """Compute calibration curve and Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    prob_true = []
    prob_pred = []
    counts = []
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            prob_true.append(acc)
            prob_pred.append(conf)
            counts.append(count)
            ece += (count / n_total) * abs(acc - conf)
        else:
            prob_true.append(np.nan)
            prob_pred.append(float((bins[i] + bins[i + 1]) / 2.0))
            counts.append(0)

    return {
        "prob_true": np.array(prob_true),
        "prob_pred": np.array(prob_pred),
        "counts": np.array(counts),
        "ece": float(ece),
        "bins": bins,
    }


def brier_score_murphy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Brier score and its Murphy decomposition.

    Brier = Reliability - Resolution + Uncertainty
    where lower reliability is better (0 is perfect), higher resolution is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    brier = float(np.mean((y_prob - y_true) ** 2))

    y_bar = float(np.mean(y_true))
    uncertainty = float(y_bar * (1.0 - y_bar))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    brier_binned = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        nk = np.sum(mask)
        if nk > 0:
            yk_bar = float(np.mean(y_true[mask]))
            pk_bar = float(np.mean(y_prob[mask]))
            reliability += (nk / n) * ((pk_bar - yk_bar) ** 2)
            resolution += (nk / n) * ((yk_bar - y_bar) ** 2)
            brier_binned += (nk / n) * float(np.mean((pk_bar - y_true[mask]) ** 2))

    return {
        "brier": brier,
        "brier_binned": float(brier_binned),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
    }


def prior_correction(
    y_prob: np.ndarray,
    pi_train: float,
    pi_real: float = 0.015,
    eps: float = 1e-12,
) -> np.ndarray:
    """Correct predicted probabilities from case-control prevalence to real prevalence.

    logit_corrected = logit_raw + log(pi_real / (1 - pi_real)) - log(pi_train / (1 - pi_train))
    """
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    logit_raw = np.log(p / (1.0 - p))
    log_odds_real = np.log(pi_real / (1.0 - pi_real))
    log_odds_train = np.log(pi_train / (1.0 - pi_train))

    logit_corrected = logit_raw + log_odds_real - log_odds_train
    return 1.0 / (1.0 + np.exp(-logit_corrected))


def positive_predictive_value(
    sensitivity: float,
    specificity: float,
    prevalence: float = 0.015,
) -> float:
    """Calculate Positive Predictive Value (PPV) under realistic population prevalence."""
    tp = sensitivity * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate Net Benefit across decision threshold probabilities (Vickers & Elkin, 2006).

    Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    prevalence = np.mean(y_true)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for pt in thresholds:
        w = pt / (1.0 - pt)
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        net_benefit_model = (tp / n) - (fp / n) * w
        net_benefit_all = prevalence - (1.0 - prevalence) * w
        net_benefit_none = 0.0

        rows.append(
            {
                "threshold": float(pt),
                "net_benefit_model": float(net_benefit_model),
                "net_benefit_all": float(net_benefit_all),
                "net_benefit_none": float(net_benefit_none),
            }
        )

    return pd.DataFrame(rows)
