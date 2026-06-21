"""Uplift evaluation: ranking metrics + a targeting-policy simulation.

Why not accuracy: you never observe the counterfactual per user, so classification accuracy is
meaningless for uplift. Instead we rank users by *predicted* uplift and measure the *actual*
treated-minus-control response along that ranking — that is exactly what Qini/AUUC and the uplift
curve do.
"""
from __future__ import annotations

import numpy as np
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score, uplift_curve


def ate(y_true, treatment) -> float:
    """Average treatment effect: response rate (treated) - response rate (control)."""
    y = np.asarray(y_true)
    t = np.asarray(treatment)
    return float(y[t == 1].mean() - y[t == 0].mean())


def score(y_true, uplift, treatment) -> dict:
    """Core uplift ranking metrics for a model's predicted uplift."""
    return {
        "qini_auc": float(qini_auc_score(y_true, uplift, treatment)),
        "uplift_auc": float(uplift_auc_score(y_true, uplift, treatment)),
        "uplift_at_10": float(uplift_at_k(y_true, uplift, treatment, strategy="overall", k=0.1)),
        "uplift_at_30": float(uplift_at_k(y_true, uplift, treatment, strategy="overall", k=0.3)),
    }


def targeting_policy(y_true, uplift, treatment, k: float = 0.3) -> dict:
    """Simulate targeting the top-k fraction by predicted uplift.

    Uses the uplift curve (cumulative incremental positive outcomes as we target users in
    descending predicted-uplift order). Compares the share of *total* incremental response
    captured by targeting the top-k vs. a random policy (which captures exactly k).
    """
    x, y = uplift_curve(y_true, uplift, treatment)  # x: # targeted, y: cum. incremental outcomes
    total_incremental = float(y[-1])
    n = x[-1]
    idx = int(np.searchsorted(x, k * n))
    idx = min(idx, len(y) - 1)
    captured = float(y[idx])
    frac_captured = captured / total_incremental if total_incremental else float("nan")
    return {
        "k": k,
        "total_incremental": total_incremental,
        "incremental_at_k": captured,
        "frac_captured_model": frac_captured,   # model targeting top-k
        "frac_captured_random": k,              # random targeting top-k
        "lift_vs_random": (frac_captured / k) if total_incremental else float("nan"),
    }
