"""Load the Hillstrom e-mail dataset and prepare it for uplift modeling.

Treatment = an e-mail was sent (Mens or Womens) vs. the No-E-Mail control.
No timestamp exists, so we use a fixed-seed split **stratified on treatment** to keep the
treatment/control balance identical in train and test (a leakage-free, reproducible holdout).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom

VALID_TARGETS = ("visit", "conversion", "spend")


@dataclass
class UpliftData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    t_train: pd.Series
    t_test: pd.Series
    feature_names: list[str]
    target_col: str


def load(target_col: str = "visit", test_size: float = 0.3, seed: int = 42) -> UpliftData:
    """Fetch Hillstrom, binarize treatment, one-hot encode, and split.

    `target_col` is one of {"visit", "conversion", "spend"}; we model the binary
    outcome (spend is binarized to "spent anything").
    """
    if target_col not in VALID_TARGETS:
        raise ValueError(f"target_col must be one of {VALID_TARGETS}, got {target_col!r}")

    bunch = fetch_hillstrom(target_col=target_col)
    X = pd.get_dummies(bunch.data.copy(), drop_first=True)
    y = bunch.target
    y = (y > 0).astype(int) if target_col == "spend" else y.astype(int)
    treatment = (bunch.treatment != "No E-Mail").astype(int)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, treatment, test_size=test_size, random_state=seed, stratify=treatment
    )
    return UpliftData(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        t_train=t_train, t_test=t_test,
        feature_names=list(X.columns), target_col=target_col,
    )
