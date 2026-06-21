"""Uplift meta-learners behind one interface.

All three turn an ordinary classifier into an individual-treatment-effect estimator:
  - S-learner: one model with treatment as a feature (sklift SoloModel)
  - T-learner: separate treated/control models, uplift = P_t - P_c (sklift TwoModels, vanilla)
  - T-learner (DDR): dependent-data-representation variant that corrects for control leakage

`.fit(X, y, t)` then `.predict(X)` -> per-row uplift score. RandomForest base keeps it CPU-only
and dependency-light (no torch, no causalml).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklift.models import SoloModel, TwoModels

N_ESTIMATORS = 200
MAX_DEPTH = 8


def _base(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=seed, n_jobs=-1
    )


def make_models(seed: int = 42) -> dict:
    """Return {name: fresh uplift model}. Fresh estimators per call (no shared state)."""
    return {
        "S-learner": SoloModel(estimator=_base(seed)),
        "T-learner": TwoModels(
            estimator_trmnt=_base(seed), estimator_ctrl=_base(seed), method="vanilla"
        ),
        "T-learner (DDR)": TwoModels(
            estimator_trmnt=_base(seed), estimator_ctrl=_base(seed), method="ddr_control"
        ),
    }


MODEL_NAMES = tuple(make_models().keys())
