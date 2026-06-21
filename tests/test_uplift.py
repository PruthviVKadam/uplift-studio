"""Tests for the uplift pipeline.

One module-scoped fit of the (best) S-learner backs every assertion: predict shape, that our
metric wrappers equal sklift's own functions (correctness), targeting-policy sanity, and a
reproducibility pin on the real Qini AUC from eval/results.md.
"""
import numpy as np
import pytest
from sklift.metrics import qini_auc_score, uplift_auc_score

import data as data_mod
import metrics as M
from models import make_models


@pytest.fixture(scope="module")
def fitted():
    d = data_mod.load(target_col="visit", seed=42)
    model = make_models(42)["S-learner"]
    model.fit(d.X_train, d.y_train, d.t_train)
    return d, model.predict(d.X_test)


def test_predict_one_finite_score_per_row(fitted):
    d, up = fitted
    assert len(up) == len(d.y_test)
    assert np.all(np.isfinite(up))


def test_metric_wrappers_match_sklift(fitted):
    # The correctness test: our metrics.score must equal sklift's own functions exactly.
    d, up = fitted
    s = M.score(d.y_test, up, d.t_test)
    assert s["qini_auc"] == pytest.approx(qini_auc_score(d.y_test, up, d.t_test))
    assert s["uplift_auc"] == pytest.approx(uplift_auc_score(d.y_test, up, d.t_test))


def test_ate_equals_manual(fitted):
    d, _ = fitted
    y, t = np.asarray(d.y_test), np.asarray(d.t_test)
    assert M.ate(d.y_test, d.t_test) == pytest.approx(y[t == 1].mean() - y[t == 0].mean())


def test_targeting_policy_sanity(fitted):
    d, up = fitted
    p = M.targeting_policy(d.y_test, up, d.t_test, k=0.3)
    assert p["frac_captured_random"] == 0.3                      # random captures exactly k
    assert 0.0 <= p["incremental_at_k"] <= p["total_incremental"]  # never more than the whole
    assert p["lift_vs_random"] == pytest.approx(p["frac_captured_model"] / 0.3, rel=1e-6)


def test_golden_qini_reproduces(fitted):
    # Pins the S-learner pipeline to the published number in eval/results.md.
    d, up = fitted
    assert M.score(d.y_test, up, d.t_test)["qini_auc"] == pytest.approx(0.01601, abs=1e-3)
