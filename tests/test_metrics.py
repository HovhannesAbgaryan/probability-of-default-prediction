from __future__ import annotations

import numpy as np
import pytest

from pd_model.models.scorecard import Scorecard, grade
from pd_model.validation.metrics import gini, ks_stat, psi, wilson_ci


def test_gini_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert gini(y, [0.1, 0.2, 0.8, 0.9]) == 1 and ks_stat(y, [0.1, 0.2, 0.8, 0.9]) == 1
    assert gini(y, [0.5] * 4) == 0


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) == pytest.approx(0, abs=1e-9)


def test_wilson_contains_rate():
    lo, hi = wilson_ci(15, 100)
    assert lo < 0.15 < hi and 0 <= lo and hi <= 1


def test_scorecard():
    s = Scorecard(20, 600, 50)
    assert s.score(-np.log(50)) == 600 and s.score(-np.log(100)) == 620


@pytest.mark.parametrize("p,g", [(0.01, "A"), (0.05, "A"), (0.0501, "B"), (0.3, "F"), (0.99, "G")])
def test_grade(cfg, p, g):
    assert grade(p, cfg.scorecard["master_scale"]) == g
