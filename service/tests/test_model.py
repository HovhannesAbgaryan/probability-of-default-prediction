from __future__ import annotations

import math

import pytest

from app.ml.artifact import load_artifact
from app.ml.model import LogisticPDModel
from tests.conftest import ARTIFACT, VALID


@pytest.fixture(scope="module")
def model() -> LogisticPDModel:
    return LogisticPDModel(load_artifact(ARTIFACT))


def pd_of(model, **overrides) -> float:
    rec = {k: v for k, v in VALID.items() if k != "application_id"} | overrides
    return float(model.predict_pd(model.transform([rec]))[0])


def test_score_scaling(model):
    # odds 50:1 (good:bad) -> logit = -ln(50) -> base score 600; doubling the odds adds 20 points
    assert model.score(-math.log(50)) == 600
    assert model.score(-math.log(100)) == 620


@pytest.mark.parametrize("pd,grade", [(0.01, "A"), (0.05, "A"), (0.0501, "B"), (0.12, "C"), (0.3, "F"), (0.99, "G")])
def test_master_scale(model, pd, grade):
    assert model.grade(pd) == grade


@pytest.mark.parametrize("field,low,high", [
    ("past_default", 0, 1), ("past_due_30", 0, 1), ("past_due_60", 0, 1), ("past_due_90", 0, 1),
    ("credit_utilization", 0.1, 0.9), ("loan_amount", 5_000, 20_000), ("region_risk", 0, 1), ("region_risk", 1, 2),
])
def test_every_driver_increases_risk(model, field, low, high):
    assert pd_of(model, **{field: high}) > pd_of(model, **{field: low})


def test_unused_fields_do_not_change_pd(model):
    assert pd_of(model, age=25, num_loans=7) == pytest.approx(pd_of(model))


def test_region_contributions_are_aggregated(model):
    Xp = model.transform([{k: v for k, v in VALID.items() if k != "application_id"}])
    contrib = model.contributions(Xp.iloc[0])
    assert "region_risk" in contrib and not any(k.startswith("region_risk_") for k in contrib)
    # contributions explain the logit relative to the average applicant
    base = model.artifact.model[-1].intercept_[0] + sum(
        model._coef[c] * model._means[c] for c in model.card.model_features)
    assert base + sum(contrib.values()) == pytest.approx(float(model.logit(Xp)[0]))
