from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pd_model.features.preprocessing import (
    build_preprocessor_from_config,
    output_sources,
    select_outputs,
)


@pytest.fixture
def fitted(cfg, frame):
    return build_preprocessor_from_config(cfg).fit(frame[cfg.raw_features], frame["default"])


def test_output_has_no_missing_and_stable_names(cfg, frame, fitted):
    out = fitted.transform(frame[cfg.raw_features])
    assert out.notna().all().all()
    assert list(out.columns) == list(fitted.get_feature_names_out())
    assert {"region_risk_1", "region_risk_2"} <= set(out.columns) and "region_risk" not in out.columns


@pytest.mark.parametrize("column", ["client_history_in_bank", "age", "income", "employment_years", "interest_rate",
                                    "num_loans", "credit_utilization", "past_due_30", "past_due_60", "past_due_90",
                                    "past_default", "region_risk"])
def test_missing_value_in_any_feature_is_imputed(cfg, frame, fitted, column):
    """No column had NaNs at fit time; a NaN in any of them at scoring time must still be handled."""
    row = frame[cfg.raw_features].head(1).copy()
    row[column] = np.nan
    out = fitted.transform(row)
    assert out.notna().all().all()


def test_absent_columns_and_placeholder(cfg, frame, fitted):
    out = fitted.transform(pd.DataFrame([{"loan_amount": 12_000, "income": 24_000, "age": 99}]))
    assert out.notna().all().all()
    median_age_scaled = fitted.transform(frame[cfg.raw_features].assign(age=np.nan).head(1))["age"].iloc[0]
    assert out["age"].iloc[0] == pytest.approx(median_age_scaled)     # 99 is treated as unknown
    assert out["dti"].iloc[0] == pytest.approx(fitted.transform(       # dti derived from loan / income
        frame[cfg.raw_features].head(1).assign(loan_amount=12_000, income=24_000, dti=0.5))["dti"].iloc[0])


def test_unseen_region_maps_to_reference_level(cfg, frame, fitted):
    row = frame[cfg.raw_features].head(1).assign(region_risk=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = fitted.transform(row)
    assert out[["region_risk_1", "region_risk_2"]].iloc[0].tolist() == [0.0, 0.0]


def test_income_capped_and_winsorised(cfg, frame, fitted):
    raw = frame[cfg.raw_features].head(1).assign(income=90_000, loan_amount=1e6, dti=np.nan)
    feats = fitted[:-1].transform(raw)  # before imputation/scaling
    assert feats["income"].iloc[0] == 30_000 and feats["income_capped"].iloc[0] == 1
    assert feats["dti"].iloc[0] == pytest.approx(fitted.named_steps["winsor"].caps_["dti"])


def test_learned_only_from_fit_data(cfg, frame):
    """Refitting on different data changes the learned medians: nothing is hard-coded."""
    a = build_preprocessor_from_config(cfg).fit(frame[cfg.raw_features])
    b = clone(a).fit(frame[cfg.raw_features].assign(income=frame.income / 2))
    def med(p):
        return p.named_steps["columns"].named_transformers_["num"].named_steps["impute"].statistics_[2]

    assert med(b) == pytest.approx(med(a) / 2)


def test_output_sources(cfg, fitted):
    names = list(fitted.get_feature_names_out())
    src = output_sources(names, cfg.categorical)
    assert src["region_risk_2"] == "region_risk" and src["income_capped"] == "income_capped"
    assert select_outputs(names, ["region_risk", "dti"], cfg.categorical) == ["dti", "region_risk_1", "region_risk_2"]
