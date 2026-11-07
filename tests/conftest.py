from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pd_model.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic records with the raw schema; dti is exactly loan / income, as in the real data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "client_history_in_bank": rng.integers(1, 60, n),
        "age": rng.uniform(21, 75, n),
        "income": np.minimum(rng.uniform(5_000, 40_000, n), 30_000),
        "employment_years": rng.uniform(0, 20, n),
        "loan_amount": rng.uniform(1_000, 40_000, n),
        "interest_rate": rng.uniform(0.03, 0.24, n),
        "num_loans": rng.integers(0, 8, n),
        "credit_utilization": rng.uniform(0, 1, n),
        "past_due_30": rng.integers(0, 3, n),
        "past_due_60": rng.integers(0, 2, n),
        "past_due_90": rng.integers(0, 2, n),
        "past_default": rng.integers(0, 2, n),
        "region_risk": rng.integers(0, 3, n),
    })
    df["dti"] = df.loan_amount / df.income
    logit = -2 + 1.2 * df.past_default + 0.5 * df.past_due_90 + 0.8 * df.dti + 0.3 * df.region_risk
    df["default"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()
