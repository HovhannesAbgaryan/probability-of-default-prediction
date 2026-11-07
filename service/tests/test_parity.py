"""The service must reproduce the validated model exactly on the hold-out set.

Needs the training environment (pd-model, pyarrow) and the training artifacts; skipped otherwise."""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
config = pytest.importorskip("pd_model.config")

from app.ml.artifact import load_artifact  # noqa: E402
from app.ml.model import LogisticPDModel  # noqa: E402
from app.schemas.prediction import LoanApplication  # noqa: E402
from app.services.pd_service import PDService  # noqa: E402
from tests.conftest import ARTIFACT  # noqa: E402

cfg = config.load_config()
pytestmark = pytest.mark.skipif(not (cfg.paths.metadata.exists() and cfg.paths.test.exists()),
                                reason="training artifacts missing")
FIELDS = ["loan_amount", "income", "past_default", "past_due_30", "past_due_60", "past_due_90", "region_risk",
          "credit_utilization", "age", "employment_years", "num_loans", "interest_rate", "client_history_in_bank"]
INTS = {"past_default", "past_due_30", "past_due_60", "past_due_90", "region_risk", "num_loans",
        "client_history_in_bank"}


def test_service_matches_validated_model():
    from pd_model.models.training import load_artifacts

    preprocessor, model, _ = load_artifacts(cfg)
    test = pd.read_parquet(cfg.paths.test)
    expected = model.predict_proba(preprocessor.transform(test[cfg.raw_features]))[:, 1]

    # The service derives dti from loan / income itself (the identity holds in the cleaned data).
    apps = [LoanApplication(**{k: (None if pd.isna(v) else int(v) if k in INTS else float(v))
                               for k, v in row.items()}) for row in test[FIELDS].to_dict("records")]
    got = [r.probability_of_default for r in PDService(LogisticPDModel(load_artifact(ARTIFACT))).score_batch(apps)]
    max_diff = max(abs(g - e) for g, e in zip(got, expected, strict=True))
    assert max_diff < 1e-6, f"max |service − validated| = {max_diff}"   # service rounds PDs to 6 decimals
