from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import SERVICE_ROOT, Settings
from app.main import create_app

ARTIFACT = SERVICE_ROOT / "artifacts" / "model_card.json"

VALID = {
    "application_id": "T-1",
    "loan_amount": 12000,
    "income": 24000,
    "past_default": 0,
    "past_due_30": 1,
    "past_due_60": 0,
    "past_due_90": 0,
    "region_risk": 1,
    "credit_utilization": 0.35,
}


def make_settings(**overrides) -> Settings:
    return Settings(**({"model_path": ARTIFACT, "log_json": False} | overrides), _env_file=None)


@pytest.fixture
def client():
    with TestClient(create_app(make_settings())) as c:  # context manager runs the lifespan (model load)
        yield c


@pytest.fixture
def valid_payload() -> dict:
    return dict(VALID)
