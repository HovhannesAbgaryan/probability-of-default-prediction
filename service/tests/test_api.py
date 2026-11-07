from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings


def test_liveness_and_readiness(client):
    assert client.get("/health/live").json() == {"status": "ok", "model_version": None}
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["model_version"] == "1.1.0"


def test_predict_happy_path(client, valid_payload):
    r = client.post("/v1/predict", json=valid_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["application_id"] == "T-1"
    assert 0 < body["probability_of_default"] < 1
    assert body["risk_grade"] in list("ABCDEFG")
    assert body["model_version"] == "1.1.0"
    assert body["warnings"] == []
    assert all(rc["impact"] > 0 for rc in body["reason_codes"])
    assert r.headers["x-request-id"]


def test_riskier_applicant_gets_higher_pd_and_lower_score(client, valid_payload):
    safe = client.post("/v1/predict", json=valid_payload).json()
    risky = client.post("/v1/predict", json={**valid_payload, "past_default": 1, "past_due_90": 2}).json()
    assert risky["probability_of_default"] > safe["probability_of_default"]
    assert risky["score"] < safe["score"]
    assert risky["reason_codes"][0]["code"] in {"PAST_DEFAULT", "DELINQUENCY_90_DPD"}


def test_leaky_field_is_rejected(client, valid_payload):
    r = client.post("/v1/predict", json={**valid_payload, "recovery_flag_after_90d": 1})
    assert r.status_code == 422
    err = r.json()
    assert err["error"]["code"] == "validation_error"
    assert err["error"]["details"][0]["type"] == "extra_forbidden"
    assert err["request_id"]


@pytest.mark.parametrize("field,value", [
    ("loan_amount", 80_001), ("loan_amount", 0), ("income", -1), ("past_default", 2),
    ("region_risk", 3), ("past_due_30", -1), ("credit_utilization", 1.5),
])
def test_out_of_range_inputs_rejected(client, valid_payload, field, value):
    assert client.post("/v1/predict", json={**valid_payload, field: value}).status_code == 422


def test_missing_required_field(client, valid_payload):
    valid_payload.pop("income")
    assert client.post("/v1/predict", json=valid_payload).status_code == 422


def test_preprocessing_warnings(client, valid_payload):
    body = client.post("/v1/predict", json={**valid_payload, "income": 1000, "loan_amount": 50_000,
                                            "credit_utilization": None}).json()
    assert set(body["warnings"]) == {"DTI_WINSORIZED", "CREDIT_UTILIZATION_IMPUTED"}
    body = client.post("/v1/predict", json={**valid_payload, "income": 90_000}).json()
    assert body["warnings"] == ["INCOME_CAPPED"]


def test_minimal_payload_is_imputed(client, valid_payload):
    """Optional fields absent and credit_utilization null: the preprocessor imputes everything."""
    payload = {k: v for k, v in valid_payload.items() if k != "credit_utilization"}
    body = client.post("/v1/predict", json=payload).json()
    assert 0 < body["probability_of_default"] < 1
    assert body["warnings"] == ["CREDIT_UTILIZATION_IMPUTED"]  # unused optional fields don't warn


def test_optional_fields_accepted_and_validated(client, valid_payload):
    full = {**valid_payload, "age": 40, "employment_years": 5, "num_loans": 2, "interest_rate": 0.12,
            "client_history_in_bank": 24}
    assert client.post("/v1/predict", json=full).status_code == 200
    assert client.post("/v1/predict", json={**valid_payload, "interest_rate": 5}).status_code == 422


def test_batch(client, valid_payload):
    apps = [{**valid_payload, "application_id": f"A{i}", "past_due_60": i % 3} for i in range(5)]
    r = client.post("/v1/predict/batch", json={"applications": apps})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["application_id"] for p in preds] == [f"A{i}" for i in range(5)]
    single = client.post("/v1/predict", json=apps[2]).json()
    assert preds[2]["probability_of_default"] == single["probability_of_default"]


def test_batch_limits(valid_payload):
    with TestClient(create_app(make_settings(max_batch_size=2))) as c:
        r = c.post("/v1/predict/batch", json={"applications": [valid_payload] * 3})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        assert c.post("/v1/predict/batch", json={"applications": []}).status_code == 422


def test_api_key(valid_payload):
    with TestClient(create_app(make_settings(api_key="s3cret"))) as c:
        assert c.post("/v1/predict", json=valid_payload).status_code == 401
        assert c.post("/v1/predict", json=valid_payload, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/v1/predict", json=valid_payload, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health/ready").status_code == 200  # probes stay open


def test_request_id_propagated(client):
    r = client.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = client.get("/health/live", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["x-request-id"] != "bad id with spaces"


def test_model_info(client):
    body = client.get("/v1/model").json()
    assert body["model_version"] == "1.1.0" and len(body["artifact_sha256"]) == 64
    assert len(body["features"]) == 7 and body["validation"]["test_gini"] > 0.4
    assert {"region_risk_1", "region_risk_2"} <= set(body["model_inputs"])


def test_metrics_exposed(client, valid_payload):
    client.post("/v1/predict", json=valid_payload)
    text = client.get("/metrics").text
    assert "pd_predictions_total" in text and 'route="/v1/predict"' in text


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def _bad_card(tmp_path, mutate) -> Path:
    src = make_settings().model_path
    card = json.loads(src.read_text())
    for f in card["files"].values():
        shutil.copy2(src.parent / f["file"], tmp_path / f["file"])
    mutate(card)
    p = tmp_path / "model_card.json"
    p.write_text(json.dumps(card))
    return p


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c["feature_means_train"].popitem(), "feature_means_train"),
    (lambda c: c["files"]["model"].update(sha256="0" * 64), "sha256"),
    (lambda c: c["versions"].update({"scikit-learn": "0.0.1"}), "scikit-learn"),
    (lambda c: c.update(model_features=c["model_features"][:-1],
                        feature_means_train=dict(list(c["feature_means_train"].items())[:-1])), "model inputs"),
])
def test_startup_fails_on_invalid_artifact(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        with TestClient(create_app(make_settings(model_path=_bad_card(tmp_path, mutate)))):
            pass
