# PD Scoring Service

FastAPI service that serves the probability-of-default model validated in `../notebooks`
(7-variable logistic regression with `region_risk` one-hot encoded, test Gini 0.418 / AUC 0.709, status *conditionally approved*, see notebook 03).

## Architecture

```
HTTP ─► api/        routes, request/response schemas, auth, error envelope   (no business logic)
        │
        ▼
      services/     PDService: raw application → PD, score, grade, reason codes, input warnings
        │
        ▼
      ml/           model card (schema-validated) + the fitted preprocessor and model (sha256-verified joblib)
```

- **The validated objects, not a re-implementation.** `scripts/export_model.py` (repository root) copies the
  fitted `preprocessor.joblib` and `model.joblib` from `models/` into `artifacts/` and writes `model_card.json`
  (version, features, score scaling, master scale, validation results, sha256 of each file, scikit-learn version).
- **Safe loading.** joblib is pickle-based, so the service checks each file's sha256 against the card *before*
  unpickling, refuses a different scikit-learn version than the one used in training, and checks that the model's
  inputs match the card. Any mismatch stops startup.
- **Same features online and offline, by construction.** The request goes through the same preprocessor used in
  training and validation: income capped at 30k, `dti = loan_amount / income`, `dti` winsorised at the train p99.5,
  **median / most-frequent imputation of any missing field**, `region_risk` one-hot encoded, scaling.
  `tests/test_parity.py` checks the service against the validated model on all 6,000 hold-out rows.
- **Depends on the `pd-model` package** (repository root), which provides the custom transformers needed to
  unpickle the preprocessor and pins scikit-learn/pandas/numpy/joblib to the training versions. LightGBM, XGBoost
  and statsmodels are not needed at runtime: only the selected logistic model is shipped.
- **Data contract.** Unknown fields are rejected (`extra="forbid"`), so the leaky `recovery_flag_after_90d`
  can never reach the model. Range checks cover the block-2 error types found in the EDA.
  Loans above the 80k product cap are rejected.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/predict` | Score one application |
| POST | `/v1/predict/batch` | Score up to `PD_MAX_BATCH_SIZE` applications |
| GET | `/v1/model` | Model version, artifact sha256, features, validation metadata |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (503 until the model is loaded) |
| GET | `/metrics` | Prometheus metrics (requests, latency, PD distribution, grades, input warnings) |
| GET | `/docs` | OpenAPI UI (disable with `PD_DOCS_ENABLED=false`) |

```bash
curl -X POST localhost:8000/v1/predict -H 'Content-Type: application/json' -d '{
  "application_id": "APP-1", "loan_amount": 12000, "income": 24000, "past_default": 0,
  "past_due_30": 1, "past_due_60": 0, "past_due_90": 0, "region_risk": 1, "credit_utilization": 0.35}'
```
```json
{"application_id": "APP-1", "probability_of_default": 0.140317, "score": 539, "risk_grade": "D",
 "reason_codes": [{"code": "DELINQUENCY_30_DPD", "description": "History of 30+ days past due", "impact": 0.2316},
                  {"code": "REGION_RISK", "description": "Higher-risk region", "impact": 0.1921},
                  {"code": "HIGH_CREDIT_UTILIZATION", "description": "High credit utilisation", "impact": 0.0069}],
 "warnings": [], "model_version": "1.1.0"}
```

- Required fields: `loan_amount`, `income`, `past_default`, `past_due_30/60/90`, `region_risk`.
  `credit_utilization` and the fields the current model doesn't use (`age`, `employment_years`, `num_loans`,
  `interest_rate`, `client_history_in_bank`) are optional and imputed when missing.

- `score`: 600 points at 50:1 good:bad odds, 20 points to double the odds. Higher means safer.
- `risk_grade`: master scale A–G (PD ≤ 5%, 8%, 12%, 18%, 25%, 40%, 100%).
- `reason_codes`: the features raising the PD most compared with an average applicant (usable for adverse-action notices). The two region dummies are reported together as `REGION_RISK`.
- `warnings`: inputs adjusted during preprocessing (`INCOME_CAPPED`, `DTI_WINSORIZED`, `<FIELD>_IMPUTED` for a missing field the model uses).
  These are also counted in `/metrics` for monitoring.
- Errors always use the same shape: `{"error": {"code", "message", "details"}, "request_id"}`.
  Every response carries an `X-Request-ID` header: the caller's value is propagated, otherwise one is generated.

## Run

```bash
# from service/ (installs ../ i.e. the pd-model package, editable)
pip install -r requirements-dev.txt
uvicorn app.main:app --reload                 # local
pytest                                        # 44 tests (parity test needs the training artifacts)

# the image needs the pd-model package, so build from the REPOSITORY ROOT
cd .. && docker build -f service/Dockerfile -t pd-scoring-service:1.1.0 .
docker run -p 8000:8000 -e PD_API_KEY=change-me pd-scoring-service:1.1.0
```

Configuration comes from `PD_*` environment variables (see `.env.example`).

## Releasing a new model version

1. From the repository root: `scripts/clean_data.py`, `scripts/train.py`, `scripts/validate.py` (or notebooks 02–03),
   and get validation sign-off.
2. `python scripts/export_model.py --version 1.2.0`.
3. `pytest` (parity must pass), commit the new artifacts and card, build and deploy the image.
   `/v1/model` and every response report the version. The sha256 appears in the startup log.

## Operational notes

- One uvicorn process per container. Scale with replicas, which also keeps the Prometheus counters correct.
- The artifact is validated at startup. A bad file makes the pod fail to start instead of serving wrong PDs.
- Monitor `pd_predicted_probability` (PSI vs. the validation distribution), `pd_predictions_total{grade}` and
  `pd_input_warnings_total`, as the validation report requires (PSI alert at > 0.10).
- The validation conditions (OOT test, long-run recalibration, confirmed definitions) still apply before
  this model is used for real credit decisions.
