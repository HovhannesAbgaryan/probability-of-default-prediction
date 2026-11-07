"""Scoring endpoints. HTTP concerns only; all logic lives in PDService."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from app.api.deps import ServiceDep, SettingsDep
from app.schemas.prediction import BatchRequest, BatchResponse, LoanApplication, ModelInfo, Prediction

router = APIRouter(prefix="/v1", tags=["scoring"])


@router.post("/predict", response_model=Prediction, summary="Score one loan application")
def predict(application: LoanApplication, service: ServiceDep) -> Prediction:
    return Prediction(**asdict(service.score(application)))


@router.post("/predict/batch", response_model=BatchResponse, summary="Score a batch of applications")
def predict_batch(body: BatchRequest, service: ServiceDep, settings: SettingsDep) -> BatchResponse:
    if len(body.applications) > settings.max_batch_size:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"Batch size {len(body.applications)} exceeds limit {settings.max_batch_size}")
    results = service.score_batch(body.applications)
    return BatchResponse(model_version=service.model_version,
                         predictions=[Prediction(**asdict(r)) for r in results])


@router.get("/model", response_model=ModelInfo, summary="Metadata of the loaded model")
def model_info(service: ServiceDep) -> ModelInfo:
    artifact = service.model.artifact
    c = artifact.card
    return ModelInfo(model_name=c.model_name, model_version=c.model_version, model_type=c.model_type,
                     artifact_sha256=artifact.sha256, features=c.model_source_features,
                     model_inputs=c.model_features, master_scale=[g.model_dump() for g in c.master_scale],
                     validation=c.validation)
