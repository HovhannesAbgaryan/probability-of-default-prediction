"""Model artifact: a model card (JSON) plus the fitted preprocessor and model (joblib).

Loading fails fast on anything inconsistent: unknown card schema, a joblib whose sha256 differs from
the card (joblib is pickle-based, so only hash-verified files are unpickled), a different scikit-learn
version than the one used for training, or a model whose inputs differ from the card."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class ScoreScaling(BaseModel):
    pdo: float = Field(gt=0)
    base_score: float
    base_odds: float = Field(gt=0)


class Grade(BaseModel):
    grade: str
    pd_upper: float = Field(gt=0, le=1)


class ArtifactFile(BaseModel):
    file: str
    sha256: str = Field(min_length=64, max_length=64)


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    schema_version: int
    model_name: str
    model_version: str
    model_type: str
    input_features: list[str]
    model_features: list[str]
    model_source_features: list[str]
    feature_means_train: dict[str, float]
    income_cap: float = Field(gt=0)
    dti_winsor_cap: float = Field(gt=0)
    score_scaling: ScoreScaling
    master_scale: list[Grade]
    files: dict[str, ArtifactFile]
    versions: dict[str, str]
    validation: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "ModelCard":
        if self.schema_version != 2:
            raise ValueError(f"unsupported model card schema_version {self.schema_version}")
        if self.model_type != "logistic_regression":
            raise ValueError(f"unsupported model_type {self.model_type}")
        if set(self.files) != {"preprocessor", "model"}:
            raise ValueError("model card must list exactly the 'preprocessor' and 'model' files")
        if set(self.feature_means_train) != set(self.model_features):
            raise ValueError("feature_means_train must cover exactly the model features")
        uppers = [g.pd_upper for g in self.master_scale]
        if uppers != sorted(uppers) or uppers[-1] != 1.0:
            raise ValueError("master_scale must be increasing and end at pd_upper = 1.0")
        return self


@dataclass(frozen=True)
class ModelArtifact:
    card: ModelCard
    preprocessor: Pipeline
    model: Pipeline
    sha256: str  # of the model card, which pins the two joblib hashes

    @property
    def model_version(self) -> str:
        return self.card.model_version


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifact(card_path: Path) -> ModelArtifact:
    card_path = Path(card_path)
    raw = card_path.read_bytes()
    card = ModelCard.model_validate_json(raw)

    trained_with, installed = card.versions.get("scikit-learn"), version("scikit-learn")
    if trained_with != installed:
        raise ValueError(f"artifact trained with scikit-learn {trained_with}, but {installed} is installed")

    loaded = {}
    for key, f in card.files.items():
        path = card_path.parent / f.file
        if _sha256(path) != f.sha256:
            raise ValueError(f"sha256 of {path.name} does not match the model card")
        loaded[key] = joblib.load(path)  # safe: content verified against the card above

    preprocessor, model = loaded["preprocessor"], loaded["model"]
    if not (isinstance(model, Pipeline) and isinstance(model[-1], LogisticRegression)):
        raise ValueError("model must be a Pipeline ending in LogisticRegression")
    if list(model.named_steps["cols"].cols) != card.model_features:
        raise ValueError("model inputs differ from model_features in the card")
    if not set(card.model_features) <= set(preprocessor.get_feature_names_out()):
        raise ValueError("preprocessor does not produce every model feature")
    return ModelArtifact(card=card, preprocessor=preprocessor, model=model, sha256=hashlib.sha256(raw).hexdigest())
