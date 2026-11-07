"""Project configuration, loaded from ``configs/config.yaml``.

The path can be overridden with the ``PD_MODEL_CONFIG`` environment variable or an explicit argument."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    raw_data: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    service_artifacts_dir: Path

    @property
    def train(self) -> Path:
        return self.processed_dir / "train.parquet"

    @property
    def test(self) -> Path:
        return self.processed_dir / "test.parquet"

    @property
    def cleaning_report(self) -> Path:
        return self.processed_dir / "cleaning_report.json"

    @property
    def preprocessor(self) -> Path:
        return self.models_dir / "preprocessor.joblib"

    @property
    def model(self) -> Path:
        return self.models_dir / "model.joblib"

    @property
    def candidates(self) -> Path:
        return self.models_dir / "candidates.joblib"

    @property
    def oof_predictions(self) -> Path:
        return self.models_dir / "oof_predictions.parquet"

    @property
    def metadata(self) -> Path:
        return self.models_dir / "metadata.json"

    @property
    def validation_dir(self) -> Path:
        return self.reports_dir / "validation"

    @property
    def training_dir(self) -> Path:
        return self.reports_dir / "training"


@dataclass(frozen=True)
class Config:
    root: Path
    seed: int
    target: str
    paths: Paths
    cleaning: dict[str, Any]
    preprocessing: dict[str, Any]
    training: dict[str, Any]
    scorecard: dict[str, Any]
    validation: dict[str, Any]

    @property
    def numeric(self) -> list[str]:
        return list(self.preprocessing["numeric"])

    @property
    def categorical(self) -> list[str]:
        return list(self.preprocessing["categorical"])

    @property
    def raw_features(self) -> list[str]:
        """Input columns the preprocessor expects (engineered columns are built inside it)."""
        from pd_model.features.transformers import ENGINEERED  # single source of the engineered names

        return [c for c in self.numeric if c not in ENGINEERED] + self.categorical


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("PD_MODEL_CONFIG") or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text())
    root = ROOT
    paths = Paths(**{k: root / v for k, v in raw["paths"].items()})
    return Config(root=root, seed=int(raw["seed"]), target=raw["target"], paths=paths,
                  cleaning=raw["cleaning"], preprocessing=raw["preprocessing"], training=raw["training"],
                  scorecard=raw["scorecard"], validation=raw["validation"])
