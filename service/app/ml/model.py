"""Scoring with the saved preprocessor + logistic regression.

Raw application fields go through exactly the preprocessor used in training and validation
(imputation of any missing field, income cap, dti, winsorising, one-hot region, scaling)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pd_model.features.preprocessing import output_sources
from pd_model.models.scorecard import Scorecard, grade

from app.ml.artifact import ModelArtifact


class LogisticPDModel:
    def __init__(self, artifact: ModelArtifact):
        self.artifact = artifact
        card = artifact.card
        self.card = card
        self._lr = artifact.model[-1]
        self._coef = pd.Series(self._lr.coef_[0], index=card.model_features)
        self._means = pd.Series(card.feature_means_train)[card.model_features]
        categorical = list(artifact.preprocessor.named_steps["features"].categorical)
        self._source = output_sources(card.model_features, categorical)
        s = card.score_scaling
        self._scorecard = Scorecard(s.pdo, s.base_score, s.base_odds)
        self._master_scale = [g.model_dump() for g in card.master_scale]

    @property
    def version(self) -> str:
        return self.artifact.model_version

    def transform(self, records: Sequence[Mapping]) -> pd.DataFrame:
        raw = pd.DataFrame(list(records), columns=self.card.input_features)
        return self.artifact.preprocessor.transform(raw)

    def logit(self, Xp: pd.DataFrame) -> np.ndarray:
        return self.artifact.model.decision_function(Xp)

    def predict_pd(self, Xp: pd.DataFrame) -> np.ndarray:
        return self.artifact.model.predict_proba(Xp)[:, 1]

    def score(self, logit: float) -> int:
        """Points scale: base_score at base_odds (good:bad), +pdo points per doubling of the odds."""
        return self._scorecard.score(logit)

    def grade(self, pd_value: float) -> str:
        return grade(pd_value, self._master_scale)

    def contributions(self, Xp_row: pd.Series) -> dict[str, float]:
        """Log-odds contribution per source feature, relative to the average training applicant.

        One-hot dummies are summed back to their source (region_risk_1 + region_risk_2 -> region_risk)."""
        per_column = self._coef * (Xp_row[self.card.model_features] - self._means)
        out: dict[str, float] = {}
        for col, value in per_column.items():
            out[self._source[col]] = out.get(self._source[col], 0.0) + float(value)
        return out
