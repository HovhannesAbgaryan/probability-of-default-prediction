"""PD -> score points and rating grade.

score = offset − factor × logit(PD), with factor = PDO / ln 2 and offset = base_score − factor × ln(base_odds):
``base_score`` points at ``base_odds`` (good:bad), +``pdo`` points every time the odds double.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Scorecard:
    pdo: float = 20
    base_score: float = 600
    base_odds: float = 50

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def score(self, logit: float) -> int:
        return round(self.offset - self.factor * logit)

    def points_per_unit(self, coef: float) -> float:
        return -self.factor * coef


def grade(pd_value: float, master_scale: Sequence[dict]) -> str:
    """First grade whose upper PD bound is >= pd_value."""
    for g in master_scale:
        if pd_value <= g["pd_upper"]:
            return g["grade"]
    return master_scale[-1]["grade"]
