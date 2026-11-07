"""Business logic: raw application -> saved preprocessor -> PD, score, grade, reason codes.

All feature preprocessing lives in the fitted preprocessor shipped with the model, so online and
offline features are identical by construction. This module only reports which inputs were adjusted."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.metrics import INPUT_WARNINGS, PD_VALUE, PREDICTIONS
from app.ml.model import LogisticPDModel
from app.schemas.prediction import LoanApplication

REASONS = {
    "past_default": ("PAST_DEFAULT", "Previous default on record"),
    "past_due_90": ("DELINQUENCY_90_DPD", "History of 90+ days past due"),
    "past_due_60": ("DELINQUENCY_60_DPD", "History of 60+ days past due"),
    "past_due_30": ("DELINQUENCY_30_DPD", "History of 30+ days past due"),
    "dti": ("HIGH_LOAN_TO_INCOME", "Loan amount high relative to income"),
    "region_risk": ("REGION_RISK", "Higher-risk region"),
    "credit_utilization": ("HIGH_CREDIT_UTILIZATION", "High credit utilisation"),
}


@dataclass(frozen=True)
class Reason:
    code: str
    description: str
    impact: float


@dataclass(frozen=True)
class ScoringResult:
    application_id: str | None
    probability_of_default: float
    score: int
    risk_grade: str
    reason_codes: list[Reason]
    warnings: list[str] = field(default_factory=list)
    model_version: str = ""


class PDService:
    def __init__(self, model: LogisticPDModel, reason_codes_top_n: int = 3):
        self.model = model
        self.top_n = reason_codes_top_n
        self._card = model.card
        # Report imputation only for request fields the model uses; others don't affect the PD.
        # (dti is not a request field: the preprocessor derives it from loan_amount / income.)
        self._model_inputs = set(self._card.model_source_features) & set(LoanApplication.model_fields)

    @property
    def model_version(self) -> str:
        return self.model.version

    def record(self, app: LoanApplication) -> dict:
        return app.model_dump(exclude={"application_id"})

    def input_warnings(self, rec: dict) -> list[str]:
        warnings = [f"{f.upper()}_IMPUTED" for f in sorted(self._model_inputs) if rec.get(f) is None]
        income = rec["income"]
        if income > self._card.income_cap:
            warnings.append("INCOME_CAPPED")  # training data is censored at the cap
        if rec["loan_amount"] / min(income, self._card.income_cap) > self._card.dti_winsor_cap:
            warnings.append("DTI_WINSORIZED")
        return warnings

    def score_batch(self, apps: list[LoanApplication]) -> list[ScoringResult]:
        records = [self.record(a) for a in apps]
        Xp = self.model.transform(records)
        logits, pds = self.model.logit(Xp), self.model.predict_pd(Xp)
        results = []
        for i, app in enumerate(apps):
            pd_value = float(pds[i])
            grade = self.model.grade(pd_value)
            contrib = sorted(((f, c) for f, c in self.model.contributions(Xp.iloc[i]).items() if c > 0),
                             key=lambda fc: fc[1], reverse=True)[: self.top_n]
            reasons = [Reason(*REASONS[f], impact=round(c, 4)) for f, c in contrib]
            warnings = self.input_warnings(records[i])

            PREDICTIONS.labels(self.model_version, grade).inc()
            PD_VALUE.labels(self.model_version).observe(pd_value)
            for w in warnings:
                INPUT_WARNINGS.labels(w).inc()
            results.append(ScoringResult(application_id=app.application_id,
                                         probability_of_default=round(pd_value, 6),
                                         score=self.model.score(float(logits[i])), risk_grade=grade,
                                         reason_codes=reasons, warnings=warnings, model_version=self.model_version))
        return results

    def score(self, app: LoanApplication) -> ScoringResult:
        return self.score_batch([app])[0]
