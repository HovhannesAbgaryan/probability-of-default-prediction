"""Custom sklearn transformers used inside the saved preprocessor and the candidate models.

They live in this importable module (not in a notebook or script) so pickled artifacts can be loaded
anywhere ``pd_model`` is installed, including the scoring service."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pd_model.features.woe import bin_edges

ENGINEERED = ["income_capped", "total_past_due", "any_past_due"]
DPD = ["past_due_30", "past_due_60", "past_due_90"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless domain rules applied to raw application records.

    - every expected input column exists (absent -> NaN, so the imputers handle it)
    - ``age == age_placeholder`` (99 = unknown) -> NaN
    - ``income`` censored at ``income_cap`` (the training data is censored there)
    - ``dti`` = loan_amount / income when not supplied (it is exactly that in the data)
    - engineered: ``income_capped``, ``total_past_due``, ``any_past_due``
    - categorical columns cast to strings ("0", "1", ...), so codes are labels, not numbers
    """

    def __init__(self, columns: Sequence[str], categorical: Sequence[str] = (),
                 age_placeholder: float = 99, income_cap: float = 30_000):
        self.columns = columns
        self.categorical = categorical
        self.age_placeholder = age_placeholder
        self.income_cap = income_cap

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for c in self.columns:
            out[c] = X[c] if c in X.columns else np.nan
        num = [c for c in self.columns if c not in self.categorical]
        out[num] = out[num].apply(pd.to_numeric, errors="coerce").astype(float)

        if "age" in out:
            out["age"] = out["age"].mask(out["age"] == self.age_placeholder)
        if "income" in out:
            out["income"] = out["income"].clip(upper=self.income_cap)
        if {"dti", "loan_amount", "income"} <= set(out.columns):
            out["dti"] = out["dti"].fillna(out["loan_amount"] / out["income"])

        income = out.get("income")
        out["income_capped"] = (income >= self.income_cap - 1e-6).astype(float).where(income.notna())
        total = out[DPD].sum(axis=1, min_count=len(DPD))  # NaN if any count is missing
        out["total_past_due"] = total
        out["any_past_due"] = (total > 0).astype(float).where(total.notna())

        for c in self.categorical:
            out[c] = out[c].map(lambda v: str(int(float(v))) if pd.notna(v) else np.nan).astype(object)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.columns) + ENGINEERED, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps columns at an upper quantile learned on the training data."""

    def __init__(self, quantiles: dict[str, float]):
        self.quantiles = quantiles

    def fit(self, X: pd.DataFrame, y=None):
        self.caps_ = {c: float(X[c].quantile(q)) for c, q in self.quantiles.items()}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "caps_")
        X = X.copy()
        for c, cap in self.caps_.items():
            X[c] = X[c].clip(upper=cap)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class Cols(BaseEstimator, TransformerMixin):
    """Select a fixed list of columns, so every candidate model can take the same input frame."""

    def __init__(self, cols: Sequence[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.cols)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.cols, dtype=object)


class WoETransformer(BaseEstimator, TransformerMixin):
    """Quantile-binned WoE encoder, fitted on training data only. Missing values get their own bin."""

    def __init__(self, max_bins: int = 6, eps: float = 0.5, min_iv: float = 0.0):
        self.max_bins = max_bins
        self.eps = eps
        self.min_iv = min_iv

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.edges_, self.maps_, self.missing_woe_, self.iv_ = {}, {}, {}, {}
        good, bad = (y == 0).sum(), (y == 1).sum()
        for c in X.columns:
            edges = bin_edges(X[c], self.max_bins)
            self.edges_[c] = edges
            codes = pd.cut(X[c], edges, labels=False)
            woe = {}
            for k in range(len(edges) - 1):
                m = codes == k
                g, bd = ((y[m] == 0).sum(), (y[m] == 1).sum())
                woe[k] = np.log(((g + self.eps) / good) / ((bd + self.eps) / bad))
            self.maps_[c] = woe
            pg = np.array([((codes == k) & (y == 0)).sum() + self.eps for k in woe]) / good
            pb = np.array([((codes == k) & (y == 1)).sum() + self.eps for k in woe]) / bad
            self.iv_[c] = float(np.sum((pg - pb) * np.log(pg / pb)))
            m = X[c].isna()
            g, bd = ((y[m] == 0).sum(), (y[m] == 1).sum())
            self.missing_woe_[c] = np.log(((g + self.eps) / good) / ((bd + self.eps) / bad)) if m.any() else 0.0
        self.feature_names_in_ = np.array([c for c in X.columns if self.iv_[c] >= self.min_iv])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.feature_names_in_:
            codes = pd.cut(X[c], self.edges_[c], labels=False)
            out[c] = codes.map(self.maps_[c]).fillna(self.missing_woe_[c]).astype(float)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
