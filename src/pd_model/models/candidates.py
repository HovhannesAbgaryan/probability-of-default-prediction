"""Candidate PD models. Every candidate consumes the *preprocessed* frame (output of the preprocessor)."""
from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pd_model.features.preprocessing import (
    MISSING_PREFIX,
    output_sources,
    select_outputs,
)
from pd_model.features.transformers import Cols, WoETransformer


def monotone_vector(feature_names_out, monotone: dict[str, int], categorical: list[str]) -> list[int]:
    """Monotone constraint per preprocessed column, from the constraint of its source feature.

    One-hot dummies inherit the sign of their source (all levels are riskier than the lowest-risk
    reference level); missing-value indicators are unconstrained."""
    src = output_sources(feature_names_out, categorical)
    return [0 if n.startswith(MISSING_PREFIX) else int(monotone.get(src[n], 0)) for n in feature_names_out]


def build_candidates(feature_names_out, cfg) -> dict:
    t, seed = cfg.training, cfg.seed
    names = list(feature_names_out)
    parsimonious = select_outputs(names, t["parsimonious"], cfg.categorical)
    src = output_sources(names, cfg.categorical)
    woe_cols = [n for n in names if src[n] not in t["woe_exclude"]]
    mono = monotone_vector(names, t["monotone"], cfg.categorical)

    lgb_base = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=seed, verbose=-1)
    xgb_base = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=seed, n_jobs=4, eval_metric="logloss")
    return {
        "A_logreg_all": Pipeline([("cols", Cols(names)), ("lr", LogisticRegression(max_iter=3000))]),
        "B_logreg_parsimonious": Pipeline([("cols", Cols(parsimonious)),
                                           ("lr", LogisticRegression(C=1e6, max_iter=3000))]),  # ≈ unpenalised
        "C_woe_scorecard": Pipeline([("cols", Cols(woe_cols)), ("woe", WoETransformer(max_bins=8, min_iv=0.02)),
                                     ("lr", LogisticRegression(max_iter=3000))]),
        "D_lgbm_monotone": lgb.LGBMClassifier(**lgb_base, monotone_constraints=mono),
        "E_lgbm_free": lgb.LGBMClassifier(**lgb_base),
        "F_xgb_monotone": xgb.XGBClassifier(**xgb_base, monotone_constraints="(" + ",".join(map(str, mono)) + ")"),
        "G_xgb_free": xgb.XGBClassifier(**xgb_base),
        # Bagging benchmark. Unconstrained: monotone constraints cost ~0.01 Gini for a forest in testing.
        "H_random_forest": RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1,
                                                  random_state=seed),
    }


GRID_KEYS = {"D_lgbm_monotone": "lgbm_grid", "E_lgbm_free": "lgbm_grid",
             "F_xgb_monotone": "xgb_grid", "G_xgb_free": "xgb_grid", "H_random_forest": "rf_grid"}
