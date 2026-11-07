"""Candidate tuning, cross-validated comparison, selection and artifact saving.

Cross-validation refits the *whole* ``preprocessor + model`` pipeline inside every fold, so imputation
medians, winsor caps and scaling are never learned on the validation fold. Only the final artifacts
are saved separately: one fitted preprocessor and the fitted selected model."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline

from pd_model.features.preprocessing import (
    build_preprocessor_from_config,
    learned_parameters,
    output_sources,
)
from pd_model.models.candidates import GRID_KEYS, build_candidates
from pd_model.validation.metrics import auc, gini, summary_metrics

log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    preprocessor: Pipeline                      # fitted on the full training set
    models: dict                                # candidate -> fitted on the preprocessed training set
    selected: str
    challenger: str
    comparison: pd.DataFrame                    # CV summary per candidate
    paired: pd.DataFrame                        # paired fold tests vs the parsimonious model
    cv_folds: dict[str, pd.DataFrame]           # per-fold metrics per candidate
    oof: pd.DataFrame                           # out-of-fold PDs per candidate + target
    best_params: dict = field(default_factory=dict)
    tuning: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def model(self):
        return self.models[self.selected]

    def predict_proba(self, X_raw: pd.DataFrame, name: str | None = None) -> np.ndarray:
        return self.models[name or self.selected].predict_proba(self.preprocessor.transform(X_raw))[:, 1]


def full_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline([("prep", clone(preprocessor)), ("model", clone(model))])


def cv_scores(pipeline, X: pd.DataFrame, y: pd.Series, folds) -> tuple[pd.DataFrame, np.ndarray]:
    rows, oof = [], np.zeros(len(y))
    for tr, va in folds:
        m = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        p_va = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p_va
        p_tr = m.predict_proba(X.iloc[tr])[:, 1]
        rows.append({**summary_metrics(y.iloc[va], p_va),
                     "Gini_train": gini(y.iloc[tr], p_tr), "AUC_train": auc(y.iloc[tr], p_tr)})
    return pd.DataFrame(rows), oof


def tune(candidates: dict, preprocessor, X, y, folds, grids: dict) -> tuple[dict, dict]:
    """Small grid per tree-based candidate; every point scored on the same folds. Mutates `candidates`."""
    best, tables = {}, {}
    for name, grid in grids.items():
        points, res = list(ParameterGrid(grid)), []
        for g in points:
            s, _ = cv_scores(full_pipeline(preprocessor, clone(candidates[name]).set_params(**g)), X, y, folds)
            res.append({**g, "cv_gini": s.Gini.mean(), "cv_auc": s.AUC.mean(), "cv_gini_std": s.Gini.std(),
                        "train_gini": s.Gini_train.mean(), "train_auc": s.AUC_train.mean()})
        res = pd.DataFrame(res).sort_values("cv_gini", ascending=False)
        best[name] = points[res.index[0]]  # original types kept (e.g. max_depth=None)
        candidates[name].set_params(**best[name])
        tables[name] = res
        log.info("%s best %s", name, best[name])
    return best, tables


def compare(cv_results: dict[str, pd.DataFrame], reference: str = "B_logreg_parsimonious"):
    comp = pd.DataFrame({
        n: {"CV Gini": s.Gini.mean(), "CV AUC": s.AUC.mean(), "CV Gini std": s.Gini.std(), "CV AUC std": s.AUC.std(),
            "CV KS": s.KS.mean(), "CV Brier": s.Brier.mean(),
            "Train Gini": s.Gini_train.mean(), "Train AUC": s.AUC_train.mean(),
            "Overfit gap (Gini)": s.Gini_train.mean() - s.Gini.mean(),
            "Overfit gap (AUC)": s.AUC_train.mean() - s.AUC.mean()}
        for n, s in cv_results.items()}).T.sort_values("CV Gini", ascending=False)
    ref = cv_results[reference].Gini
    paired = pd.DataFrame({
        n: {"mean Δ Gini vs ref": (s.Gini - ref).mean(),
            "mean Δ AUC vs ref": (s.AUC - cv_results[reference].AUC).mean(),
            "folds better than ref": int((s.Gini > ref).sum()),
            "paired t p-value": stats.ttest_rel(s.Gini, ref).pvalue}
        for n, s in cv_results.items() if n != reference}).T
    return comp, paired


def select(comp: pd.DataFrame, complexity: dict[str, int], equivalence: float) -> tuple[str, str]:
    """Best CV Gini, but among models within `equivalence` of the best the simplest one wins.

    Gini = 2·AUC − 1, so this is the same rule as "within equivalence / 2 AUC of the best"."""
    best = comp["CV Gini"].max()
    equivalent = comp.index[comp["CV Gini"] >= best - equivalence].tolist()
    selected = min(equivalent, key=complexity.get)
    challenger = "D_lgbm_monotone" if selected != "D_lgbm_monotone" else "B_logreg_parsimonious"
    return selected, challenger


def run_training(train: pd.DataFrame, cfg, tune_boosting: bool = True) -> TrainingResult:
    t = cfg.training
    X, y = train[cfg.raw_features], train[cfg.target]
    folds = list(StratifiedKFold(n_splits=t["cv_folds"], shuffle=True, random_state=cfg.seed).split(X, y))

    # Fitted once only to fix the output column names (candidate column lists, monotone vector).
    # Inside CV the preprocessor is re-fitted per fold.
    preprocessor = build_preprocessor_from_config(cfg).fit(X, y)
    names = list(preprocessor.get_feature_names_out())
    candidates = build_candidates(names, cfg)

    best_params, tuning = {}, {}
    if tune_boosting:
        grids = {n: t[k] for n, k in GRID_KEYS.items()}
        best_params, tuning = tune(candidates, preprocessor, X, y, folds, grids)

    cv_results, oof = {}, {}
    for name, model in candidates.items():
        cv_results[name], oof[name] = cv_scores(full_pipeline(preprocessor, model), X, y, folds)
        log.info("%s CV Gini %.4f | CV AUC %.4f", name, cv_results[name].Gini.mean(), cv_results[name].AUC.mean())
    comp, paired = compare(cv_results)
    selected, challenger = select(comp, t["complexity"], t["equivalence_gini"])

    Xp = preprocessor.transform(X)
    fitted = {name: clone(m).fit(Xp, y) for name, m in candidates.items()}
    oof_df = pd.DataFrame(oof, index=X.index).assign(**{cfg.target: y.values})
    return TrainingResult(preprocessor=preprocessor, models=fitted, selected=selected, challenger=challenger,
                          comparison=comp, paired=paired, cv_folds=cv_results, oof=oof_df,
                          best_params=best_params, tuning=tuning)


def model_input_columns(model) -> list[str]:
    """Preprocessed columns the model actually reads."""
    if isinstance(model, Pipeline) and "cols" in model.named_steps:
        return list(model.named_steps["cols"].cols)
    return list(model.feature_names_in_)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_training_report(result: TrainingResult, out_dir: Path) -> None:
    """CV evidence for the development notebook: comparison, paired tests, per-fold scores, tuning grids."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(out_dir / "comparison.csv")
    result.paired.to_csv(out_dir / "paired_tests.csv")
    folds = pd.concat({n: s for n, s in result.cv_folds.items()}, names=["candidate", "fold"])
    folds.to_csv(out_dir / "cv_folds.csv")
    for name, table in result.tuning.items():
        table.to_csv(out_dir / f"tuning_{name}.csv", index=False)


def load_training_report(out_dir: Path) -> dict:
    """Read what `save_training_report` wrote (the notebook never re-trains)."""
    return {
        "comparison": pd.read_csv(out_dir / "comparison.csv", index_col=0),
        "paired": pd.read_csv(out_dir / "paired_tests.csv", index_col=0),
        "cv_folds": pd.read_csv(out_dir / "cv_folds.csv", index_col=[0, 1]),
        "tuning": {f.stem.removeprefix("tuning_"): pd.read_csv(f) for f in sorted(out_dir.glob("tuning_*.csv"))},
    }


def save_artifacts(result: TrainingResult, cfg) -> dict:
    """The only writer of models/ (called by scripts/train.py). Also writes reports/training/."""
    p = cfg.paths
    p.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.preprocessor, p.preprocessor)
    joblib.dump(result.model, p.model)
    joblib.dump({"models": result.models, "selected": result.selected, "challenger": result.challenger},
                p.candidates)
    result.oof.to_parquet(p.oof_predictions)

    names = list(result.preprocessor.get_feature_names_out())
    used = model_input_columns(result.model)
    src = output_sources(names, cfg.categorical)
    metadata = {
        "selected": result.selected,
        "challenger": result.challenger,
        "input_features": cfg.raw_features,
        "preprocessed_features": names,
        "model_features": used,
        "model_source_features": sorted({src[c] for c in used}),
        "preprocessor_params": learned_parameters(result.preprocessor),
        "cv_comparison": json.loads(result.comparison.to_json(orient="index")),
        "best_params": result.best_params,
        "versions": {pkg: version(pkg) for pkg in ("pd-model", "scikit-learn", "pandas", "numpy", "joblib")},
        "sha256": {"preprocessor": _sha256(p.preprocessor), "model": _sha256(p.model)},
    }
    p.metadata.write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    save_training_report(result, p.training_dir)
    return metadata


def load_artifacts(cfg) -> tuple[Pipeline, object, dict]:
    """Load the fitted preprocessor + selected model, verifying their hashes against metadata.json."""
    p = cfg.paths
    meta = json.loads(p.metadata.read_text())
    for key, path in (("preprocessor", p.preprocessor), ("model", p.model)):
        if _sha256(path) != meta["sha256"][key]:
            raise ValueError(f"{path} does not match the sha256 recorded in {p.metadata}")
    return joblib.load(p.preprocessor), joblib.load(p.model), meta
