from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from pd_model.features.preprocessing import build_preprocessor_from_config
from pd_model.features.transformers import Winsorizer
from pd_model.models.training import cv_scores, full_pipeline, run_training, select


def test_cv_refits_preprocessor_per_fold(cfg, frame):
    X, y = frame[cfg.raw_features], frame["default"]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    pipe = full_pipeline(build_preprocessor_from_config(cfg), LogisticRegression(max_iter=1000))
    with mock.patch.object(Winsorizer, "fit", autospec=True, side_effect=Winsorizer.fit) as fit:
        cv_scores(pipe, X, y, folds)
    sizes = [len(call.args[1]) for call in fit.call_args_list]
    assert sizes == [len(tr) for tr, _ in folds]  # fitted on each training fold only, never on all rows


def test_select_prefers_simplest_equivalent():
    comp = pd.DataFrame({"CV Gini": [0.441, 0.440, 0.420]}, index=["A_logreg_all", "B_logreg_parsimonious", "C"])
    sel, chal = select(comp, {"A_logreg_all": 3, "B_logreg_parsimonious": 1, "C": 2}, 0.01)
    assert sel == "B_logreg_parsimonious" and chal == "D_lgbm_monotone"


def test_run_training_smoke(cfg, frame):
    res = run_training(frame, cfg, tune_boosting=False)
    assert res.selected in res.models and len(res.models) == 8
    p = res.predict_proba(frame[cfg.raw_features].head(5).assign(age=np.nan, region_risk=np.nan))
    assert ((p > 0) & (p < 1)).all()


def test_artifacts_and_training_report_roundtrip(cfg, frame, tmp_path):
    """save_artifacts writes everything the notebooks and the service read; load_artifacts verifies hashes."""
    import dataclasses

    from pd_model.models.training import load_artifacts, load_training_report, save_artifacts

    paths = dataclasses.replace(cfg.paths, models_dir=tmp_path / "models", reports_dir=tmp_path / "reports")
    tmp_cfg = dataclasses.replace(cfg, paths=paths)
    res = run_training(frame, tmp_cfg, tune_boosting=False)
    meta = save_artifacts(res, tmp_cfg)

    preprocessor, model, loaded_meta = load_artifacts(tmp_cfg)
    assert loaded_meta["sha256"] == meta["sha256"] and loaded_meta["selected"] == res.selected
    X = frame[cfg.raw_features].head(20)
    assert np.allclose(model.predict_proba(preprocessor.transform(X))[:, 1], res.predict_proba(X))

    rep = load_training_report(paths.training_dir)
    assert list(rep["comparison"].index) == list(res.comparison.index)
    assert set(rep["cv_folds"].index.get_level_values(0)) == set(res.models)

    paths.model.write_bytes(paths.model.read_bytes() + b"tampered")
    try:
        load_artifacts(tmp_cfg)
    except ValueError as e:
        assert "sha256" in str(e)
    else:
        raise AssertionError("tampered model.joblib was loaded")
