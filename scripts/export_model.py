"""Export the validated model to the scoring service.

    python scripts/export_model.py [--version 1.1.0] [--config configs/config.yaml]

Copies models/preprocessor.joblib + models/model.joblib into service/artifacts/ and writes
model_card.json next to them. The card records version, features, scorecard, validation results,
library versions and the sha256 of each joblib; the service refuses to load files that do not match.
Run scripts/validate.py first: the card embeds reports/validation/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from importlib.metadata import version
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pd_model.config import load_config
from pd_model.data.loading import load_split
from pd_model.models.training import load_artifacts, model_input_columns

CARD_SCHEMA_VERSION = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(model_version: str, config: str | None = None) -> Path:
    cfg = load_config(config)
    preprocessor, model, meta = load_artifacts(cfg)  # hash-checked against metadata.json
    if not (isinstance(model, Pipeline) and isinstance(model[-1], LogisticRegression)):
        raise SystemExit(f"selected model {meta['selected']} is not a logistic regression; "
                         "the service's score and reason codes need one")
    summary_path = cfg.paths.validation_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} missing: run scripts/validate.py first")
    summary = json.loads(summary_path.read_text())

    train, _ = load_split(cfg.paths.processed_dir)
    cols = model_input_columns(model)
    means = preprocessor.transform(train[cfg.raw_features])[cols].mean()

    out = cfg.paths.service_artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, src in (("preprocessor", cfg.paths.preprocessor), ("model", cfg.paths.model)):
        dst = out / src.name
        shutil.copy2(src, dst)
        files[key] = {"file": dst.name, "sha256": sha256(dst)}

    s = summary
    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "model_name": "retail_pd_application",
        "model_version": model_version,
        "model_type": "logistic_regression",
        "source_candidate": meta["selected"],
        "created": date.today().isoformat(),
        "target": cfg.target,
        "input_features": meta["input_features"],
        "model_features": cols,
        "model_source_features": meta["model_source_features"],
        "feature_means_train": {c: float(means[c]) for c in cols},
        "income_cap": float(cfg.preprocessing["income_cap"]),
        "dti_winsor_cap": float(meta["preprocessor_params"]["winsor_caps"]["dti"]),
        "score_scaling": {k: cfg.scorecard[k] for k in ("pdo", "base_score", "base_odds")},
        "master_scale": cfg.scorecard["master_scale"],
        "files": files,
        "versions": {"scikit-learn": version("scikit-learn"), "pd-model": version("pd-model")},
        "validation": {
            "test_gini": round(s["discrimination"]["gini"], 4),
            "test_auc": round(s["discrimination"]["auc"], 4),
            "test_ks": round(s["discrimination"]["ks"], 4),
            "test_mean_pd": round(s["calibration"]["mean_pd"], 4),
            "test_observed_dr": round(s["calibration"]["observed_dr"], 4),
            "hosmer_lemeshow_p": round(s["calibration"]["hosmer_lemeshow_p"], 4),
            "status": "conditionally_approved",
            "conditions": [
                "confirm default / DPD definitions with data owner",
                "production feed must not contain recovery_flag_after_90d",
                "out-of-time validation on latest vintage",
                "recalibrate intercept to long-run default rate",
            ],
        },
    }
    path = out / "model_card.json"
    path.write_text(json.dumps(card, indent=2) + "\n")
    print(f"wrote {path} ({meta['selected']}, test Gini {card['validation']['test_gini']}, "
          f"AUC {card['validation']['test_auc']})")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="1.1.0")
    ap.add_argument("--config")
    a = ap.parse_args()
    main(a.version, a.config)
