"""Independent validation of the selected model on the untouched hold-out.

    python scripts/validate.py [--config configs/config.yaml]

Writes reports/validation/summary.json and one CSV per table.
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import pandas as pd

from pd_model.config import load_config
from pd_model.data.loading import load_split
from pd_model.models.training import load_artifacts
from pd_model.validation.validate import ValidationData, run_validation, save_report


def main(config: str | None = None) -> dict:
    cfg = load_config(config)
    train, test = load_split(cfg.paths.processed_dir)
    preprocessor, _, meta = load_artifacts(cfg)        # hash-checked
    bundle = joblib.load(cfg.paths.candidates)
    oof = pd.read_parquet(cfg.paths.oof_predictions)
    v = ValidationData.build(train, test, preprocessor, bundle["models"], bundle["selected"], bundle["challenger"],
                             cfg, oof)
    report = run_validation(v, cfg)
    save_report(report, cfg.paths.validation_dir)
    s = report["summary"]
    print(json.dumps({"selected": s["selected"], **s["discrimination"],
                      "mean_pd": s["calibration"]["mean_pd"], "observed_dr": s["calibration"]["observed_dr"],
                      "hosmer_lemeshow_p": s["calibration"]["hosmer_lemeshow_p"]}, indent=2, default=float))
    logging.info("report in %s", cfg.paths.validation_dir)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    main(ap.parse_args().config)
