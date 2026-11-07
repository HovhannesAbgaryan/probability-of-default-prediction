"""Fit the preprocessor and all candidates, compare them by cross-validation, select one, save artifacts.

    python scripts/train.py [--config configs/config.yaml] [--no-tuning]

Writes models/{preprocessor,model,candidates}.joblib, oof_predictions.parquet and metadata.json.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from pd_model.config import load_config
from pd_model.data.loading import load_split
from pd_model.models.training import run_training, save_artifacts


def main(config: str | None = None, tune: bool = True) -> None:
    cfg = load_config(config)
    train, _ = load_split(cfg.paths.processed_dir)  # the test set is never touched here
    result = run_training(train, cfg, tune_boosting=tune)
    meta = save_artifacts(result, cfg)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(result.comparison)
    logging.info("selected %s (challenger %s), model features %s", result.selected, result.challenger,
                 meta["model_features"])
    logging.info("artifacts in %s", cfg.paths.models_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--no-tuning", action="store_true", help="skip the boosting grid search (faster)")
    a = ap.parse_args()
    main(a.config, tune=not a.no_tuning)
