"""Raw extract -> repaired, split data in data/processed/.

    python scripts/clean_data.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging

from pd_model.config import load_config
from pd_model.data.cleaning import run_cleaning
from pd_model.data.loading import load_raw


def main(config: str | None = None) -> None:
    cfg = load_config(config)
    train, test, report = run_cleaning(load_raw(cfg.paths.raw_data), cfg)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cfg.paths.train)
    test.to_parquet(cfg.paths.test)
    cfg.paths.cleaning_report.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    logging.info("cleaning: %s", report.to_dict())
    logging.info("wrote %s, %s", cfg.paths.train, cfg.paths.test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    main(ap.parse_args().config)
