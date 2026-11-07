#!/usr/bin/env bash
# Run the whole PD-model pipeline end to end, from the raw file to the scoring-service artifacts.
#
#   ./run_all.sh                    # run every step on top of existing outputs
#   ./run_all.sh --fresh            # first delete ALL generated outputs, then rebuild everything
#   ./run_all.sh --no-notebooks     # skip re-executing the notebooks (~3 min instead of ~3.5)
#   ./run_all.sh --no-tests         # skip both test suites
#   ./run_all.sh --version 1.2.0    # model version written to the service model card (default 1.1.0)
#   PYTHON=/path/to/python ./run_all.sh   # use another interpreter (default: .venv/bin/python)
#
# Steps:
#   0. [--fresh]  delete data/processed, models, reports/{training,validation}, service/artifacts
#   1. package tests          pytest tests/                              (synthetic data, no artifacts)
#   2. clean                  scripts/clean_data.py   raw → data/processed/
#   3. train                  scripts/train.py        → models/ + reports/training/ (the ONLY step that trains)
#   4. validate               scripts/validate.py     → reports/validation/
#   5. notebooks              01_eda, 02_modelling, 03_validation_and_verdict executed in place (read-only)
#   6. export                 scripts/export_model.py → service/artifacts/ (+ model_card.json)
#   7. service tests          pytest service/tests/   (incl. parity with the validated model)
#
# The raw data (data/raw/data.pkl) is never modified or deleted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FRESH=0
NOTEBOOKS=1
TESTS=1
VERSION="1.1.0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FRESH=1 ;;
    --no-notebooks) NOTEBOOKS=0 ;;
    --no-tests) TESTS=0 ;;
    --version) VERSION="${2:?--version needs a value}"; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON." >&2
  echo "Create the environment first:  python -m venv .venv && .venv/bin/pip install -e \".[dev]\"" >&2
  exit 1
fi
if ! "$PYTHON" -c "import pd_model" 2>/dev/null; then
  echo "The pd_model package is not installed in $PYTHON." >&2
  echo "Install it:  $PYTHON -m pip install -e \".[dev]\"" >&2
  exit 1
fi
if [[ ! -f data/raw/data.pkl ]]; then
  echo "Raw data missing: put the supplied file at data/raw/data.pkl" >&2
  exit 1
fi

START=$SECONDS
step() { printf '\n\033[1;34m==> [%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
done_step() { printf '\033[0;32m    ok (%ss)\033[0m\n' "$1"; }
run() {  # run a step and report its duration
  local t=$SECONDS
  "$@"
  done_step $((SECONDS - t))
}

if [[ $FRESH -eq 1 ]]; then
  step "0. Deleting generated outputs (raw data is kept)"
  rm -f  data/processed/train.parquet data/processed/test.parquet data/processed/cleaning_report.json
  rm -f  models/preprocessor.joblib models/model.joblib models/candidates.joblib \
         models/oof_predictions.parquet models/metadata.json
  rm -rf reports/validation reports/training
  rm -f  service/artifacts/preprocessor.joblib service/artifacts/model.joblib service/artifacts/model_card.json
  done_step 0
fi

if [[ $TESTS -eq 1 ]]; then
  step "1. Package tests"
  run "$PYTHON" -m pytest -q tests
fi

step "2. Clean the raw extract and split train/test"
run "$PYTHON" scripts/clean_data.py

step "3. Train: preprocessor + 8 candidates, 5-fold CV, selection"
run "$PYTHON" scripts/train.py

step "4. Validate the selected model on the hold-out"
run "$PYTHON" scripts/validate.py

if [[ $NOTEBOOKS -eq 1 ]]; then
  step "5. Execute the notebooks"
  for nb in 01_eda 02_modelling 03_validation_and_verdict; do
    echo "    - $nb"
    run "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace "notebooks/$nb.ipynb" --log-level=WARN
  done
fi

step "6. Export the model to the scoring service"
run "$PYTHON" scripts/export_model.py --version "$VERSION"

if [[ $TESTS -eq 1 ]]; then
  step "7. Service tests (API, model, parity with the validated model)"
  (cd service && run "$PYTHON" -m pytest -q)
fi

step "Summary"
"$PYTHON" - <<'EOF'
import json
from pd_model.config import load_config

cfg = load_config()
meta = json.loads(cfg.paths.metadata.read_text())
s = json.loads((cfg.paths.validation_dir / "summary.json").read_text())
card = json.loads((cfg.paths.service_artifacts_dir / "model_card.json").read_text())
d, c, ch = s["discrimination"], s["calibration"], s["challenger"]
print(f"  selected model     : {meta['selected']} (challenger {meta['challenger']})")
print(f"  model inputs       : {', '.join(meta['model_features'])}")
print(f"  test Gini / AUC    : {d['gini']:.3f} (95% CI {d['gini_ci'][0]:.3f}–{d['gini_ci'][1]:.3f}) / "
      f"{d['auc']:.3f} (CI {d['auc_ci'][0]:.3f}–{d['auc_ci'][1]:.3f})")
print(f"  test KS            : {d['ks']:.3f}")
print(f"  mean PD vs DR      : {c['mean_pd']:.2%} vs {c['observed_dr']:.2%} | Hosmer–Lemeshow p = {c['hosmer_lemeshow_p']:.2f}")
print(f"  challenger Δ Gini  : {ch['delta_gini_mean']:+.4f} (CI {ch['delta_gini_ci'][0]:+.4f} to {ch['delta_gini_ci'][1]:+.4f})")
print(f"  service artifact   : version {card['model_version']}, status {card['validation']['status']}")
print(f"  sha256 preprocessor: {meta['sha256']['preprocessor'][:16]}…  model: {meta['sha256']['model'][:16]}…")
EOF
printf '\n\033[1;32mAll steps finished in %ss.\033[0m\n' $((SECONDS - START))
