# Probability-of-Default Model: Development, Validation & Serving

Build a **probability-of-default (PD) model** for retail loans, validate it independently, give a production verdict, and serve it through an API.
The task is described in [`docs/candidate _task_DS.docx`](docs/).

| | |
|---|---|
| **Data** | 20,000 loans, 15 features + observed `default` (15.7% default rate), `data/raw/data.pkl` |
| **Final model** | Logistic regression on 7 drivers (`region_risk` one-hot encoded → 8 inputs) |
| **Test performance** | Gini 0.418 (95% CI 0.38–0.46) = AUC 0.709, KS 0.314, mean PD 15.69% vs observed 15.73% |
| **Verdict** | **Conditionally suitable** for application scoring; **not suitable as-is** for IRB capital or IFRS 9 |
| **Serving** | FastAPI service in [`service/`](service/README.md), loading the same fitted preprocessor + model |
| **Run everything** | `./run_all.sh --fresh` |

## Contents
1. [Quick start](#1-quick-start)
2. [Repository layout](#2-repository-layout)
3. [How it works: the pipeline](#3-how-it-works-the-pipeline)
4. [Stage by stage](#4-stage-by-stage)
5. [Configuration reference](#5-configuration-reference)
6. [Generated outputs](#6-generated-outputs)
7. [Results](#7-results)
8. [Verdict](#8-verdict)
9. [Scoring service](#9-scoring-service)
10. [Testing](#10-testing)
11. [Reproducibility](#11-reproducibility)
12. [Common tasks](#12-common-tasks)
13. [Glossary of metrics and terms](#13-glossary-of-metrics-and-terms)
14. [Limitations and open questions](#14-limitations-and-open-questions)

---

## 1. Quick start

```bash
# 1. environment (Python ≥ 3.11; developed on 3.15, service image uses 3.13)
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # pd_model package + training/notebook/test extras

#    macOS only: LightGBM needs the OpenMP runtime → brew install libomp
#    (use a Python built for your CPU; an Intel/x86_64 Python on Apple Silicon can't load Homebrew's arm64 libomp)

# 2. the supplied data file must be at data/raw/data.pkl

# 3. run everything: tests, clean, train, validate, notebooks, export, service tests
./run_all.sh --fresh
```

`run_all.sh` options:

| Option | Effect |
|---|---|
| `--fresh` | delete every generated output first (`data/processed`, `models`, `reports/training`, `reports/validation`, `service/artifacts`), then rebuild. The raw data is never touched |
| `--no-notebooks` | skip re-executing the three notebooks (about 3 minutes) |
| `--no-tests` | skip both test suites |
| `--version X.Y.Z` | model version written to the service model card (default `1.1.0`) |
| `PYTHON=...` | use another interpreter (default `.venv/bin/python`) |

A full run with notebooks takes about 3.5 minutes; most of it is the grid search in `scripts/train.py`. **Training happens only there**: the notebooks are read-only and load the saved models and reports, so running them can never change `models/`. The script ends with a summary of the selected model, test metrics and artifact hashes.

To serve the model locally:

```bash
cd service && pip install -r requirements-dev.txt && uvicorn app.main:app   # http://localhost:8000/docs
```

## 2. Repository layout

```
├── run_all.sh                      one command for the whole pipeline
├── pyproject.toml                  package "pd-model" + pinned dependencies (runtime / train / dev extras)
├── configs/config.yaml             every parameter: paths, seed, features, caps, CV grids, scorecard, master scale
├── data/
│   ├── raw/data.pkl                the supplied dataset (input, never modified)
│   └── processed/                  repaired train/test parquet + cleaning_report.json        [generated]
├── src/pd_model/                   the package: all logic lives here
│   ├── config.py                   loads configs/config.yaml into a typed Config object
│   ├── data/
│   │   ├── loading.py              load_raw (pickle), load_split (parquet)
│   │   └── cleaning.py             repairs specific to this extract + stratified split
│   ├── features/
│   │   ├── transformers.py         FeatureBuilder, Winsorizer, Cols, WoETransformer (sklearn-compatible)
│   │   ├── preprocessing.py        build_preprocessor(): THE preprocessor (imputers, one-hot, scaling)
│   │   └── woe.py                  Weight of Evidence / Information Value
│   ├── models/
│   │   ├── candidates.py           the 8 candidate models + monotone constraints
│   │   ├── training.py             CV, tuning, comparison, selection, saving/loading artifacts
│   │   └── scorecard.py            PD → score points, PD → rating grade
│   └── validation/
│       ├── metrics.py              Gini, AUC, KS, PSI, Hosmer–Lemeshow, binomial test, bootstrap, Wilson, SMD
│       └── validate.py             all hold-out validation checks + report writer
├── scripts/                        thin command-line entry points to the package
│   ├── clean_data.py               raw → data/processed/
│   ├── train.py                    → models/ + reports/training/  (the only step that trains)
│   ├── validate.py                 → reports/validation/
│   └── export_model.py             models/ → service/artifacts/ (+ model_card.json)
├── notebooks/                      narrative + plots; they call the package, they don't re-implement it
│   ├── 01_eda.ipynb                data quality & integrity findings → decisions
│   ├── 02_modelling.ipynb          development evidence: data, feature selection, 8 candidates, CV, selection
│   └── 03_validation_and_verdict.ipynb   independent validation, model risk, FINAL VERDICT
├── models/                         fitted preprocessor, selected model, all candidates, metadata  [generated]
├── reports/
│   ├── training/                   CV comparison, paired tests, per-fold scores, tuning grids    [generated]
│   └── validation/                 summary.json + one CSV per validation table                   [generated]
├── tests/                          package tests (synthetic data, no artifacts needed)
├── service/                        FastAPI scoring service: own README, Dockerfile, tests, artifacts/
└── docs/                           task description
```

The same code is used 3 ways:

```
                ┌──────────── src/pd_model (single implementation) ────────────┐
                │ config · data · features · models · validation                │
                └─────▲──────────────▲───────────────▲───────────────▲──────────┘
                      │              │               │               │
                 scripts/*.py    notebooks/      tests/          service/
                 (production     (narrative,     (32 tests)      (imports only features.*
                  pipeline)       plots)                          and models.scorecard)
```

## 3. How it works: the pipeline

```
 notebooks/01_eda.ipynb ──► findings ──► decisions ──┬──► configs/config.yaml   (VALUES: caps, lists, grids…)
                                                     └──► src/pd_model/         (RULES: repairs, transformers…)

 data/raw/data.pkl
      │  ① scripts/clean_data.py           pd_model.data.cleaning
      ▼
 data/processed/train.parquet (14,000) + test.parquet (6,000) + cleaning_report.json
      │  ② scripts/train.py                pd_model.models.training          (train only)
      │     5-fold CV, each fold refits [preprocessor + candidate]; 8 candidates; select
      ▼
 models/preprocessor.joblib · model.joblib · candidates.joblib · oof_predictions.parquet · metadata.json
 reports/training/ (CV evidence)                 ◄── notebook 02 presents it (read-only)
      │  ③ scripts/validate.py             pd_model.validation.validate      (test only)
      ▼
 reports/validation/summary.json + CSVs          ◄── notebook 03 presents them + verdict
      │  ④ scripts/export_model.py
      ▼
 service/artifacts/preprocessor.joblib · model.joblib · model_card.json (sha256, versions, metrics)
      │
      ▼
 FastAPI service: POST /v1/predict → PD, score, grade, reason codes
```

Each step reads only the previous step's outputs, so any step can be re-run on its own.

## 4. Stage by stage

### 4.1 EDA → decisions (`notebooks/01_eda.ipynb`)

The EDA checks structure, integrity, distributions, consistency, leakage and predictive power, and ends with a table of actions and a list of questions for the data owner. Each finding became a value in the config, a rule in the code, or both:

| EDA finding | Value in `config.yaml` | Rule in code |
|---|---|---|
| `recovery_flag_after_90d` is post-default information (matches the target in 95% of rows) | `cleaning.leaky_columns` | dropped in `data/cleaning.py`; the API rejects it |
| Index restarts at row 13,000 → two concatenated blocks; all injected errors are in block 2 | `cleaning.block_boundary` | `row_id`, `block` columns (validation only) |
| `dti` is exactly `loan_amount / income` in block 1 | — | used to detect and reverse the corruption |
| 80 loans multiplied by 10 (incl. 800k vs the 80k product cap) | — | detection + ÷10 in `data/cleaning.py` |
| 700 incomes deleted: every block-2 income below ≈16.8k (missing-not-at-random) | `cleaning.income_cap` | rebuilt exactly as `loan / dti` |
| Income censored at 30k for 62.5% of clients | `preprocessing.income_cap` | cap + `income_capped` flag in `FeatureBuilder` |
| `age = 99` is a placeholder | `preprocessing.age_placeholder` | 99 → missing in `FeatureBuilder` |
| `dti` has a genuine heavy tail | `preprocessing.winsorize` | `Winsorizer` (train p99.5) |
| A future record could miss *any* field | `preprocessing.numeric` / `categorical` | an imputer on every column |
| `region_risk`: risk rises 0 → 1 → 2 in **unequal** steps | `preprocessing.categorical` | one-hot encoding |
| Risk rises with DPD, dti, utilisation; falls with income | `training.monotone` | constraints for LightGBM / XGBoost |
| DPD buckets not nested, DPD > number of loans, implausible age/tenure | — | rows kept; `inconsistent_record` flag for validation |

### 4.2 Cleaning the extract (`pd_model.data.cleaning`, `scripts/clean_data.py`)

These repairs fix **this file** and run once, before the split. They are not part of scoring-time preprocessing, because new applications don't carry these errors.

1. Reset the non-unique index → `row_id`; `block` = 1 for rows < 13,000, else 2.
2. Drop `recovery_flag_after_90d`.
3. Loans first: rows where `loan / (dti × income) = 10`, or where income is missing and the implied income `loan / dti` exceeds the 30k cap, get `loan_amount ÷ 10` (80 rows).
4. Then incomes: missing income = `loan_amount / dti` (700 rows, flagged `income_reconstructed`). After this, `dti = loan / income` holds for every row (max error 2e-9).
5. Flag `inconsistent_record` (employment > age − 14, tenure > adult life, DPD sum > number of loans): 2,281 rows, kept.
6. Stratified 70/30 split (seed 42). The data has no dates, so an out-of-time split is impossible.

The cleaned data still has the `age = 99` placeholders and 17 missing `credit_utilization` values. Those are the preprocessor's job.

### 4.3 The preprocessor (`pd_model.features.preprocessing`)

One sklearn `Pipeline`, fitted on the training set, saved as `models/preprocessor.joblib`, and used unchanged in cross-validation, validation and the service:

```
raw record (13 fields; any may be missing or absent)
   │
   ▼  FeatureBuilder (stateless domain rules)
   │    • absent column → NaN       • age == 99 → NaN       • income capped at 30,000
   │    • dti = loan / income when not supplied
   │    • + income_capped, total_past_due (sum of DPD counts), any_past_due
   │    • region_risk cast to a label ("0", "1", "2")
   ▼  Winsorizer (learned): dti capped at the train 99.5th percentile (2.79)
   ▼  ColumnTransformer (learned)
   │    numeric (16):    SimpleImputer(median, add_indicator=True) → StandardScaler
   │    categorical (1): SimpleImputer(most_frequent) → OneHotEncoder(drop="first", handle_unknown="ignore")
   ▼
20 columns, no missing values, stable names:
   16 scaled numeric + missingindicator_age + missingindicator_credit_utilization + region_risk_1 + region_risk_2
```

Design choices:
- **Every column has an imputer**, not only the 2 that had gaps in this extract. A future application missing any field is still scored: median for numbers, most frequent value for categories. Tests check each feature in turn.
- **Missing-value indicators** are added automatically for columns that were missing in training. `missingindicator_age` replaces a hand-made `age_missing` flag.
- **`region_risk` is categorical.** Risk rises by +0.34 log-odds from region 0 to 1 but only +0.14 from 1 to 2 (EDA §9.1), so one numeric term would force equal steps. Region 0 is the reference. An unseen code is encoded as the reference.
- **Scaling** makes logistic-regression coefficients comparable (per standard deviation). Notebook 02 converts them back to per-raw-unit odds ratios.
- **No leakage in validation.** Cross-validation refits the whole `preprocessor + model` in every fold.

### 4.4 Training and selection (`pd_model.models`, `scripts/train.py`, notebook 02)

All 8 candidates take the preprocessed frame:

| Candidate | Purpose |
|---|---|
| **A** Logistic regression, all 20 inputs | transparent baseline |
| **B** Logistic regression, parsimonious (7 drivers, 8 inputs) | simplest defensible model |
| **C** WoE scorecard (WoE binning + logistic regression) | industry-standard scorecard |
| **D** LightGBM, monotone constraints | non-linearity with economically sensible directions |
| **E** LightGBM, unconstrained | upper bound on available signal |
| **F** XGBoost, monotone constraints | second boosting library |
| **G** XGBoost, unconstrained | as E, for XGBoost |
| **H** Random forest | bagging instead of boosting, so the conclusion doesn't depend on one family of tree models |

- **Parsimonious drivers** (`training.parsimonious` in the config) were chosen in notebook 02. A variable is kept if it is significant (p < 0.01), has an economically sensible sign, and is material. The result is `past_default`, `past_due_30/60/90`, `dti`, `region_risk`, `credit_utilization`. `client_history_in_bank` is significant but has a counter-intuitive sign and no information value, so it was excluded. Income and loan amount add nothing once `dti` (= loan / income) is in the model.
- **Tuning:** a small, strongly regularised grid for each tree model (`lgbm_grid`, `xgb_grid`, `rf_grid`). The shallowest trees and the largest leaves win, so there is little non-linear structure.
- **Comparison:** 5-fold stratified CV, reporting Gini, AUC, KS, Brier, train-vs-CV gap, and paired t-tests on identical folds.
- **Selection rule:** among models within **0.01 CV Gini** (= 0.005 AUC) of the best, pick the simplest (`training.complexity`). The challenger is the monotone LightGBM.
- **Saved:** the fitted preprocessor, the selected model alone (`model.joblib`, a `Pipeline[Cols → LogisticRegression]`), all candidates, out-of-fold PDs, and `metadata.json` with features, learned parameters, the CV table, library versions and sha256 hashes.
- **Single writer:** `scripts/train.py` is the only code that trains or writes `models/` and `reports/training/`. Notebook 02 loads them (hash-checked) and adds the statistical analysis (feature selection, VIF, coefficients, score points), which takes seconds.

Scoring is always `model.predict_proba(preprocessor.transform(raw))[:, 1]`.

### 4.5 Validation (`pd_model.validation`, `scripts/validate.py`, notebook 03)

The validator loads the **saved** artifacts (hash-checked) and uses only the untouched 30% hold-out:

| § | Check | Function |
|---|---|---|
| 1 | Replication: independent statsmodels refit | `replication` |
| 2 | Discrimination: Gini, AUC, KS with bootstrap CIs; dev → test drop | `discrimination`, `discrimination_ci` |
| 3 | Calibration: in-the-large z-test, Hosmer–Lemeshow, binomial test per decile | `calibration` |
| 4 | Stability: score PSI, input CSI | `stability` |
| 5 | Segments: calibration and Gini/AUC by block, region, age, dti, data-quality flags | `segments` |
| 6 | Robustness: leakage impact, coefficient bootstrap, driver removal, input shocks | notebook 03 |
| 7 | Benchmark: challenger Δ Gini/AUC with bootstrap CI, SHAP | `challenger_comparison` |
| 8 | Business: master scale A–G, cut-off strategy, expected loss | `grade_table`, `cutoff_table` |
| 9–10 | Model-risk register and **final verdict** | notebook 03 (written by hand) |

`scripts/validate.py` writes the numbers to `reports/validation/`. Notebook 03 is the **validation report**: it presents the same numbers with plots, the robustness analyses, the model-risk table and the verdict. To hand it to a committee without code:

```bash
jupyter nbconvert --to html --no-input notebooks/03_validation_and_verdict.ipynb
```

### 4.6 Export (`scripts/export_model.py`)

The export does the following:
- Loads `models/` (verifying sha256 against `metadata.json`) and refuses if the selected model isn't a logistic regression, because the service's score and reason codes need one.
- Requires `reports/validation/summary.json`.
- Copies the two joblib files into `service/artifacts/`.
- Writes `model_card.json`: version, input/model features, training means of the model inputs (for reason codes), income and dti caps, scorecard, master scale, file hashes, scikit-learn and pd-model versions, and the validation results and status.

## 5. Configuration reference

All parameters are in `configs/config.yaml`. Paths are relative to the repository root. Override the file with `PD_MODEL_CONFIG=/path/to/config.yaml`.

| Key | Meaning |
|---|---|
| `seed`, `target` | random seed (split, CV, models, bootstrap); target column |
| `paths.*` | raw data, processed data, models, reports, service artifacts |
| `cleaning.leaky_columns` | columns dropped as target leakage |
| `cleaning.block_boundary` | row where the second block starts |
| `cleaning.income_cap` | cap used to detect inflated loans via implied income |
| `cleaning.test_size` | hold-out share (0.30) |
| `preprocessing.age_placeholder` | value treated as "unknown age" (99) |
| `preprocessing.income_cap` | income censoring level (30,000) |
| `preprocessing.winsorize` | `{column: upper quantile}` learned on train |
| `preprocessing.numeric` | numeric features (raw + engineered), each gets a median imputer and scaling |
| `preprocessing.categorical` | categorical features, each gets a mode imputer and one-hot encoding |
| `training.cv_folds`, `equivalence_gini` | CV folds; "practically equivalent" Gini margin |
| `training.parsimonious` | source features of candidate B (the selected model) |
| `training.woe_exclude` | features left out of the WoE scorecard |
| `training.monotone` | `+1` / `-1` monotone constraints for the boosting models |
| `training.lgbm_grid`, `xgb_grid`, `rf_grid` | tuning grids |
| `training.complexity` | tie-breaker ranking (lower = simpler) |
| `scorecard.pdo`, `base_score`, `base_odds` | score scale: 600 points at 50:1 odds, +20 per doubling |
| `scorecard.master_scale` | grade A–G upper PD bounds |
| `validation.n_bootstrap`, `n_grades`, `lgd` | bootstrap resamples; calibration groups; illustrative LGD (45%) |

## 6. Generated outputs

| File | Produced by | Contents |
|---|---|---|
| `data/processed/train.parquet`, `test.parquet` | clean | repaired raw features + target + `row_id`, `block`, `income_reconstructed`, `inconsistent_record` |
| `data/processed/cleaning_report.json` | clean | rows repaired, flags, split sizes and default rates |
| `models/preprocessor.joblib` | train | the fitted preprocessor |
| `models/model.joblib` | train | the selected model (reads the preprocessor's output) |
| `models/candidates.joblib` | train | all 8 fitted candidates + selected / challenger names |
| `models/oof_predictions.parquet` | train | out-of-fold CV PDs of every candidate |
| `models/metadata.json` | train | features, learned preprocessing parameters, CV table, versions, sha256 |
| `reports/training/*.csv` | train | CV comparison, paired tests, per-fold scores, tuning grid results |
| `reports/validation/summary.json` | validate | headline results of every check |
| `reports/validation/*.csv` | validate | discrimination, Hosmer–Lemeshow, binomial by grade, CSI, segments, master scale, cut-offs |
| `service/artifacts/*` | export | the two joblib files + `model_card.json` |

## 7. Results

### 7.1 Data quality
| Issue | Treatment |
|---|---|
| **Target leakage.** `recovery_flag_after_90d` matches the target in 95% of rows; with it the test Gini jumps to 0.90 | Dropped; the API rejects the field |
| **Two concatenated files**; every injected error is in the 7,000-row second block | Index reset; block kept as a validation segment |
| **`dti` = `loan_amount / income`** exactly, i.e. loan-to-income, not total debt-to-income | Used to repair the corrupted fields |
| **80 loan amounts ×10** | Corrected exactly (÷10) |
| **700 incomes deleted (MNAR)**: every block-2 income below ≈16.8k; these clients default at 23.7% | Reconstructed exactly (median imputation would have put them at 30k) |
| Income censored at 30k; age 99 placeholder; age clipped to [21, 75]; 3% interest floor | Flags and imputation in the preprocessor; documented |
| Logical contradictions, equally frequent in both blocks | Rows kept; flagged; definitions to confirm |

### 7.2 Model comparison (5-fold CV on the 70% training set)
Gini = 2·AUC − 1; both are reported and rank models identically.

| Model | CV Gini | CV AUC | Overfit gap (Gini) |
|---|---|---|---|
| A · Logistic regression, all 20 inputs | 0.4414 | 0.7207 | 0.006 |
| **B · Logistic regression, 7 drivers (selected)** | **0.4406** | **0.7203** | **0.001** |
| D · LightGBM, monotone | 0.4354 | 0.7177 | 0.026 |
| F · XGBoost, monotone | 0.4351 | 0.7175 | 0.048 |
| E · LightGBM, unconstrained | 0.4347 | 0.7174 | 0.033 |
| G · XGBoost, unconstrained | 0.4335 | 0.7168 | 0.050 |
| H · Random forest | 0.4222 | 0.7111 | 0.068 |
| C · WoE scorecard | 0.4162 | 0.7081 | 0.007 |

The top six are statistically indistinguishable (paired t-tests, p > 0.2), so the simplest was selected. Neither boosting nor the random forest finds extra signal, and both overfit more. The random forest is 0.018 Gini behind B, outside the 0.01 equivalence margin, and on the test set it has the largest development → test drop of all candidates (16%, amber).

### 7.3 The selected model
| Driver | Odds ratio | Score points |
|---|---|---|
| past default (yes vs no) | ×4.6 | −44 |
| each 30+ DPD loan | ×1.5 | −11 |
| each 60+ DPD loan | ×2.0 | −20 |
| each 90+ DPD loan | ×3.1 | −32 |
| dti +1.0 (loan larger by one more unit of income) | ×2.5 | −26 |
| credit utilisation 0% → 100% | ×1.9 | −19 |
| region 1 / region 2 (vs region 0) | ×1.43 / ×1.81 | −10 / −17 |

Every effect is significant, correctly signed and independent (VIF ≈ 1), and keeps its sign in 100% of bootstrap samples. Score scale: 600 points at 50:1 good:bad odds, 20 points to double the odds.

### 7.4 Validation on the untouched 30% hold-out
- **Replication:** an independent statsmodels fit reproduces the PDs to within 0.0005.
- **Discrimination:** Gini 0.418 (CI 0.38–0.46) = AUC 0.709 (CI 0.69–0.73), KS 0.314. A 5.3% relative drop from development, which is green.
- **Calibration:** mean PD 15.69% vs observed 15.73% (p = 0.92). Hosmer–Lemeshow p = 0.44. No decile significantly under-predicts.
- **Stability:** score PSI 0.002. This is expected for a random split; stability over time is not tested.
- **Segments:** Gini 0.31–0.53 in every segment, and both data blocks perform the same. The one significant under-prediction is records with logical contradictions (22.8% observed vs 20.0% predicted, p = 0.026).
- **Robustness:** with the leaky flag, Gini would be 0.90. `dti` and `past_default` carry most of the model (about 0.07 Gini each). Input shocks give smooth, directionally correct responses.
- **Challenger:** LightGBM is not better (Δ Gini −0.003, CI includes 0; rank correlation 0.97).
- **Business:** grades A–G rank-order from 4.2% to 52.2% observed default rate. Declining the riskiest 10% avoids 28% of defaults and about 41% of losses, at the cost of 6.6% of good clients. Expected loss matches realised loss within 0.4% (illustrative LGD 45%).

## 8. Verdict

**Conditionally suitable** for production use as an **application-scoring PD model** (credit decisioning and risk-based pricing support). The full reasoning is in notebook 03 §9–10.

The model is sound: simple, transparent, well calibrated on hold-out data, free of overfitting, and matched by a far more flexible challenger. Its moderate Gini reflects the limited information in the data. The remaining risks are in the data and the validation evidence. Conditions before go-live:
1. Confirm the default, DPD, `num_loans` and `dti` definitions with the data owner, then re-test the `inconsistent_record` segment.
2. Lock the production data contract: `recovery_flag_after_90d` excluded, and input validation for the block-2 error types.
3. Run an out-of-time validation on the most recent vintage (impossible here, because the data has no dates).
4. Recalibrate the intercept to the bank's long-run default rate.

After go-live: monthly PSI/CSI monitoring, quarterly backtests, the challenger kept in parallel, and annual re-validation.

**Not suitable as-is for IRB capital or IFRS 9.** Those need long-run (through-the-cycle) or point-in-time, forward-looking PDs, multi-year data, lifetime PD term structures and a margin of conservatism, none of which this dataset supports. See the [glossary](#13-glossary-of-metrics-and-terms).

## 9. Scoring service

[`service/`](service/README.md) is a FastAPI service for the selected model:
- `POST /v1/predict` and `/v1/predict/batch` return the PD, score, grade A–G, reason codes and input warnings. `GET /v1/model` returns metadata; health probes and Prometheus metrics are also exposed.
- It loads the **same fitted preprocessor and model** that were validated. Before unpickling it verifies their sha256 against `model_card.json` and checks the scikit-learn version. A parity test matches the validated PDs on all 6,000 hold-out rows.
- **Request fields.** Required: `loan_amount`, `income`, `past_default`, `past_due_30/60/90`, `region_risk`. `credit_utilization` and the fields the model doesn't use are optional and imputed when missing. Unknown fields (including the leaky flag) are rejected.
- **Image.** It depends on the `pd-model` package, so it's built from the repository root: `docker build -f service/Dockerfile -t pd-scoring-service:1.1.0 .`. Only runtime dependencies are installed (no LightGBM, XGBoost or statsmodels).

## 10. Testing

| Suite | Command | What it covers |
|---|---|---|
| Package (32 tests) | `pytest` | cleaning repairs, split; preprocessor (NaN in **any** feature, absent columns, age 99, unseen region, income cap / winsorising, nothing hard-coded); per-fold preprocessor refit in CV; selection rule; training smoke test; metrics, scorecard, grades |
| Service (44 tests) | `cd service && pytest` | API contract, validation errors, warnings, batch, auth, metrics; model monotonicity; reason codes add up to the logit; startup refuses tampered or mismatched artifacts; **parity with the validated model** on the full hold-out |

The package tests use synthetic data and need no artifacts. The service parity test needs the training artifacts and is skipped if they're missing.

## 11. Reproducibility

- Everything is seeded (`seed: 42`): split, CV folds, models and bootstrap. In the same environment, re-running produces **byte-identical** model files (the sha256 hashes in `models/metadata.json` don't change). A different Python version pickles the same model to different bytes, so the hashes differ, but the results don't: a clean install on Python 3.13 reproduced `reports/validation/summary.json` byte for byte (developed on 3.15).
- Library versions are pinned in `pyproject.toml`. scikit-learn must match between training and serving, because the artifacts are pickled. The model card records it, and the service enforces it.
- The notebooks are committed with their outputs, so you can read them without re-running anything.

## 12. Common tasks

| Task | How |
|---|---|
| Rebuild everything from scratch | `./run_all.sh --fresh` |
| Change a cap, quantile, grid or the master scale | edit `configs/config.yaml`, then `./run_all.sh` |
| Add a new raw feature | add it to `preprocessing.numeric` or `categorical` (it gets an imputer automatically); add a constraint in `training.monotone` if needed; to use it in the selected model, add it to `training.parsimonious` (justify it in notebook 02) and to the service request schema |
| Run one step | `python scripts/clean_data.py`, `scripts/train.py [--no-tuning]`, `scripts/validate.py`, `scripts/export_model.py --version X.Y.Z` |
| Release a new model version | rebuild, get validation sign-off in notebook 03, `./run_all.sh --version 1.2.0`, commit `service/artifacts/`, build and deploy the image |
| Hand over the validation report | `jupyter nbconvert --to html --no-input notebooks/03_validation_and_verdict.ipynb` |

## 13. Glossary of metrics and terms

| Term | Meaning |
|---|---|
| **PD** | probability of default: the model's estimated chance that a loan defaults |
| **AUC** | area under the ROC curve: the probability that a random defaulter gets a higher PD than a random non-defaulter. 0.5 = random, 1 = perfect |
| **Gini** (Accuracy Ratio, Somers' D) | 2·AUC − 1: 0 = random, 1 = perfect. The credit-risk convention for reporting discrimination. Not the Gini impurity used for tree splits. Rules of thumb for application scoring: < 0.30 weak, 0.30–0.45 acceptable, 0.45–0.60 good |
| **KS** | Kolmogorov–Smirnov: the largest gap between the cumulative shares of defaulters and non-defaulters when clients are sorted by PD. Our 0.314: flagging the riskiest 31.7% (PD ≥ 15.9%) catches 58.2% of defaulters but only 26.8% of non-defaulters |
| **Calibration-in-the-large** | does the mean PD equal the observed default rate? |
| **Hosmer–Lemeshow** | goodness-of-fit test over PD deciles; a high p-value means no evidence of miscalibration |
| **Binomial test by grade** | per grade, are observed defaults significantly **above** the predicted PD (the conservative concern)? |
| **PSI / CSI** | Population / Characteristic Stability Index: distribution shift of the score / an input. < 0.10 stable, 0.10–0.25 monitor, > 0.25 shift |
| **WoE / IV** | Weight of Evidence per bin / Information Value of a feature. IV < 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, > 0.5 suspicious (leakage) |
| **Wilson interval** | confidence interval for a default rate; reliable for small groups |
| **SMD** | standardised mean difference, an effect size for comparing two groups (\|SMD\| < 0.1 negligible) |
| **MNAR** | missing-not-at-random: whether a value is missing depends on the value itself (here, low incomes were deleted) |
| **EL = PD × LGD × EAD** | expected loss = probability of default × loss given default × exposure at default |
| **Master scale** | mapping of PD ranges to rating grades A–G |
| **IRB** | Basel Internal Ratings-Based approach: banks use their own PD (and LGD/EAD) models to compute **regulatory capital**. Requires a long-run (through-the-cycle) PD, ≥ 5 years of data, the regulatory default definition, a margin of conservatism and supervisory approval |
| **IFRS 9** | accounting standard for **loan-loss provisions**: 12-month expected loss (stage 1) or lifetime expected loss (stages 2–3), based on point-in-time, forward-looking PDs under macroeconomic scenarios |
| **Out-of-time (OOT) validation** | testing on a later period than the development data: the main test of stability over time |

## 14. Limitations and open questions

- **No time dimension.** Only a random hold-out was possible: no out-of-time test, no stability over time, and no long-run default rate.
- **Unconfirmed definitions.** Default and horizon, DPD bucket semantics, `num_loans`, `dti` (loan-to-income), and the meaning of the `region_risk` codes. The full list of 12 questions is in notebook 01 §12.
- **Moderate, concentrated signal.** Gini 0.42, dominated by `dti` and `past_default`. Income is censored at 30k, so the model can't tell a 30k earner from a 100k earner.
- **Synthetic-looking data.** Features are almost independent, and there are implausible age/tenure combinations. Re-develop on real portfolio data before relying on the model.
- **Sample selection.** Only booked loans are observed (no rejected applicants).
