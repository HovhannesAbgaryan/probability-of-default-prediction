"""Independent hold-out validation of the selected model (notebook 03 runs the same functions).

Each check is a separate function returning a table or a dict, so the notebook can show them one at a
time; ``run_validation`` runs them all and ``save_report`` writes them to ``reports/validation/``."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from pd_model.models.scorecard import grade
from pd_model.models.training import model_input_columns
from pd_model.validation.metrics import (
    auc,
    binomial_test_by_grade,
    bootstrap_metric,
    gini,
    hosmer_lemeshow,
    ks_stat,
    psi,
    summary_metrics,
)


@dataclass
class ValidationData:
    """Everything the checks need: raw split, preprocessed frames, fitted candidates and their PDs."""
    train: pd.DataFrame
    test: pd.DataFrame
    Xp_train: pd.DataFrame
    Xp_test: pd.DataFrame
    models: dict
    selected: str
    challenger: str
    target: str
    oof: pd.DataFrame | None = None
    p_train: dict = field(default_factory=dict)
    p_test: dict = field(default_factory=dict)

    @classmethod
    def build(cls, train, test, preprocessor, models, selected, challenger, cfg, oof=None) -> "ValidationData":
        Xp_tr = preprocessor.transform(train[cfg.raw_features])
        Xp_te = preprocessor.transform(test[cfg.raw_features])
        p_tr = {n: m.predict_proba(Xp_tr)[:, 1] for n, m in models.items()}
        p_te = {n: m.predict_proba(Xp_te)[:, 1] for n, m in models.items()}
        return cls(train, test, Xp_tr, Xp_te, models, selected, challenger, cfg.target, oof, p_tr, p_te)

    @property
    def y_train(self) -> pd.Series:
        return self.train[self.target]

    @property
    def y_test(self) -> pd.Series:
        return self.test[self.target]

    @property
    def pd_test(self) -> np.ndarray:
        return self.p_test[self.selected]

    @property
    def model_columns(self) -> list[str]:
        return model_input_columns(self.models[self.selected])


def replication(v: ValidationData):
    """Re-estimate the selected logistic specification with statsmodels MLE and compare test PDs."""
    cols = v.model_columns
    rep = sm.Logit(v.y_train.values, sm.add_constant(v.Xp_train[cols])).fit(disp=0)
    pd_rep = rep.predict(sm.add_constant(v.Xp_test[cols], has_constant="add"))
    diff = np.abs(np.asarray(pd_rep) - v.pd_test)
    return rep, {"max_abs_pd_diff": float(diff.max()), "mean_abs_pd_diff": float(diff.mean()),
                 "gini_dev": gini(v.y_test, v.pd_test), "gini_validator": gini(v.y_test, pd_rep),
                 "auc_dev": auc(v.y_test, v.pd_test), "auc_validator": auc(v.y_test, pd_rep)}


def discrimination(v: ValidationData) -> pd.DataFrame:
    rows = {}
    for n in v.models:
        has_oof = v.oof is not None
        rows[n] = {"Train Gini": gini(v.y_train, v.p_train[n]),
                   "CV Gini (OOF)": gini(v.oof[v.target], v.oof[n]) if has_oof else np.nan,
                   "Test Gini": gini(v.y_test, v.p_test[n]),
                   "Train AUC": auc(v.y_train, v.p_train[n]),
                   "CV AUC (OOF)": auc(v.oof[v.target], v.oof[n]) if has_oof else np.nan,
                   "Test AUC": auc(v.y_test, v.p_test[n]),
                   "Test KS": ks_stat(v.y_test, v.p_test[n])}
    d = pd.DataFrame(rows).T
    d["Train→Test drop"] = d["Train Gini"] - d["Test Gini"]
    d["Rel. drop %"] = 100 * d["Train→Test drop"] / d["Train Gini"]   # tolerance rules are defined on Gini
    d["Train→Test drop (AUC)"] = d["Train AUC"] - d["Test AUC"]
    return d.sort_values("Test Gini", ascending=False)


def discrimination_ci(v: ValidationData, n_boot: int, seed: int) -> dict:
    g = bootstrap_metric(v.y_test, v.pd_test, gini, n_boot=n_boot, seed=seed)
    a = bootstrap_metric(v.y_test, v.pd_test, auc, n_boot=n_boot, seed=seed)
    k = bootstrap_metric(v.y_test, v.pd_test, ks_stat, n_boot=n_boot, seed=seed)
    return {"gini": gini(v.y_test, v.pd_test), "gini_ci": [g[1], g[2]],
            "auc": auc(v.y_test, v.pd_test), "auc_ci": [a[1], a[2]],
            "ks": ks_stat(v.y_test, v.pd_test), "ks_ci": [k[1], k[2]]}


def calibration(v: ValidationData, n_grades: int = 10) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y, p = v.y_test.values, v.pd_test
    d = y.sum()
    z = (d - p.sum()) / np.sqrt((p * (1 - p)).sum())
    hl, hl_p, hl_tbl = hosmer_lemeshow(y, p, n_grades)
    binom = binomial_test_by_grade(y, p, n_grades)
    binom["flag (p<0.05)"] = binom["p_value_underestimation"] < 0.05
    summary = {"mean_pd": float(p.mean()), "observed_dr": float(y.mean()), "defaults": int(d),
               "expected_defaults": float(p.sum()), "citl_z": float(z),
               "citl_p": float(2 * (1 - stats.norm.cdf(abs(z)))), "hosmer_lemeshow": float(hl),
               "hosmer_lemeshow_p": float(hl_p), "grades_underestimating": int(binom["flag (p<0.05)"].sum()),
               **{k: summary_metrics(y, p)[k] for k in ("Brier", "LogLoss")}}
    return summary, hl_tbl, binom


def stability(v: ValidationData) -> tuple[float, pd.DataFrame]:
    """Score PSI and per-input CSI, train vs test (random split: low values are expected by construction)."""
    score_psi = psi(v.p_train[v.selected], v.pd_test)
    csi = pd.Series({c: psi(v.Xp_train[c], v.Xp_test[c]) for c in v.model_columns}, name="CSI").to_frame()
    return score_psi, csi


def p_underestimation(defaults: float, pds: np.ndarray) -> float:
    """One-sided p-value that observed defaults exceed the sum of PDs (normal approx., continuity-corrected)."""
    mu, sd = pds.sum(), np.sqrt((pds * (1 - pds)).sum())
    return float(1 - stats.norm.cdf((defaults - 0.5 - mu) / sd))


def segment_frame(v: ValidationData, income_cap: float, age_placeholder: float) -> pd.DataFrame:
    t = v.test.assign(pd=v.pd_test)
    t["age_band"] = pd.cut(t.age.mask(t.age == age_placeholder), [0, 25, 35, 50, 65, 120],
                           labels=["≤25", "26–35", "36–50", "51–65", "65+"])
    t["dti_band"] = pd.cut(t.dti, [0, 0.3, 0.5, 0.8, np.inf], labels=["≤0.3", "0.3–0.5", "0.5–0.8", ">0.8"])
    t["income_capped"] = (t.income >= income_cap - 1e-6).astype(int)
    return t


def segments(v: ValidationData, income_cap: float, age_placeholder: float,
             columns=("block", "region_risk", "age_band", "dti_band", "income_capped", "income_reconstructed",
                      "past_default", "inconsistent_record")) -> pd.DataFrame:
    t, y = segment_frame(v, income_cap, age_placeholder), v.target
    out = []
    for col in columns:
        for val, g in t.groupby(col, observed=True):
            out.append({"segment": f"{col} = {val}", "n": len(g), "defaults": int(g[y].sum()),
                        "observed DR": g[y].mean(), "mean PD": g.pd.mean(), "PD / DR": g.pd.mean() / g[y].mean(),
                        "p underest.": p_underestimation(g[y].sum(), g.pd.values),
                        "Gini": gini(g[y], g.pd) if g[y].nunique() > 1 else np.nan,
                        "AUC": auc(g[y], g.pd) if g[y].nunique() > 1 else np.nan})
    return pd.DataFrame(out).set_index("segment")


def challenger_comparison(v: ValidationData, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    yv, a, b = v.y_test.values, v.p_test[v.challenger], v.pd_test
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), len(yv))
        diffs.append(gini(yv[idx], a[idx]) - gini(yv[idx], b[idx]))
    diffs = np.array(diffs)
    auc_diffs = diffs / 2  # Δ AUC = Δ Gini / 2 on every resample
    return {"challenger": v.challenger, "gini_challenger": gini(yv, a), "gini_selected": gini(yv, b),
            "delta_gini_mean": float(diffs.mean()),
            "delta_gini_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "auc_challenger": auc(yv, a), "auc_selected": auc(yv, b), "delta_auc_mean": float(auc_diffs.mean()),
            "delta_auc_ci": [float(np.percentile(auc_diffs, 2.5)), float(np.percentile(auc_diffs, 97.5))],
            "spearman_pd": float(pd.Series(a).corr(pd.Series(b), method="spearman"))}


def grade_table(v: ValidationData, master_scale: list[dict]) -> pd.DataFrame:
    t = v.test.assign(pd=v.pd_test)
    labels = [g["grade"] for g in master_scale]
    t["grade"] = pd.Categorical([grade(p, master_scale) for p in t.pd], categories=labels, ordered=True)
    g = t.groupby("grade", observed=False).agg(n=(v.target, "size"), defaults=(v.target, "sum"),
                                                mean_PD=("pd", "mean"), observed_DR=(v.target, "mean"))
    g["share"] = g.n / g.n.sum()
    lower = [0.0] + [m["pd_upper"] for m in master_scale[:-1]]
    g["PD range"] = [f"{lo:.0%}–{m['pd_upper']:.0%}" for lo, m in zip(lower, master_scale, strict=True)]
    return g


def cutoff_table(v: ValidationData, lgd: float,
                 approval_rates=(1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60)) -> tuple[pd.DataFrame, dict]:
    t = v.test.assign(pd=v.pd_test)
    t["EL"] = t.pd * lgd * t.loan_amount
    t["loss_if_default"] = t[v.target] * lgd * t.loan_amount
    base_bad, base_loss, y = t[v.target].mean(), t.loss_if_default.sum(), v.target
    rows = []
    for appr in approval_rates:
        cut = t.pd.quantile(appr) if appr < 1 else 1.0
        a = t[t.pd <= cut]
        rows.append({"approval rate": len(a) / len(t), "PD cut-off": cut, "bad rate approved": a[y].mean(),
                     "bad rate reduction": 1 - a[y].mean() / base_bad,
                     "defaults avoided": 1 - a[y].sum() / t[y].sum(),
                     "goods lost": 1 - (1 - a[y]).sum() / (1 - t[y]).sum(),
                     "realised loss avoided": 1 - a.loss_if_default.sum() / base_loss})
    el = {"expected_loss": float(t.EL.sum()), "realised_loss": float(base_loss),
          "ratio": float(t.EL.sum() / base_loss)}
    return pd.DataFrame(rows), el


def run_validation(v: ValidationData, cfg) -> dict:
    """Run every check. Returns {"summary": dict, "tables": {name: DataFrame}}."""
    val, seed = cfg.validation, cfg.seed
    _, rep = replication(v)
    disc = discrimination(v)
    ci = discrimination_ci(v, val["n_bootstrap"], seed)
    cal, hl_tbl, binom = calibration(v, val["n_grades"])
    score_psi, csi = stability(v)
    segs = segments(v, cfg.preprocessing["income_cap"], cfg.preprocessing["age_placeholder"])
    chal = challenger_comparison(v, val["n_bootstrap"], seed)
    grades = grade_table(v, cfg.scorecard["master_scale"])
    cutoff, el = cutoff_table(v, val["lgd"])
    summary = {"selected": v.selected, "test_rows": len(v.test), "replication": rep, "discrimination": ci,
               "calibration": cal, "score_psi": score_psi, "challenger": chal, "expected_loss": el,
               "segments_underestimated_p05": segs.index[segs["p underest."] < 0.05].tolist()}
    tables = {"discrimination": disc, "hosmer_lemeshow": hl_tbl, "binomial_by_grade": binom, "csi": csi,
              "segments": segs, "master_scale": grades, "cutoff": cutoff}
    return {"summary": summary, "tables": tables}


def save_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report["summary"], indent=2, default=float) + "\n")
    for name, table in report["tables"].items():
        table.to_csv(out_dir / f"{name}.csv")
