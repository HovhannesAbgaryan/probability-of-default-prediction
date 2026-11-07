"""Discrimination, calibration, stability and descriptive-statistics helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 42
def auc(y_true, y_score) -> float:
    """Area under the ROC curve: probability that a random defaulter scores above a random non-defaulter."""
    return float(roc_auc_score(y_true, y_score))


def gini(y_true, y_score) -> float:
    """Gini coefficient (Accuracy Ratio, Somers' D) = 2·AUC − 1. The credit-risk convention; ranks
    models exactly like AUC. Not the Gini impurity used for tree splits."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_stat(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def hosmer_lemeshow(y_true, y_prob, n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    """Hosmer-Lemeshow goodness-of-fit test on probability deciles.

    Returns (statistic, p-value, per-bin table). df = n_bins - 2.
    """
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"),
                             mean_pd=("p", "mean"), observed_dr=("y", "mean"))
    hl = (((g.observed - g.expected) ** 2) / (g.expected * (1 - g.expected / g.n))).sum()
    p_value = 1 - stats.chi2.cdf(hl, n_bins - 2)
    return float(hl), float(p_value), g


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index; bins defined on `expected` quantiles."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + eps
    a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def binomial_test_by_grade(y_true, y_prob, n_grades: int = 10) -> pd.DataFrame:
    """Per-grade one-sided binomial test: is the observed default count above the predicted PD?"""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    d["grade"] = pd.qcut(d["p"].rank(method="first"), n_grades, labels=range(1, n_grades + 1))
    g = d.groupby("grade", observed=True).agg(n=("y", "size"), defaults=("y", "sum"),
                                              mean_pd=("p", "mean"))
    g["observed_dr"] = g.defaults / g.n
    g["p_value_underestimation"] = [1 - stats.binom.cdf(k - 1, n, p)
                                    for k, n, p in zip(g.defaults, g.n, g.mean_pd, strict=True)]
    return g


def bootstrap_metric(y_true, y_score, fn=gini, n_boot: int = 1000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        vals.append(fn(y_true[idx], y_score[idx]))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson_ci(k, n, z: float = 1.959964) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion k / n (95% by default)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def rate_table(groups, y) -> pd.DataFrame:
    """Default rate per group with 95% Wilson intervals."""
    d = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y)})
    t = d.groupby("g", dropna=False).y.agg(n="size", defaults="sum")
    t["default_rate"] = t.defaults / t.n
    ci = [wilson_ci(k, n) for k, n in zip(t.defaults, t.n, strict=True)]
    t["ci_low"], t["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    return t.rename_axis(None)


def smd(a, b) -> float:
    """Standardised mean difference (b - a) / pooled SD, NaNs ignored."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("nan")


def summary_metrics(y_true, y_prob) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Gini": gini(y_true, y_prob),
        "KS": ks_stat(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "LogLoss": log_loss(y_true, y_prob),
        "MeanPD": float(np.mean(y_prob)),
        "ObservedDR": float(np.mean(y_true)),
    }

