"""Weight of Evidence (WoE) and Information Value (IV)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bin_edges(x: pd.Series, max_bins: int) -> np.ndarray:
    """Quantile edges, or midpoints between levels when a feature has few distinct values."""
    x = x.dropna()
    if x.nunique() <= max_bins:
        vals = np.sort(x.unique())
        edges = np.concatenate([[-np.inf], (vals[:-1] + vals[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    return edges


def woe_table(x: pd.Series, y: pd.Series, max_bins: int = 10, eps: float = 0.5) -> pd.DataFrame:
    """WoE/IV table (WoE = ln(%good / %bad); positive WoE = lower risk). Missing gets its own bin."""
    edges = bin_edges(x, max_bins)
    b = pd.cut(x, edges).astype(object).where(x.notna(), "Missing")
    t = pd.crosstab(b, y)
    t.columns = ["good", "bad"]
    t["n"] = t.good + t.bad
    t["bad_rate"] = t.bad / t.n
    pg = (t.good + eps) / (t.good.sum() + eps)
    pb = (t.bad + eps) / (t.bad.sum() + eps)
    t["woe"] = np.log(pg / pb)
    t["iv"] = (pg - pb) * t["woe"]
    return t


def information_value(df: pd.DataFrame, y: pd.Series, max_bins: int = 10) -> pd.Series:
    return pd.Series({c: woe_table(df[c], y, max_bins)["iv"].sum() for c in df.columns}).sort_values(
        ascending=False)
