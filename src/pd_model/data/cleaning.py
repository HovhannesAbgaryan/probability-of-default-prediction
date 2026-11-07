"""Repairs specific to the supplied training extract (EDA notebook 01, §1.1, §7.1, §8).

These steps fix known corruption in *this file* and run once, before the split. They are not part of
the scoring-time preprocessing (see ``pd_model.features.preprocessing``): production applications do not
carry these errors, and imputation / winsorisation / encoding are learned by the preprocessor instead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Validation-only columns: used for segment analysis, never as model features.
META_COLUMNS = ["row_id", "block", "income_reconstructed", "inconsistent_record"]


@dataclass
class CleaningReport:
    rows: int
    dropped_leaky: list[str]
    loans_divided_by_10: int
    incomes_reconstructed: int
    inconsistent_records: int
    max_identity_error: float
    train_rows: int = 0
    test_rows: int = 0
    train_default_rate: float = 0.0
    test_default_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def clean_raw(raw: pd.DataFrame, *, leaky_columns: list[str], block_boundary: int, income_cap: float,
              age_placeholder: float = 99) -> tuple[pd.DataFrame, CleaningReport]:
    """Return the repaired frame (raw feature columns + target + META_COLUMNS) and a report."""
    df = raw.reset_index(drop=True)  # original index is non-unique: two concatenated blocks
    df.insert(0, "row_id", df.index)
    df["block"] = np.where(df.row_id < block_boundary, 1, 2)

    df = df.drop(columns=[c for c in leaky_columns if c in df.columns])

    # dti == loan_amount / income exactly in the clean block. In block 2, 80 loans were multiplied
    # by 10 (detected via the identity, or via an implied income above the cap when income is missing),
    # and every income below ~16.8k was deleted (rebuilt exactly as loan / dti). Order matters: loans first.
    ratio = df.loan_amount / (df.dti * df.income)
    implied_income = df.loan_amount / df.dti
    loan_x10 = np.isclose(ratio, 10, rtol=1e-4) | (df.income.isna() & (implied_income > income_cap))
    df.loc[loan_x10, "loan_amount"] /= 10

    df["income_reconstructed"] = df.income.isna().astype(int)
    df.loc[df.income.isna(), "income"] = df.loan_amount / df.dti
    identity_error = float((df.loan_amount / df.income - df.dti).abs().max())

    # Logical contradictions (equally frequent in both blocks): kept, flagged for validation segmentation.
    # A placeholder age (99) never triggers the age rules, same as an unknown age would not.
    age = df.age.fillna(age_placeholder)
    dpd_sum = df.past_due_30 + df.past_due_60 + df.past_due_90
    df["inconsistent_record"] = ((df.employment_years > age - 14)
                                 | (df.client_history_in_bank > (age - 18) * 12)
                                 | (dpd_sum > df.num_loans)).astype(int)

    report = CleaningReport(rows=len(df), dropped_leaky=list(leaky_columns),
                            loans_divided_by_10=int(loan_x10.sum()),
                            incomes_reconstructed=int(df.income_reconstructed.sum()),
                            inconsistent_records=int(df.inconsistent_record.sum()),
                            max_identity_error=identity_error)
    return df, report


def split(df: pd.DataFrame, target: str, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random hold-out. The data has no dates, so an out-of-time split is impossible."""
    train, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    return train.copy(), test.copy()


def run_cleaning(raw: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    c = cfg.cleaning
    df, report = clean_raw(raw, leaky_columns=c["leaky_columns"], block_boundary=c["block_boundary"],
                           income_cap=c["income_cap"], age_placeholder=cfg.preprocessing["age_placeholder"])
    train, test = split(df, cfg.target, c["test_size"], cfg.seed)
    assert train.row_id.is_unique and not set(train.row_id) & set(test.row_id)
    assert not set(c["leaky_columns"]) & set(train.columns)
    report.train_rows, report.test_rows = len(train), len(test)
    report.train_default_rate = float(train[cfg.target].mean())
    report.test_default_rate = float(test[cfg.target].mean())
    return train, test, report
