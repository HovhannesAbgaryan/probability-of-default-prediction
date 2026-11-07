from __future__ import annotations

import numpy as np

from pd_model.data.cleaning import clean_raw, split


def corrupt(df):
    raw = df.drop(columns="default").assign(recovery_flag_after_90d=0, default=df["default"])
    raw.index = list(range(400)) + list(range(len(raw) - 400))  # two concatenated blocks
    raw.iloc[450, raw.columns.get_loc("loan_amount")] *= 10     # loan ×10 (identity off by 10)
    raw.iloc[460, raw.columns.get_loc("income")] = np.nan       # deleted income
    return raw


def test_clean_raw_repairs_extract(frame):
    raw = corrupt(frame)
    out, report = clean_raw(raw, leaky_columns=["recovery_flag_after_90d"], block_boundary=400, income_cap=30_000)
    assert "recovery_flag_after_90d" not in out
    assert out.row_id.is_unique and set(out.block) == {1, 2} and (out.block[:400] == 1).all()
    assert report.loans_divided_by_10 == 1 and report.incomes_reconstructed == 1
    assert np.isclose(out.loan_amount[450], frame.loan_amount[450])
    assert np.isclose(out.income[460], frame.income[460])
    assert report.max_identity_error < 1e-8
    assert out.income_reconstructed.sum() == 1


def test_split_is_stratified_and_disjoint(frame):
    frame = frame.assign(row_id=range(len(frame)))
    tr, te = split(frame, "default", 0.3, 0)
    assert len(te) == 180 and not set(tr.row_id) & set(te.row_id)
    assert abs(tr["default"].mean() - te["default"].mean()) < 0.01
