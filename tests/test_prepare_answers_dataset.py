"""Tests for prepare_answers_dataset pipeline step."""
from __future__ import annotations

import pandas as pd
import pytest


def _make_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_required_columns_present():
    """After preparation the expected columns must exist."""
    from scripts.prepare_answers_dataset import prepare  # type: ignore

    raw = _make_raw([
        {"userId": 1, "questionId": "q1", "skillId": "s1", "correct": 1, "answeredAt": "2024-01-01"},
        {"userId": 1, "questionId": "q2", "skillId": "s1", "correct": 0, "answeredAt": "2024-01-02"},
    ])
    result = prepare(raw)
    for col in ("userId", "questionId", "skillId", "correct"):
        assert col in result.columns, f"Missing column: {col}"


def test_correct_values_are_binary():
    from scripts.prepare_answers_dataset import prepare  # type: ignore

    raw = _make_raw([
        {"userId": 1, "questionId": "q1", "skillId": "s1", "correct": 1, "answeredAt": "2024-01-01"},
        {"userId": 1, "questionId": "q2", "skillId": "s2", "correct": 0, "answeredAt": "2024-01-02"},
    ])
    result = prepare(raw)
    assert set(result["correct"].unique()).issubset({0, 1})
