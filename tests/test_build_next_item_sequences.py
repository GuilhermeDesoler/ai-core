"""Tests for build_next_item_training_sequences."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brain_kt.preprocessing.build_next_item_training_sequences import (
    build_next_item_training_sequences,
)


def _make_sequences(n_questions: int = 5) -> list[dict]:
    return [
        {
            "user_id": "u1",
            "question_ids": [f"q{i}" for i in range(n_questions)],
            "skill_ids": [f"s{i}" for i in range(n_questions)],
            "corrects": [i % 2 for i in range(n_questions)],
            "delta_ts": [0.0] * n_questions,
            "time_responses": [1.0] * n_questions,
        }
    ]


def test_returns_list():
    seqs = _make_sequences()
    result = build_next_item_training_sequences(seqs)
    assert isinstance(result, list)
    assert len(result) > 0


def test_each_item_has_required_keys():
    seqs = _make_sequences(10)
    result = build_next_item_training_sequences(seqs)
    for item in result:
        assert "user_id" in item
        assert "input" in item
        assert "target" in item


def test_max_seq_len_respected():
    seqs = _make_sequences(200)
    result = build_next_item_training_sequences(seqs, max_seq_len=50)
    for item in result:
        assert len(item["input"]["question_ids"]) <= 50


def test_short_sequence_produces_single_window():
    seqs = _make_sequences(5)
    result = build_next_item_training_sequences(seqs, max_seq_len=100)
    assert len(result) == 1
    assert result[0]["user_id"] == "u1"
