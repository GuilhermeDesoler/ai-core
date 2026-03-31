"""Tests for PyTorch KT datasets."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brain_kt.dataset.kt_next_item_dataset import KTNextItemDataset, build_id_mappings


def _make_items(n: int = 4) -> list[dict]:
    return [
        {
            "user_id": f"u{i}",
            "input": {
                "question_ids": ["q0", "q1", "q2"],
                "skill_ids": ["s0", "s1", "s0"],
                "corrects": [1, 0, 1],
                "delta_ts": [0.0, 1.0, 2.0],
                "time_responses": [1.0, 1.0, 1.0],
            },
            "target": {
                "next_question_ids": ["q1", "q2", "q0"],
                "next_skill_ids": ["s1", "s0", "s0"],
                "next_corrects": [0, 1, 0],
            },
        }
        for i in range(n)
    ]


def test_dataset_length():
    items = _make_items(6)
    mappings = build_id_mappings(items)
    ds = KTNextItemDataset(items, mappings.question_to_idx, mappings.skill_to_idx)
    assert len(ds) == 6


def test_item_shapes_consistent():
    items = _make_items(3)
    mappings = build_id_mappings(items)
    ds = KTNextItemDataset(items, mappings.question_to_idx, mappings.skill_to_idx)
    sample = ds[0]
    seq_len = sample["targets"].shape[0]
    for key, tensor in sample.items():
        assert tensor.shape[0] == seq_len, f"Shape mismatch for '{key}': {tensor.shape}"


def test_mask_is_boolean_or_binary():
    items = _make_items(2)
    mappings = build_id_mappings(items)
    ds = KTNextItemDataset(items, mappings.question_to_idx, mappings.skill_to_idx)
    mask = ds[0]["mask"]
    assert mask.dtype in (torch.bool, torch.float32, torch.int64)
    assert set(mask.unique().tolist()).issubset({0, 1, True, False})


def test_unknown_question_uses_pad():
    items = _make_items(1)
    mappings = build_id_mappings(items)
    ds = KTNextItemDataset(
        [
            {
                "user_id": "u_new",
                "input": {
                    "question_ids": ["unseen_q"],
                    "skill_ids": ["unseen_s"],
                    "corrects": [1],
                    "delta_ts": [0.0],
                    "time_responses": [1.0],
                },
                "target": {
                    "next_question_ids": ["unseen_q2"],
                    "next_skill_ids": ["unseen_s2"],
                    "next_corrects": [0],
                },
            }
        ],
        mappings.question_to_idx,
        mappings.skill_to_idx,
    )
    sample = ds[0]
    assert sample is not None
