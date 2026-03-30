from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch.utils.data import Dataset


PAD_TOKEN = 0
UNK_TOKEN = 1


@dataclass
class KTEncodingMaps:
    question_to_idx: dict[str, int]
    skill_to_idx: dict[str, int]


def build_id_mappings(training_sequences: list[dict[str, Any]]) -> KTEncodingMaps:
    question_ids: set[str] = set()
    skill_ids: set[str] = set()

    for item in training_sequences:
        input_seq = item["input"]
        question_ids.update(input_seq["question_ids"])
        skill_ids.update(input_seq["skill_ids"])

    question_to_idx = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}
    skill_to_idx = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}

    for idx, question_id in enumerate(sorted(question_ids), start=2):
        question_to_idx[question_id] = idx

    for idx, skill_id in enumerate(sorted(skill_ids), start=2):
        skill_to_idx[skill_id] = idx

    return KTEncodingMaps(
        question_to_idx=question_to_idx,
        skill_to_idx=skill_to_idx,
    )


def encode_sequence(
    values: list[str],
    mapping: dict[str, int],
) -> list[int]:
    return [mapping.get(value, UNK_TOKEN) for value in values]


def pad_sequence(
    values: list[int | float],
    max_seq_len: int,
    pad_value: int | float,
) -> list[int | float]:
    if len(values) >= max_seq_len:
        return values[:max_seq_len]

    padding = [pad_value] * (max_seq_len - len(values))
    return values + padding


class KTDataset(Dataset):
    def __init__(
        self,
        training_sequences: list[dict[str, Any]],
        question_to_idx: dict[str, int],
        skill_to_idx: dict[str, int],
        max_seq_len: int = 100,
    ) -> None:
        self.training_sequences = training_sequences
        self.question_to_idx = question_to_idx
        self.skill_to_idx = skill_to_idx
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.training_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.training_sequences[idx]

        input_seq = item["input"]
        target_seq = item["target"]

        question_ids = encode_sequence(
            input_seq["question_ids"],
            self.question_to_idx,
        )
        skill_ids = encode_sequence(
            input_seq["skill_ids"],
            self.skill_to_idx,
        )

        corrects = [int(x) for x in input_seq["corrects"]]
        delta_ts = [math.log1p(float(x)) for x in input_seq["delta_ts"]]
        time_responses = [math.log1p(float(x)) for x in input_seq["time_responses"]]
        targets = [int(x) for x in target_seq["corrects"]]

        seq_len = min(len(question_ids), self.max_seq_len)

        question_ids = pad_sequence(question_ids, self.max_seq_len, PAD_TOKEN)
        skill_ids = pad_sequence(skill_ids, self.max_seq_len, PAD_TOKEN)
        corrects = pad_sequence(corrects, self.max_seq_len, 0)
        delta_ts = pad_sequence(delta_ts, self.max_seq_len, 0.0)
        time_responses = pad_sequence(time_responses, self.max_seq_len, 0.0)
        targets = pad_sequence(targets, self.max_seq_len, -1)

        mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        return {
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "skill_ids": torch.tensor(skill_ids, dtype=torch.long),
            "corrects": torch.tensor(corrects, dtype=torch.float32),
            "delta_ts": torch.tensor(delta_ts, dtype=torch.float32),
            "time_responses": torch.tensor(time_responses, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }
