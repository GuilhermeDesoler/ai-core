from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch.utils.data import Dataset

PAD_TOKEN = 0
UNK_TOKEN = 1


@dataclass
class KTQMatrixEncodingMaps:
    concept_to_idx: dict[str, int]
    skill_to_idx: dict[str, int]


def build_qmatrix_id_mappings(training_sequences: list[dict[str, Any]]) -> KTQMatrixEncodingMaps:
    concept_ids = set()
    skill_ids = set()

    for item in training_sequences:
        inp = item["input"]
        tgt = item["target"]

        for concept_list in inp["concept_ids"]:
            concept_ids.update(concept_list)
        for concept_list in tgt["next_concept_ids"]:
            concept_ids.update(concept_list)

        skill_ids.update(inp["skill_ids"])
        skill_ids.update(tgt["next_skill_ids"])

    concept_map = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}
    skill_map = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}

    for i, cid in enumerate(sorted(concept_ids), start=2):
        concept_map[cid] = i
    for i, sid in enumerate(sorted(skill_ids), start=2):
        skill_map[sid] = i

    return KTQMatrixEncodingMaps(concept_map, skill_map)


def encode(values: list[str], mapping: dict[str, int]) -> list[int]:
    return [mapping.get(v, UNK_TOKEN) for v in values]


def pad_1d(values: list[int | float], max_len: int, pad_val: int | float):
    if len(values) >= max_len:
        return values[:max_len]
    return values + [pad_val] * (max_len - len(values))


def pad_2d(values: list[list[int]], max_len: int, inner_max_len: int, pad_val: int):
    values = values[:max_len]
    padded = []
    inner_mask = []
    for row in values:
        row = row[:inner_max_len]
        row_len = len(row)
        padded.append(row + [pad_val] * (inner_max_len - row_len))
        inner_mask.append([1] * row_len + [0] * (inner_max_len - row_len))

    while len(padded) < max_len:
        padded.append([pad_val] * inner_max_len)
        inner_mask.append([0] * inner_max_len)

    return padded, inner_mask


class KTQMatrixNextItemDataset(Dataset):
    def __init__(
        self,
        data: list[dict[str, Any]],
        concept_map: dict[str, int],
        skill_map: dict[str, int],
        max_seq_len: int = 100,
        max_concepts_per_question: int = 8,
    ):
        self.data = data
        self.concept_map = concept_map
        self.skill_map = skill_map
        self.max_seq_len = max_seq_len
        self.max_concepts_per_question = max_concepts_per_question

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inp = item["input"]
        tgt = item["target"]

        s = encode(inp["skill_ids"], self.skill_map)
        ns = encode(tgt["next_skill_ids"], self.skill_map)

        concept_ids = [encode(x, self.concept_map) for x in inp["concept_ids"]]
        next_concept_ids = [encode(x, self.concept_map) for x in tgt["next_concept_ids"]]

        c = [int(x) for x in inp["corrects"]]
        dt = [math.log1p(float(x)) for x in inp["delta_ts"]]
        tr = [math.log1p(float(x)) for x in inp["time_responses"]]
        y = [int(x) for x in tgt["next_corrects"]]

        seq_len = min(len(s), self.max_seq_len)

        concept_ids, concept_mask = pad_2d(concept_ids, self.max_seq_len, self.max_concepts_per_question, PAD_TOKEN)
        next_concept_ids, next_concept_mask = pad_2d(next_concept_ids, self.max_seq_len, self.max_concepts_per_question, PAD_TOKEN)

        s = pad_1d(s, self.max_seq_len, PAD_TOKEN)
        ns = pad_1d(ns, self.max_seq_len, PAD_TOKEN)
        c = pad_1d(c, self.max_seq_len, 0)
        dt = pad_1d(dt, self.max_seq_len, 0.0)
        tr = pad_1d(tr, self.max_seq_len, 0.0)
        y = pad_1d(y, self.max_seq_len, -1)
        mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        return {
            "concept_ids": torch.tensor(concept_ids, dtype=torch.long),
            "concept_mask": torch.tensor(concept_mask, dtype=torch.bool),
            "skill_ids": torch.tensor(s, dtype=torch.long),
            "next_concept_ids": torch.tensor(next_concept_ids, dtype=torch.long),
            "next_concept_mask": torch.tensor(next_concept_mask, dtype=torch.bool),
            "next_skill_ids": torch.tensor(ns, dtype=torch.long),
            "corrects": torch.tensor(c, dtype=torch.float32),
            "delta_ts": torch.tensor(dt, dtype=torch.float32),
            "time_responses": torch.tensor(tr, dtype=torch.float32),
            "targets": torch.tensor(y, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }
