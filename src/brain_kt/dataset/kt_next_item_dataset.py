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
    q_ids = set()
    s_ids = set()

    for item in training_sequences:
        inp = item["input"]
        tgt = item["target"]
        q_ids.update(inp["question_ids"])
        q_ids.update(tgt["next_question_ids"])
        s_ids.update(inp["skill_ids"])
        s_ids.update(tgt["next_skill_ids"])

    q_map = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}
    s_map = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}

    for i, q in enumerate(sorted(q_ids), start=2):
        q_map[q] = i
    for i, s in enumerate(sorted(s_ids), start=2):
        s_map[s] = i

    return KTEncodingMaps(q_map, s_map)


def encode(values: list[str], mapping: dict[str, int]) -> list[int]:
    return [mapping.get(v, UNK_TOKEN) for v in values]


def pad(values: list[int | float], max_len: int, pad_val: int | float):
    if len(values) >= max_len:
        return values[:max_len]
    return values + [pad_val] * (max_len - len(values))


class KTNextItemDataset(Dataset):
    def __init__(self, data, q_map, s_map, max_seq_len=100):
        self.data = data
        self.q_map = q_map
        self.s_map = s_map
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        inp = item["input"]
        tgt = item["target"]

        q = encode(inp["question_ids"], self.q_map)
        s = encode(inp["skill_ids"], self.s_map)
        nq = encode(tgt["next_question_ids"], self.q_map)
        ns = encode(tgt["next_skill_ids"], self.s_map)

        c = [int(x) for x in inp["corrects"]]
        dt = [math.log1p(float(x)) for x in inp["delta_ts"]]
        tr = [math.log1p(float(x)) for x in inp["time_responses"]]
        y = [int(x) for x in tgt["next_corrects"]]

        seq_len = min(len(q), self.max_seq_len)

        q = pad(q, self.max_seq_len, PAD_TOKEN)
        s = pad(s, self.max_seq_len, PAD_TOKEN)
        nq = pad(nq, self.max_seq_len, PAD_TOKEN)
        ns = pad(ns, self.max_seq_len, PAD_TOKEN)
        c = pad(c, self.max_seq_len, 0)
        dt = pad(dt, self.max_seq_len, 0.0)
        tr = pad(tr, self.max_seq_len, 0.0)
        y = pad(y, self.max_seq_len, -1)

        mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        return {
            "question_ids": torch.tensor(q, dtype=torch.long),
            "skill_ids": torch.tensor(s, dtype=torch.long),
            "next_question_ids": torch.tensor(nq, dtype=torch.long),
            "next_skill_ids": torch.tensor(ns, dtype=torch.long),
            "corrects": torch.tensor(c, dtype=torch.float32),
            "delta_ts": torch.tensor(dt, dtype=torch.float32),
            "time_responses": torch.tensor(tr, dtype=torch.float32),
            "targets": torch.tensor(y, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }
