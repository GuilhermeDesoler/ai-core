from __future__ import annotations

from typing import Any
import math

import torch
from torch.utils.data import Dataset


PAD = 0
UNK = 1


class KTTopicH3Dataset(Dataset):
    def __init__(self, data, h3_map, max_seq_len=100):
        self.data = data
        self.h3_map = h3_map
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def encode(self, values):
        return [self.h3_map.get(v, UNK) for v in values]

    def pad(self, values, pad_val):
        if len(values) >= self.max_seq_len:
            return values[: self.max_seq_len]
        return values + [pad_val] * (self.max_seq_len - len(values))

    def __getitem__(self, idx):
        item = self.data[idx]

        h3 = self.encode(item["input"]["h3_ids"])
        nh3 = self.encode(item["target"]["next_h3_ids"])

        c = [int(x) for x in item["input"]["corrects"]]
        dt = [math.log1p(float(x)) for x in item["input"]["delta_ts"]]
        tr = [math.log1p(float(x)) for x in item["input"]["time_responses"]]
        y = [int(x) for x in item["target"]["next_corrects"]]

        seq_len = min(len(h3), self.max_seq_len)

        h3 = self.pad(h3, PAD)
        nh3 = self.pad(nh3, PAD)
        c = self.pad(c, 0)
        dt = self.pad(dt, 0.0)
        tr = self.pad(tr, 0.0)
        y = self.pad(y, -1)

        mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        return {
            "h3_ids": torch.tensor(h3, dtype=torch.long),
            "next_h3_ids": torch.tensor(nh3, dtype=torch.long),
            "corrects": torch.tensor(c, dtype=torch.float32),
            "delta_ts": torch.tensor(dt, dtype=torch.float32),
            "time_responses": torch.tensor(tr, dtype=torch.float32),
            "targets": torch.tensor(y, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }
