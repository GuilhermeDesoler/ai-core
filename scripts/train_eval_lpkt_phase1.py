import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_kt.dataset.kt_next_item_dataset import KTNextItemDataset, build_id_mappings
from brain_kt.models.lpkt_phase1 import LPKTPhase1
from brain_kt.preprocessing.sequence_feature_engineering import transform_sequences
from brain_kt.preprocessing.build_next_item_training_sequences import build_next_item_training_sequences

INPUT = Path("data/processed/sequences/user_sequences.json")

with open(INPUT) as f:
    raw = json.load(f)

raw = transform_sequences(raw)
data = build_next_item_training_sequences(raw)

mappings = build_id_mappings(data)
ds = KTNextItemDataset(data, mappings.question_to_idx, mappings.skill_to_idx)
loader = DataLoader(ds, batch_size=16, shuffle=True)

model = LPKTPhase1(len(mappings.question_to_idx), len(mappings.skill_to_idx))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# weighted BCE
targets = [c for d in data for c in d["target"]["next_corrects"]]
pos = sum(targets)
neg = len(targets) - pos
pos_weight = torch.tensor([neg / max(pos,1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

for epoch in range(10):
    for batch in loader:
        logits = model(batch)
        loss = criterion(logits, batch["targets"])
        loss = (loss * batch["mask"]).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"epoch {epoch} done")
