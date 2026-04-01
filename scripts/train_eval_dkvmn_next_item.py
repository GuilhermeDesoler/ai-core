from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_kt.dataset.kt_next_item_dataset import (
    KTNextItemDataset,
    build_id_mappings,
)
from brain_kt.models.dkvmn_next_item import DKVMNNextItemModel
from brain_kt.preprocessing.build_next_item_training_sequences import (
    build_next_item_training_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCES_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "runs_dkvmn"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sequences():
    with open(SEQUENCES_PATH) as f:
        return json.load(f)


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    n = len(data)
    train = data[: int(n * train_ratio)]
    val = data[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    test = data[int(n * (train_ratio + val_ratio)) :]
    return train, val, test


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["targets"])
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    sequences = load_sequences()
    data = build_next_item_training_sequences(sequences)

    train, val, test = split_data(data)

    maps = build_id_mappings(train)

    train_ds = KTNextItemDataset(train, maps.question_to_idx, maps.skill_to_idx)
    val_ds = KTNextItemDataset(val, maps.question_to_idx, maps.skill_to_idx)
    test_ds = KTNextItemDataset(test, maps.question_to_idx, maps.skill_to_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = DKVMNNextItemModel(
        num_questions=len(maps.question_to_idx),
        num_skills=len(maps.skill_to_idx),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["targets"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    test_loss = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS / "dkvmn_next_item.pt")


if __name__ == "__main__":
    main()
