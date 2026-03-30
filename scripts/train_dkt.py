from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brain_kt.dataset.kt_dataset import KTDataset, build_id_mappings
from brain_kt.models.dkt import DKTModel


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "training_sequences.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        training_sequences = json.load(f)

    mappings = build_id_mappings(training_sequences)

    dataset = KTDataset(
        training_sequences=training_sequences,
        question_to_idx=mappings.question_to_idx,
        skill_to_idx=mappings.skill_to_idx,
        max_seq_len=100,
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DKTModel(
        num_questions=len(mappings.question_to_idx),
        num_skills=len(mappings.skill_to_idx),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    print("\nStarting training...\n")

    for epoch in range(5):
        model.train()

        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch)
            targets = batch["targets"]

            if epoch == 0 and batch_idx == 0:
                preds = torch.sigmoid(logits)
                print(f"Pred mean: {preds.mean().item():.4f}")

            mask = batch["mask"]

            loss = criterion(logits, targets)

            loss = loss * mask

            loss = loss.sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
