from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_kt.dataset.kt_next_item_dataset import KTNextItemDataset, build_id_mappings
from brain_kt.models.lpkt_next_item import LPKTNextItemModel
from brain_kt.preprocessing.build_next_item_training_sequences import build_next_item_training_sequences
from brain_kt.utils.experiment_tracking import create_run_dir, save_json

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs_lpkt"


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def compute_auc(probs, targets):
    probs = probs.detach().cpu()
    targets = targets.detach().cpu()
    pos = probs[targets == 1]
    neg = probs[targets == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    correct = 0
    total = 0
    for p in pos:
        correct += (p > neg).sum().item()
        total += len(neg)
    return correct / total


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits)
            mask = batch["mask"]
            all_probs.append(probs[mask])
            all_targets.append(batch["targets"][mask])

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    auc = compute_auc(probs, targets)
    acc = ((probs > 0.5) == targets).float().mean().item()

    return auc, acc


def main():
    set_seed()

    with open(INPUT_PATH) as f:
        data = json.load(f)

    dataset = build_next_item_training_sequences(data, max_seq_len=100, stride=50)

    random.shuffle(dataset)
    n = len(dataset)

    train = dataset[: int(0.7 * n)]
    val = dataset[int(0.7 * n): int(0.85 * n)]
    test = dataset[int(0.85 * n):]

    mappings = build_id_mappings(train)

    train_ds = KTNextItemDataset(train, mappings.question_to_idx, mappings.skill_to_idx)
    val_ds = KTNextItemDataset(val, mappings.question_to_idx, mappings.skill_to_idx)
    test_ds = KTNextItemDataset(test, mappings.question_to_idx, mappings.skill_to_idx)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPKTNextItemModel(len(mappings.question_to_idx), len(mappings.skill_to_idx)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    run_dir = create_run_dir(RUNS_DIR)

    best_auc = 0

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["targets"])
            loss = (loss * batch["mask"]).sum() / batch["mask"].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_auc, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), run_dir / "model.pt")

    test_auc, test_acc = evaluate(model, test_loader, device)

    save_json(run_dir / "metrics.json", {
        "best_val_auc": best_auc,
        "test_auc": test_auc,
        "test_acc": test_acc
    })

    print(f"Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
