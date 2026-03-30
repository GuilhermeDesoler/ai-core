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

from brain_kt.dataset.kt_topic_h2_dataset import KTTopicH2Dataset
from brain_kt.models.dkt_topic_h2 import DKTTopicH2Model
from brain_kt.preprocessing.build_topic_h2_training_sequences import build_h2_topic_training_sequences
from brain_kt.utils.experiment_tracking import create_run_dir, save_json

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
H2_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "mappings" / "skill_to_h2.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs_h2"


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


def build_h2_map(data):
    unique = sorted(set(data))
    mapping = {"<PAD>": 0, "<UNK>": 1}
    for i, k in enumerate(unique, start=2):
        mapping[k] = i
    return mapping


def main():
    set_seed()

    with open(INPUT_PATH) as f:
        user_sequences = json.load(f)

    if not H2_MAP_PATH.exists():
        raise FileNotFoundError("Run build_skill_to_h2_from_raw_json.py first")

    with open(H2_MAP_PATH) as f:
        skill_to_h2 = json.load(f)

    dataset = build_h2_topic_training_sequences(user_sequences, skill_to_h2)

    all_h2 = [h for item in dataset for h in item["input"]["h2_ids"]]
    h2_map = build_h2_map(all_h2)

    random.shuffle(dataset)
    n = len(dataset)

    train = dataset[: int(0.7 * n)]
    val = dataset[int(0.7 * n): int(0.85 * n)]
    test = dataset[int(0.85 * n):]

    train_ds = KTTopicH2Dataset(train, h2_map)
    val_ds = KTTopicH2Dataset(val, h2_map)
    test_ds = KTTopicH2Dataset(test, h2_map)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKTTopicH2Model(len(h2_map)).to(device)

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
