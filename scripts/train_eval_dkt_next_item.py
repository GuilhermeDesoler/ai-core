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
from sklearn.metrics import roc_auc_score

from brain_kt.dataset.kt_next_item_dataset import KTNextItemDataset, build_id_mappings
from brain_kt.models.dkt_next_item import DKTNextItemModel
from brain_kt.preprocessing.build_next_item_training_sequences import build_next_item_training_sequences
from brain_kt.utils.experiment_tracking import create_run_dir, save_json

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def compute_auc(probs, targets):
    probs = probs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    if len(set(targets)) < 2:
        return 0.5
    return float(roc_auc_score(targets, probs))


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

    # Student-level split: evita data leakage entre janelas do mesmo usuário
    user_ids = list({item["user_id"] for item in dataset})
    random.shuffle(user_ids)
    n_users = len(user_ids)
    train_users = set(user_ids[: int(0.7 * n_users)])
    val_users = set(user_ids[int(0.7 * n_users): int(0.85 * n_users)])
    test_users = set(user_ids[int(0.85 * n_users):])
    train = [item for item in dataset if item["user_id"] in train_users]
    val = [item for item in dataset if item["user_id"] in val_users]
    test = [item for item in dataset if item["user_id"] in test_users]

    mappings = build_id_mappings(train)

    train_ds = KTNextItemDataset(train, mappings.question_to_idx, mappings.skill_to_idx)
    val_ds = KTNextItemDataset(val, mappings.question_to_idx, mappings.skill_to_idx)
    test_ds = KTNextItemDataset(test, mappings.question_to_idx, mappings.skill_to_idx)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKTNextItemModel(len(mappings.question_to_idx), len(mappings.skill_to_idx)).to(device)

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
