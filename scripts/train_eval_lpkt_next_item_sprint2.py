from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from brain_kt.dataset.kt_next_item_dataset import KTNextItemDataset, build_id_mappings
from brain_kt.models.lpkt_next_item import LPKTNextItemModel
from brain_kt.preprocessing.build_next_item_training_sequences import (
    build_next_item_training_sequences,
)
from brain_kt.preprocessing.temporal_features_v2 import (
    transform_user_sequences_time_features,
)
from brain_kt.utils.experiment_tracking import create_run_dir, save_json

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs_lpkt_sprint2"

EPOCHS = 30
PATIENCE = 3


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=75)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    set_seed()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = transform_user_sequences_time_features(data)

    dataset = build_next_item_training_sequences(
        data,
        max_seq_len=args.max_seq_len,
        stride=args.stride,
    )

    user_ids = list({item["user_id"] for item in dataset})
    random.shuffle(user_ids)
    n_users = len(user_ids)

    train_users = set(user_ids[: int(0.7 * n_users)])
    val_users = set(user_ids[int(0.7 * n_users) : int(0.85 * n_users)])
    test_users = set(user_ids[int(0.85 * n_users) :])

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
    model = LPKTNextItemModel(
        len(mappings.question_to_idx),
        len(mappings.skill_to_idx),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    all_targets = [c for item in train for c in item["target"]["next_corrects"]]
    n_pos = sum(all_targets)
    n_neg = len(all_targets) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    run_dir = create_run_dir(RUNS_DIR)

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["targets"])
            loss = (loss * batch["mask"]).sum() / batch["mask"].sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_auc, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    test_auc, test_acc = evaluate(model, test_loader, device)

    save_json(
        run_dir / "metrics.json",
        {
            "best_val_auc": best_auc,
            "test_auc": test_auc,
            "test_acc": test_acc,
            "max_seq_len": args.max_seq_len,
            "stride": args.stride,
            "dropout": args.dropout,
        },
    )

    print(f"Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
