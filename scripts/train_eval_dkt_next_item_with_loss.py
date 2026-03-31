from __future__ import annotations

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
from brain_kt.models.dkt_next_item import DKTNextItemModel
from brain_kt.preprocessing.build_next_item_training_sequences import build_next_item_training_sequences
from brain_kt.utils.experiment_tracking import create_run_dir, save_json

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs_with_loss" / "dkt_next_item"

EPOCHS = 30
PATIENCE = 5


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def compute_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    if len(set(targets_np)) < 2:
        return 0.5
    return float(roc_auc_score(targets_np, probs_np))


def evaluate(model, loader, device, criterion):
    model.eval()
    all_probs, all_targets = [], []
    total_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["targets"])
            weight = batch["mask"].sum().item()
            batch_loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
            total_loss += batch_loss.item() * weight
            total_weight += weight

            probs = torch.sigmoid(logits)
            mask = batch["mask"]
            all_probs.append(probs[mask])
            all_targets.append(batch["targets"][mask])

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    auc = compute_auc(probs, targets)
    acc = ((probs > 0.5) == targets).float().mean().item()
    avg_loss = total_loss / max(total_weight, 1.0)
    return avg_loss, auc, acc


def main():
    set_seed()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = build_next_item_training_sequences(data, max_seq_len=100, stride=50)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    all_targets = [c for item in train for c in item["target"]["next_corrects"]]
    n_pos = sum(all_targets)
    n_neg = len(all_targets) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    run_dir = create_run_dir(RUNS_DIR)
    best_val_auc = 0.0
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["targets"])
            batch_loss = (loss * batch["mask"]).sum() / batch["mask"].sum()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            weight = batch["mask"].sum().item()
            train_loss_sum += batch_loss.item() * weight
            train_weight += weight

        train_loss = train_loss_sum / max(train_weight, 1.0)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_auc)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping na epoch {epoch+1} (sem melhora há {PATIENCE} epochs)")
                break

    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    test_loss, test_auc, test_acc = evaluate(model, test_loader, device, criterion)

    save_json(run_dir / "metrics.json", {
        "best_val_auc": best_val_auc,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "test_acc": test_acc,
    })
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
