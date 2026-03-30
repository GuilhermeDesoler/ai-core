from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

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
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "dkt"

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sequences(
    sequences: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    items = sequences.copy()
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = items[:n_train]
    val_data = items[n_train:n_train + n_val]
    test_data = items[n_train + n_val:]

    return train_data, val_data, test_data


def compute_accuracy(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()


def compute_auc(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    # Implementação simples sem dependência externa
    probs = probs.detach().cpu()
    targets = targets.detach().cpu()

    pos = probs[targets == 1]
    neg = probs[targets == 0]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    correct_pairs = 0.0
    total_pairs = 0.0

    for p in pos:
        correct_pairs += (p > neg).sum().item()
        correct_pairs += 0.5 * (p == neg).sum().item()
        total_pairs += len(neg)

    return correct_pairs / total_pairs if total_pairs > 0 else 0.5


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch)
            targets = batch["targets"]
            mask = batch["mask"]

            loss = criterion(logits, targets)
            loss = loss * mask
            loss = loss.sum() / mask.sum()

            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            valid_probs = probs[mask]
            valid_targets = targets[mask]

            # ignora padding do target, se existir
            valid_mask = valid_targets >= 0
            valid_probs = valid_probs[valid_mask]
            valid_targets = valid_targets[valid_mask]

            all_probs.append(valid_probs)
            all_targets.append(valid_targets)

    avg_loss = total_loss / len(dataloader)

    probs_cat = torch.cat(all_probs)
    targets_cat = torch.cat(all_targets)

    auc = compute_auc(probs_cat, targets_cat)
    acc = compute_accuracy(probs_cat, targets_cat)

    return avg_loss, auc, acc


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        training_sequences = json.load(f)

    train_sequences, val_sequences, test_sequences = split_sequences(training_sequences)

    print(f"Total sequences: {len(training_sequences)}")
    print(f"Train: {len(train_sequences)}")
    print(f"Val: {len(val_sequences)}")
    print(f"Test: {len(test_sequences)}")

    mappings = build_id_mappings(train_sequences)

    train_dataset = KTDataset(
        training_sequences=train_sequences,
        question_to_idx=mappings.question_to_idx,
        skill_to_idx=mappings.skill_to_idx,
        max_seq_len=100,
    )
    val_dataset = KTDataset(
        training_sequences=val_sequences,
        question_to_idx=mappings.question_to_idx,
        skill_to_idx=mappings.skill_to_idx,
        max_seq_len=100,
    )
    test_dataset = KTDataset(
        training_sequences=test_sequences,
        question_to_idx=mappings.question_to_idx,
        skill_to_idx=mappings.skill_to_idx,
        max_seq_len=100,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = DKTModel(
        num_questions=len(mappings.question_to_idx),
        num_skills=len(mappings.skill_to_idx),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_model_state = None

    print("\nStarting training...\n")

    for epoch in range(5):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch)

            if epoch == 0 and batch_idx == 0:
                preds = torch.sigmoid(logits)
                print(f"Pred mean: {preds.mean().item():.4f}")

            targets = batch["targets"]
            mask = batch["mask"]

            loss = criterion(logits, targets)
            loss = loss * mask
            loss = loss.sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_model_state is None:
        raise RuntimeError("Best model state was not captured.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "dkt_model.pt"
    torch.save(best_model_state, model_path)

    save_json(ARTIFACTS_DIR / "question_to_idx.json", mappings.question_to_idx)
    save_json(ARTIFACTS_DIR / "skill_to_idx.json", mappings.skill_to_idx)

    model.load_state_dict(best_model_state)

    test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion, device)

    print("\n=== FINAL TEST METRICS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    print(f"\nSaved model to: {model_path}")
    print(f"Saved mappings to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
