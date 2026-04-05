from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from brain_kt.dataset.kt_next_item_dataset import (
    KTNextItemDataset,
    build_id_mappings,
)
from brain_kt.models.dkvmn_next_item import DKVMNNextItemModel
from brain_kt.preprocessing.build_next_item_training_sequences import (
    build_next_item_training_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
)
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs_dkvmn"

EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequences():
    with open(SEQUENCES_PATH) as f:
        return json.load(f)


def compute_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    if len(set(targets_np)) < 2:
        return 0.5
    return float(roc_auc_score(targets_np, probs_np))


def evaluate(model, loader, criterion, device):
    model.eval()
    all_probs, all_targets = [], []
    total_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["targets"])

            mask = batch["mask"]
            batch_loss = (loss * mask).sum() / mask.sum()
            weight = mask.sum().item()

            total_loss += batch_loss.item() * weight
            total_weight += weight

            probs = torch.sigmoid(logits)
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

    print("\n[STEP 1] Carregando dados...")
    sequences = load_sequences()
    data = build_next_item_training_sequences(sequences, max_seq_len=100, stride=50)
    print(f"[OK] Dataset construído | tamanho: {len(data)}")

    print("\n[STEP 2] Split por usuário...")
    user_ids = list({item["user_id"] for item in data})
    random.shuffle(user_ids)
    n_users = len(user_ids)

    train_users = set(user_ids[: int(0.7 * n_users)])
    val_users = set(user_ids[int(0.7 * n_users) : int(0.85 * n_users)])
    test_users = set(user_ids[int(0.85 * n_users) :])

    train = [item for item in data if item["user_id"] in train_users]
    val = [item for item in data if item["user_id"] in val_users]
    test = [item for item in data if item["user_id"] in test_users]

    print(f"[OK] Train: {len(train)} amostras | {len(train_users)} usuários")
    print(f"[OK] Val:   {len(val)} amostras | {len(val_users)} usuários")
    print(f"[OK] Test:  {len(test)} amostras | {len(test_users)} usuários")

    print("\n[STEP 3] Construindo mappings...")
    maps = build_id_mappings(train)
    print(
        f"[OK] Questions: {len(maps.question_to_idx)} | Skills: {len(maps.skill_to_idx)}"
    )

    train_ds = KTNextItemDataset(train, maps.question_to_idx, maps.skill_to_idx)
    val_ds = KTNextItemDataset(val, maps.question_to_idx, maps.skill_to_idx)
    test_ds = KTNextItemDataset(test, maps.question_to_idx, maps.skill_to_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    print("\n[STEP 4] Inicializando modelo...")
    model = DKVMNNextItemModel(
        num_questions=len(maps.question_to_idx),
        num_skills=len(maps.skill_to_idx),
    ).to(DEVICE)
    print(f"[OK] Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Pos weight para balanceamento
    all_targets = [c for item in train for c in item["target"]["next_corrects"]]
    n_pos = sum(all_targets)
    n_neg = len(all_targets) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / f"run_{len(list(RUNS_DIR.iterdir()))+1:03d}"
    run_dir.mkdir()
    print(f"[OK] Run directory: {run_dir}")

    print("\n[STEP 5] Iniciando treinamento...")
    print("-" * 70)

    best_val_auc = 0.0
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0.0
        train_probs, train_targets = [], []

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["targets"])

            mask = batch["mask"]
            batch_loss = (loss * mask).sum() / mask.sum()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            weight = mask.sum().item()
            train_loss_sum += batch_loss.item() * weight
            train_weight += weight

            probs = torch.sigmoid(logits)
            train_probs.append(probs[mask].detach())
            train_targets.append(batch["targets"][mask].detach())

        train_probs = torch.cat(train_probs)
        train_targets = torch.cat(train_targets)
        train_loss = train_loss_sum / max(train_weight, 1.0)
        train_auc = compute_auc(train_probs, train_targets)
        train_acc = ((train_probs > 0.5) == train_targets).float().mean().item()

        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "model.pt")
            print(f"  ✓ Novo melhor modelo! AUC: {val_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(
                    f"\n⏹️ Early stopping na epoch {epoch+1} (sem melhora há {PATIENCE} epochs)"
                )
                break

    print("\n[STEP 6] Avaliando no teste...")
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion, DEVICE)

    print("\n" + "=" * 70)
    print("📊 RESULTADOS FINAIS")
    print("=" * 70)
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(
        f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}"
    )
    print(f"\nSaved run to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
