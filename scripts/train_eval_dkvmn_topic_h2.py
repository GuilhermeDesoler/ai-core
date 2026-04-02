from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from tkinter.filedialog import test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from brain_kt.dataset.kt_topic_h2_dataset import KTTopicH2Dataset
from brain_kt.models.dkvmn_topic_h2 import DKVMNTopicH2Model
from brain_kt.preprocessing.build_topic_h2_training_sequences import build_h2_topic_training_sequences
from brain_kt.utils.experiment_tracking import create_run_dir, save_json

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
H2_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "mappings" / "skill_to_h2.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs_dkvmn_h2"
EPOCHS = 10
PATIENCE = 4


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def compute_auc(probs, targets):
    probs = probs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    if len(set(targets)) < 2:
        return 0.5
    return float(roc_auc_score(targets, probs))


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
            batch_loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
            weight = batch["mask"].sum().item()
            total_loss += batch_loss.item() * weight
            total_weight += weight
            probs = torch.sigmoid(logits)
            mask = batch["mask"]
            all_probs.append(probs[mask])
            all_targets.append(batch["targets"][mask])
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    return total_loss / max(total_weight, 1.0), compute_auc(probs, targets), ((probs > 0.5) == targets).float().mean().item()


def build_h2_map(data):
    unique = sorted(set(data))
    mapping = {"<PAD>": 0, "<UNK>": 1}
    for i, k in enumerate(unique, start=2):
        mapping[k] = i
    return mapping


def main():
    set_seed()
    print("\n[STEP 1] Carregando dados...")
    with open(INPUT_PATH) as f:
        user_sequences = json.load(f)
    with open(H2_MAP_PATH) as f:
        skill_to_h2 = json.load(f)
    print("[OK] Dados carregados")
    print("\n[STEP 2] Construindo dataset...")
    dataset = build_h2_topic_training_sequences(user_sequences, skill_to_h2, max_seq_len=30, stride=25)
    print(f"[OK] Dataset pronto | tamanho: {len(dataset)}")
    all_h2 = [h for item in dataset for h in item["input"]["h2_ids"]]
    h2_map = build_h2_map(all_h2)
    user_ids = list({item["user_id"] for item in dataset})
    random.shuffle(user_ids)
    n_users = len(user_ids)
    train_users = set(user_ids[: int(0.7 * n_users)])
    val_users = set(user_ids[int(0.7 * n_users): int(0.85 * n_users)])
    test_users = set(user_ids[int(0.85 * n_users):])
    train = [item for item in dataset if item["user_id"] in train_users]
    val = [item for item in dataset if item["user_id"] in val_users]
    test = [item for item in dataset if item["user_id"] in test_users]
    print("\n[STEP 3] Fazendo split train/val/test...")
    print(f"[OK] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    train_ds = KTTopicH2Dataset(train, h2_map, max_seq_len=50)
    val_ds = KTTopicH2Dataset(val, h2_map, max_seq_len=50)
    test_ds = KTTopicH2Dataset(test, h2_map, max_seq_len=50)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKVMNTopicH2Model(len(h2_map)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    all_targets = [c for item in train for c in item["target"]["next_corrects"]]
    n_pos = sum(all_targets)
    n_neg = len(all_targets) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    run_dir = create_run_dir(RUNS_DIR)
    print("\n[STEP 4] Iniciando treino...")
    history = []
    best_val_auc = 0.0
    best_val_loss = float("inf")
    no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0.0
        train_probs, train_targets = [], []
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
            probs = torch.sigmoid(logits)
            mask = batch["mask"]
            train_probs.append(probs[mask].detach())
            train_targets.append(batch["targets"][mask].detach())
        train_probs = torch.cat(train_probs)
        train_targets = torch.cat(train_targets)
        train_loss = train_loss_sum / max(train_weight, 1.0)
        train_auc = compute_auc(train_probs, train_targets)
        train_acc = ((train_probs > 0.5) == train_targets).float().mean().item()
        val_loss, val_auc, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_auc)
        history.append({"epoch": epoch+1, "train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc, "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']})
        print(f"[EPOCH {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "model.pt")
            print(f"[BEST MODEL] Epoch {epoch+1} | Val AUC: {val_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    print("\n[STEP 5] Avaliando no teste...")
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    test_loss, test_auc, test_acc = evaluate(model, test_loader, device, criterion)
    save_json(run_dir / "metrics.json", {"best_val_auc": best_val_auc, "best_val_loss": best_val_loss, "test_loss": test_loss, "test_auc": test_auc, "test_acc": test_acc})
    save_json(run_dir / "history.json", history)
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Saved run to {run_dir}")
    print("\n[FINALIZADO]")


if __name__ == "__main__":
    main()
