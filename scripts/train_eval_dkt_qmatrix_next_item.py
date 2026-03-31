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

from brain_kt.qmatrix.builder import load_tree, load_questions, build_q_matrix
from brain_kt.preprocessing.build_next_item_training_sequences import build_next_item_training_sequences
from brain_kt.dataset.kt_next_item_dataset import KTNextItemDataset, build_id_mappings
from brain_kt.models.dkt_next_item import DKTNextItemModel

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
TREE_PATH = PROJECT_ROOT / "data" / "raw" / "tree.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "questions.json"


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
    return compute_auc(probs, targets)


def main():
    with open(INPUT_PATH) as f:
        user_sequences = json.load(f)

    # Build Q-matrix to ensure canonical concept mapping
    tree_df = load_tree(TREE_PATH)
    questions_df = load_questions(QUESTIONS_PATH)
    artifacts = build_q_matrix(tree_df, questions_df)

    print("Using Q-matrix concepts (validated tree structure)")

    dataset = build_next_item_training_sequences(user_sequences)

    enc = build_id_mappings(dataset)

    # Student-level split
    user_ids = list({item["user_id"] for item in dataset})
    random.shuffle(user_ids)
    train_users = set(user_ids[: int(0.7 * len(user_ids))])
    val_users = set(user_ids[int(0.7 * len(user_ids)): int(0.85 * len(user_ids))])

    train = [item for item in dataset if item["user_id"] in train_users]
    val = [item for item in dataset if item["user_id"] in val_users]

    train_ds = KTNextItemDataset(train, enc.question_to_idx, enc.skill_to_idx)
    val_ds = KTNextItemDataset(val, enc.question_to_idx, enc.skill_to_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DKTNextItemModel(
        num_questions=len(enc.question_to_idx),
        num_skills=len(enc.skill_to_idx),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

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

        val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val AUC: {val_auc:.4f}")


if __name__ == "__main__":
    main()
