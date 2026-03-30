from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brain_kt.dataset.kt_dataset import KTDataset, build_id_mappings


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "training_sequences.json"


def main() -> None:
    print(f"Loading training sequences from: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        training_sequences = json.load(f)

    print(f"Training sequences: {len(training_sequences)}")

    mappings = build_id_mappings(training_sequences)

    print(f"Questions mapped: {len(mappings.question_to_idx)}")
    print(f"Skills mapped: {len(mappings.skill_to_idx)}")

    dataset = KTDataset(
        training_sequences=training_sequences,
        question_to_idx=mappings.question_to_idx,
        skill_to_idx=mappings.skill_to_idx,
        max_seq_len=100,
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print("\nSample keys:")
    print(sample.keys())

    print("\nSample shapes:")
    for key, value in sample.items():
        print(f"{key}: {tuple(value.shape)}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    print("\nBatch shapes:")
    for key, value in batch.items():
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
