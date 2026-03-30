from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.sequences.build_training_sequences import build_training_sequences


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "training_sequences.json"


def main():
    print(f"Loading sequences from: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    print(f"Sequences: {len(sequences)}")

    training_sequences = build_training_sequences(sequences)

    print(f"Training sequences: {len(training_sequences)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(training_sequences, f)

    print(f"Saved training data to: {OUTPUT_PATH}")

    print("\nSample:")
    print(training_sequences[0])


if __name__ == "__main__":
    main()
