from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import json

# 🔧 path fix
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.sequences.build_sequences import build_user_sequences


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset" / "answers_prepared.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"


def main():
    print(f"Loading dataset from: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows: {len(df)}")
    print(f"Users: {df['user_id'].nunique()}")

    sequences = build_user_sequences(df)

    print(f"Built sequences: {len(sequences)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sequences, f)

    print(f"Saved sequences to: {OUTPUT_PATH}")

    print("\nSample sequence:")
    print(sequences[0])


if __name__ == "__main__":
    main()
