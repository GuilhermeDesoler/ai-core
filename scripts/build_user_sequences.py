from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "dataset" / "answers_prepared.csv"
INPUT_PARQUET = (
    PROJECT_ROOT / "data" / "processed" / "dataset" / "answers_prepared.parquet"
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"


def load_answers() -> pd.DataFrame:
    if INPUT_CSV.exists():
        return pd.read_csv(INPUT_CSV)
    if INPUT_PARQUET.exists():
        return pd.read_parquet(INPUT_PARQUET)

    raise FileNotFoundError("answers_prepared not found")


def build_user_sequences(df: pd.DataFrame):
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    sequences = []

    for user_id, group in df.groupby("user_id"):
        if len(group) < 2:
            continue

        sequences.append(
            {
                "user_id": str(user_id),
                "question_ids": group["question_id"].astype(str).tolist(),
                "skill_ids": group["skill_id"].astype(str).tolist(),
                "corrects": group["correct"].astype(int).tolist(),
                "delta_ts": group["delta_t"].astype(int).tolist(),
                "time_responses": group["time_response"].astype(int).tolist(),
            }
        )

    return sequences


def main():
    df = load_answers()

    sequences = build_user_sequences(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sequences, f)

    print(f"Saved {len(sequences)} sequences to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
