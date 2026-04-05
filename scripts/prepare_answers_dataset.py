from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from data.pipelines.build_answers_dataset import build_answers_dataset

RAW_ANSWERS_PATH = PROJECT_ROOT / "data" / "raw" / "answers.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"
OUTPUT_PATH = OUTPUT_DIR / "answers_prepared.parquet"


def load_answers_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def main() -> None:
    print(f"Loading answers from: {RAW_ANSWERS_PATH}")

    answers_df = load_answers_json(RAW_ANSWERS_PATH)
    print(f"Raw rows: {len(answers_df)}")

    prepared_df, report = build_answers_dataset(
        answers_df=answers_df,
        min_interactions=10,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = OUTPUT_DIR / "answers_prepared.csv"
    prepared_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== ANSWERS REPORT ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nPrepared dataset preview:")
    print(prepared_df.head())

    print(f"\nSaved prepared dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
