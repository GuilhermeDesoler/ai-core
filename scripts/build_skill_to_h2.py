import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "questions.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mappings" / "skill_to_h2.json"


def extract_h2(path_str: str) -> str | None:
    if not isinstance(path_str, str):
        return None

    parts = [p.strip() for p in path_str.split(">")]

    if len(parts) < 2:
        return None

    return parts[1]  # H2


def main():
    df = pd.read_parquet(INPUT_PATH)

    mapping = {}

    for _, row in df.iterrows():
        skill_id = str(row["subject_id"])
        path = row["subject_path_names"]

        h2 = extract_h2(path)

        if h2 is None:
            continue

        mapping[skill_id] = h2

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(mapping)} mappings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
