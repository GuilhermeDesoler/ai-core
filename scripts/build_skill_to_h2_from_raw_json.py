from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "questions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mappings" / "skill_to_h2.json"


def extract_h2_label(subject_accumulated_names: list[str]) -> str | None:
    if not isinstance(subject_accumulated_names, list):
        return None

    names = [str(x).strip() for x in subject_accumulated_names if str(x).strip()]
    if len(names) < 2:
        return None

    return f"{names[0]} > {names[1]}"


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    skill_to_h2: dict[str, str] = {}
    skipped = 0

    for row in questions:
        subject_id = row.get("subjectId") or row.get("subject_id")
        accumulated_names = row.get("subjectAccumulatedNames") or row.get("subject_accumulated_names")

        if not subject_id:
            skipped += 1
            continue

        h2_label = extract_h2_label(accumulated_names)
        if h2_label is None:
            skipped += 1
            continue

        skill_to_h2[str(subject_id)] = h2_label

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(skill_to_h2, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(skill_to_h2)} mappings to: {OUTPUT_PATH}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
