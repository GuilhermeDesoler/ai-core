from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brain_kt.qmatrix.builder import build_qmatrix_from_files

TREE_PATH = PROJECT_ROOT / "data" / "raw" / "tree.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "questions.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "qmatrix"


def main() -> None:
    artifacts = build_qmatrix_from_files(
        tree_json_path=TREE_PATH,
        questions_json_path=QUESTIONS_PATH,
        output_dir=OUTPUT_DIR,
    )
    print("Q-matrix built successfully")
    print(artifacts.metadata)
    print(f"Saved artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
