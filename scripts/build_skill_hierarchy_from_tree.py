"""
build_skill_hierarchy_from_tree.py
===================================
Constrói os mapeamentos skill_to_h2.json e skill_to_h3.json usando
subjectId + tree.json como fonte canônica.

Vantagens sobre a abordagem anterior (subjectAccumulatedNames):
  - Usa a árvore oficial como fonte de verdade hierárquica
  - Independe de encoding ou texto das questões
  - Recupera questões cujo subjectAccumulatedNames divergia da árvore
  - Gera H2 e H3 em uma única passagem

Saídas:
  data/processed/mappings/skill_to_h2.json
  data/processed/mappings/skill_to_h3.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from brain_kt.qmatrix.builder import load_tree, load_questions

QUESTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "questions.json"
TREE_PATH      = PROJECT_ROOT / "data" / "raw" / "tree.json"
OUTPUT_DIR     = PROJECT_ROOT / "data" / "processed" / "mappings"


def build_hierarchy_mappings(
    questions_path: Path,
    tree_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    tree_df    = load_tree(tree_path)
    questions_df = load_questions(questions_path)

    # Índice rápido: subjectId → path_names canônico da árvore
    tree_index: dict[str, list[str]] = (
        tree_df.set_index("node_id")["path_names"].to_dict()
    )

    skill_to_h2: dict[str, str] = {}
    skill_to_h3: dict[str, str] = {}

    skipped_not_in_tree = 0
    skipped_too_shallow = 0

    for row in questions_df.itertuples(index=False):
        sid = str(row.subject_id)

        path_names = tree_index.get(sid)
        if path_names is None:
            skipped_not_in_tree += 1
            continue

        if len(path_names) >= 2:
            skill_to_h2[sid] = f"{path_names[0]} > {path_names[1]}"
        else:
            skipped_too_shallow += 1
            continue

        if len(path_names) >= 3:
            skill_to_h3[sid] = f"{path_names[0]} > {path_names[1]} > {path_names[2]}"
        else:
            # Para skills rasas (profundidade 2), H3 = H2 como fallback
            skill_to_h3[sid] = skill_to_h2[sid]

    print(f"Skills mapeadas para H2 : {len(skill_to_h2):,}")
    print(f"Skills mapeadas para H3 : {len(skill_to_h3):,}")
    print(f"  - com nível H3 real   : {sum(1 for s, h2 in skill_to_h2.items() if skill_to_h3.get(s, '') != h2):,}")
    print(f"  - com fallback H2→H3  : {sum(1 for s, h2 in skill_to_h2.items() if skill_to_h3.get(s, '') == h2):,}")
    print(f"Skipped (fora da árvore): {skipped_not_in_tree:,}")
    print(f"Skipped (hierarquia < 2): {skipped_too_shallow:,}")

    return skill_to_h2, skill_to_h3


def main() -> None:
    print("Construindo mapeamentos hierárquicos via subjectId + tree...\n")

    skill_to_h2, skill_to_h3 = build_hierarchy_mappings(QUESTIONS_PATH, TREE_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    h2_path = OUTPUT_DIR / "skill_to_h2.json"
    h3_path = OUTPUT_DIR / "skill_to_h3.json"

    with h2_path.open("w", encoding="utf-8") as f:
        json.dump(skill_to_h2, f, ensure_ascii=False, indent=2)

    with h3_path.open("w", encoding="utf-8") as f:
        json.dump(skill_to_h3, f, ensure_ascii=False, indent=2)

    print(f"\nSalvo: {h2_path}")
    print(f"Salvo: {h3_path}")


if __name__ == "__main__":
    main()
