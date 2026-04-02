"""
run_data_pipeline.py
====================
Prepara os dados brutos e constrói as sequências de usuários.

Etapas:
  1. prepare_answers_dataset  →  data/processed/dataset/answers_prepared.csv
  2. build_user_sequences     →  data/processed/sequences/user_sequences.json
  3. build_skill_hierarchy    →  data/processed/mappings/skill_to_h2.json
                                  data/processed/mappings/skill_to_h3.json

Uso:
  python run_data_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"


def _step(label: str, script: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if not script.exists():
        print(f"[ERRO] Script não encontrado: {script}")
        sys.exit(1)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n[ERRO] Etapa falhou (código {result.returncode})")
        sys.exit(result.returncode)

    print(f"\n[OK] Concluído em {elapsed:.1f}s")


def main() -> None:
    print("\nIniciando pipeline de dados...")
    total = time.perf_counter()

    _step(
        "1/3  Preparando dataset de respostas",
        SCRIPTS / "prepare_answers_dataset.py",
    )
    _step(
        "2/3  Construindo sequências de usuários",
        SCRIPTS / "build_user_session_sequences.py",
    )
    _step(
        "3/3  Construindo mapeamentos hierárquicos skill → H2 / H3 (via tree)",
        SCRIPTS / "build_skill_hierarchy_from_tree.py",
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline de dados concluído em {time.perf_counter() - total:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
