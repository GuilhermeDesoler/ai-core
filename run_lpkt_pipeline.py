"""
run_lpkt_pipeline.py
====================
Treina os modelos LPKT (Learning Process Knowledge Tracing).

Pré-requisito: run_data_pipeline.py já executado.

Etapas:
  1. train_eval_lpkt_next_item  →  artifacts/runs_lpkt/
  2. train_eval_lpkt_topic_h2  →  artifacts/runs_lpkt_h2/
  3. train_eval_lpkt_topic_h3  →  artifacts/runs_lpkt_h3/

Uso:
  python run_lpkt_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"

_REQUIRED = [
    PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json",
    PROJECT_ROOT / "data" / "processed" / "mappings" / "skill_to_h2.json",
    PROJECT_ROOT / "data" / "processed" / "mappings" / "skill_to_h3.json",
]


def _check_prerequisites() -> None:
    missing = [p for p in _REQUIRED if not p.exists()]
    if missing:
        print("[ERRO] Arquivos necessários não encontrados:")
        for p in missing:
            print(f"  - {p}")
        print("\nExecute run_data_pipeline.py primeiro.")
        sys.exit(1)


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
    _check_prerequisites()

    print("\nIniciando pipeline LPKT...")
    total = time.perf_counter()

    _step(
        "1/3  LPKT Next-Item",
        SCRIPTS / "train_eval_lpkt_next_item.py",
    )
    _step(
        "2/3  LPKT Topic H2",
        SCRIPTS / "train_eval_lpkt_topic_h2.py",
    )
    _step(
        "3/3  LPKT Topic H3",
        SCRIPTS / "train_eval_lpkt_topic_h3.py",
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline LPKT concluído em {time.perf_counter() - total:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
