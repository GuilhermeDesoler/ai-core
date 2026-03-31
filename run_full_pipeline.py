"""
run_full_pipeline.py
====================
Executa o pipeline completo do zero: dados → DKT → LPKT.

Etapas:
  [DATA]  1. prepare_answers_dataset
          2. build_user_sequences
          3. build_skill_hierarchy_from_tree  (→ skill_to_h2.json + skill_to_h3.json)

  [DKT]   4. train_eval_dkt_next_item
          5. train_eval_dkt_topic_h2
          6. train_eval_dkt_topic_h3

  [LPKT]  7. train_eval_lpkt_next_item
          8. train_eval_lpkt_topic_h2
          9. train_eval_lpkt_topic_h3

Uso:
  python run_full_pipeline.py

  # Pular etapa de dados (sequências já existem):
  python run_full_pipeline.py --skip-data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_pipeline(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"\n[ERRO] Pipeline falhou: {script.name} (código {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo brain-kt")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Pula o pipeline de dados (usa sequências já existentes)",
    )
    args = parser.parse_args()

    total = time.perf_counter()

    print("\n" + "=" * 60)
    print("  brain-kt  |  Full Pipeline")
    print("=" * 60)

    if not args.skip_data:
        print("\n[1/3] Pipeline de dados")
        _run_pipeline(PROJECT_ROOT / "run_data_pipeline.py")
    else:
        print("\n[1/3] Pipeline de dados  →  ignorado (--skip-data)")

    print("\n[2/3] Pipeline DKT")
    _run_pipeline(PROJECT_ROOT / "run_dkt_pipeline.py")

    print("\n[3/3] Pipeline LPKT")
    _run_pipeline(PROJECT_ROOT / "run_lpkt_pipeline.py")

    elapsed = time.perf_counter() - total
    print(f"\n{'='*60}")
    print(f"  Pipeline completo finalizado em {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
