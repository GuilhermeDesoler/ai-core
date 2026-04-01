from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"

REQUIRED = [
    PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json",
]


def check():
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("[ERRO] Rode o data pipeline antes")
        sys.exit(1)


def step(label, script):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print(f"[OK] {time.time()-t0:.1f}s")


def main():
    check()

    print("Iniciando pipeline DKVMN...")

    step("1/1 DKVMN Next-Item", SCRIPTS / "train_eval_dkvmn_next_item.py")


if __name__ == "__main__":
    main()
