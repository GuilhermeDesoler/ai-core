from pathlib import Path
import sys


def setup_src_path():
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    return project_root
