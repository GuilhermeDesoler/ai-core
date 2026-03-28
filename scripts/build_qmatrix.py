from pathlib import Path

from brain_kt.qmatrix import build_qmatrix_from_files


def main() -> None:
    artifacts = build_qmatrix_from_files(
        tree_json_path=Path("data/raw/tree.json"),
        questions_json_path=Path("data/raw/questions.json"),
        output_dir=Path("data/processed/qmatrix"),
    )
    print("Q-matrix gerada com sucesso.")
    print(artifacts.metadata)


if __name__ == "__main__":
    main()
