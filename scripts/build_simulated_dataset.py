from pathlib import Path
import json

import pandas as pd
from scipy import sparse

from brain_kt.dataset import simulate_students, build_sequences


def main() -> None:
    qmatrix_path = Path("data/processed/qmatrix/q_matrix.npz")
    output_dir = Path("data/processed/dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    q_matrix = sparse.load_npz(qmatrix_path)

    interactions_df = simulate_students(
        q_matrix=q_matrix,
        n_students=2000,
        min_interactions=30,
        max_interactions=120,
        seed=42,
    )

    seq_df = build_sequences(
        interactions_df,
        max_seq_len=100,
        min_seq_len=10,
    )

    interactions_df.to_parquet(output_dir / "simulated_interactions.parquet", index=False)
    seq_df.to_parquet(output_dir / "simulated_sequences.parquet", index=False)

    metadata = {
        "n_interactions": int(len(interactions_df)),
        "n_sequences": int(len(seq_df)),
        "n_students": int(interactions_df["student_id"].nunique()),
    }

    (output_dir / "simulated_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Dataset simulado gerado com sucesso.")
    print(metadata)


if __name__ == "__main__":
    main()
