import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    max_seq_len: int = 100,
    min_seq_len: int = 10,
):
    sequences = []

    df = df.sort_values(["student_id", "timestamp"])

    for student_id, g in df.groupby("student_id"):
        q = g["question_id"].values
        r = g["is_correct"].values

        if len(q) < min_seq_len:
            continue

        # sliding window
        for start in range(0, len(q), max_seq_len):
            q_seq = q[start:start + max_seq_len]
            r_seq = r[start:start + max_seq_len]

            if len(q_seq) < min_seq_len:
                continue

            # 🔥 SHIFT CORRETO (anti-leakage)
            r_prev = np.concatenate([[0], r_seq[:-1]])
            target = r_seq

            mask = np.ones(len(q_seq), dtype=int)

            sequences.append({
                "student_id": student_id,
                "q_seq": q_seq.astype(int),
                "r_prev_seq": r_prev.astype(float),
                "target_seq": target.astype(float),
                "mask_seq": mask,
            })

    return pd.DataFrame(sequences)
