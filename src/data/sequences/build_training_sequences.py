from __future__ import annotations

from typing import List, Dict, Any


def build_training_sequences(
    sequences: List[Dict[str, Any]],
    min_seq_len: int = 3,
) -> List[Dict[str, Any]]:
    training_data = []

    for seq in sequences:
        length = len(seq["question_ids"])

        if length < min_seq_len:
            continue

        input_seq = {
            "user_id": seq["user_id"],
            "question_ids": seq["question_ids"][:-1],
            "skill_ids": seq["skill_ids"][:-1],
            "corrects": seq["corrects"][:-1],
            "delta_ts": seq["delta_ts"][:-1],
            "time_responses": seq["time_responses"][:-1],
        }

        target_seq = {
            "corrects": seq["corrects"][1:],  # 🔥 shift
        }

        training_data.append({
            "input": input_seq,
            "target": target_seq,
        })

    return training_data
