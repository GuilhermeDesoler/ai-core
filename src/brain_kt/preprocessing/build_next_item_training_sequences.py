from __future__ import annotations

from typing import Any


def build_next_item_training_sequences(
    user_sequences: list[dict[str, Any]],
    max_seq_len: int = 100,
    stride: int | None = None,
    min_seq_len: int = 3,
) -> list[dict[str, Any]]:
    if stride is None:
        stride = max_seq_len

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    training_data: list[dict[str, Any]] = []

    for seq in user_sequences:
        full_len = len(seq["question_ids"])
        if full_len < min_seq_len:
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
            "next_question_ids": seq["question_ids"][1:],
            "next_skill_ids": seq["skill_ids"][1:],
            "next_corrects": seq["corrects"][1:],
        }

        shifted_len = len(input_seq["question_ids"])
        if shifted_len < (min_seq_len - 1):
            continue

        start = 0
        while start < shifted_len:
            end = min(start + max_seq_len, shifted_len)
            if (end - start) < (min_seq_len - 1):
                break

            training_data.append(
                {
                    "user_id": seq["user_id"],
                    "input": {
                        "question_ids": input_seq["question_ids"][start:end],
                        "skill_ids": input_seq["skill_ids"][start:end],
                        "corrects": input_seq["corrects"][start:end],
                        "delta_ts": input_seq["delta_ts"][start:end],
                        "time_responses": input_seq["time_responses"][start:end],
                    },
                    "target": {
                        "next_question_ids": target_seq["next_question_ids"][start:end],
                        "next_skill_ids": target_seq["next_skill_ids"][start:end],
                        "next_corrects": target_seq["next_corrects"][start:end],
                    },
                }
            )

            if end == shifted_len:
                break
            start += stride

    return training_data
