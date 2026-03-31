from __future__ import annotations

from typing import Any


def map_user_sequences_to_h3(
    user_sequences: list[dict[str, Any]],
    skill_to_h3: dict[str, str],
) -> list[dict[str, Any]]:
    mapped_sequences: list[dict[str, Any]] = []

    for seq in user_sequences:
        mapped_question_ids: list[str] = []
        mapped_skill_ids: list[str] = []
        mapped_h3_ids: list[str] = []
        mapped_corrects: list[int] = []
        mapped_delta_ts: list[int | float] = []
        mapped_time_responses: list[int | float] = []

        for i, skill_id in enumerate(seq["skill_ids"]):
            h3_id = skill_to_h3.get(skill_id)
            if h3_id is None:
                continue

            mapped_question_ids.append(seq["question_ids"][i])
            mapped_skill_ids.append(skill_id)
            mapped_h3_ids.append(h3_id)
            mapped_corrects.append(seq["corrects"][i])
            mapped_delta_ts.append(seq["delta_ts"][i])
            mapped_time_responses.append(seq["time_responses"][i])

        if len(mapped_h3_ids) >= 3:
            mapped_sequences.append(
                {
                    "user_id": seq["user_id"],
                    "question_ids": mapped_question_ids,
                    "skill_ids": mapped_skill_ids,
                    "h3_ids": mapped_h3_ids,
                    "corrects": mapped_corrects,
                    "delta_ts": mapped_delta_ts,
                    "time_responses": mapped_time_responses,
                }
            )

    return mapped_sequences


def build_h3_topic_training_sequences(
    user_sequences: list[dict[str, Any]],
    skill_to_h3: dict[str, str],
    max_seq_len: int = 100,
    stride: int | None = None,
    min_seq_len: int = 3,
) -> list[dict[str, Any]]:
    if stride is None:
        stride = max_seq_len

    mapped_sequences = map_user_sequences_to_h3(user_sequences, skill_to_h3)
    training_data: list[dict[str, Any]] = []

    for seq in mapped_sequences:
        shifted_len = len(seq["h3_ids"]) - 1
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
                        "question_ids": seq["question_ids"][start:end],
                        "skill_ids": seq["skill_ids"][start:end],
                        "h3_ids": seq["h3_ids"][start:end],
                        "corrects": seq["corrects"][start:end],
                        "delta_ts": seq["delta_ts"][start:end],
                        "time_responses": seq["time_responses"][start:end],
                    },
                    "target": {
                        "next_h3_ids": seq["h3_ids"][start + 1 : end + 1],
                        "next_corrects": seq["corrects"][start + 1 : end + 1],
                    },
                }
            )

            if end == shifted_len:
                break
            start += stride

    return training_data
