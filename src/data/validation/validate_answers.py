from __future__ import annotations

from typing import Any
import pandas as pd


REQUIRED_COLUMNS = [
    "user_id",
    "question_id",
    "skill_id",
    "correct",
    "timestamp",
    "time_response",
]


def validate_answers(answers_df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in answers_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    report["n_rows"] = int(len(answers_df))
    report["n_unique_users"] = int(answers_df["user_id"].nunique())
    report["n_unique_questions"] = int(answers_df["question_id"].nunique())
    report["n_unique_skills"] = int(answers_df["skill_id"].nunique())

    report["null_user_id"] = int(answers_df["user_id"].isna().sum())
    report["null_question_id"] = int(answers_df["question_id"].isna().sum())
    report["null_skill_id"] = int(answers_df["skill_id"].isna().sum())
    report["null_correct"] = int(answers_df["correct"].isna().sum())
    report["null_timestamp"] = int(answers_df["timestamp"].isna().sum())
    report["null_time_response"] = int(answers_df["time_response"].isna().sum())

    report["duplicated_rows"] = int(answers_df.duplicated().sum())

    valid_correct_values = {0, 1}
    invalid_correct_mask = ~answers_df["correct"].isin(valid_correct_values)
    report["invalid_correct_values"] = int(invalid_correct_mask.sum())

    negative_timestamp_mask = answers_df["timestamp"] < 0
    report["negative_timestamp"] = int(negative_timestamp_mask.sum())

    negative_time_response_mask = answers_df["time_response"] < 0
    report["negative_time_response"] = int(negative_time_response_mask.sum())

    if report["null_user_id"] > 0:
        raise ValueError("Found null user_id values")
    if report["null_question_id"] > 0:
        raise ValueError("Found null question_id values")
    if report["null_skill_id"] > 0:
        raise ValueError("Found null skill_id values")
    if report["null_correct"] > 0:
        raise ValueError("Found null correct values")
    if report["null_timestamp"] > 0:
        raise ValueError("Found null timestamp values")
    if report["null_time_response"] > 0:
        raise ValueError("Found null time_response values")
    if report["invalid_correct_values"] > 0:
        raise ValueError("Found invalid correct values; expected only 0 or 1")
    if report["negative_timestamp"] > 0:
        raise ValueError("Found negative timestamp values")
    if report["negative_time_response"] > 0:
        raise ValueError("Found negative time_response values")

    return report
