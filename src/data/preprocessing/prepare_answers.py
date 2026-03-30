from __future__ import annotations

import pandas as pd


def prepare_answers(answers_df: pd.DataFrame) -> pd.DataFrame:
    df = answers_df.copy()

    df["user_id"] = df["user_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(str)
    df["correct"] = df["correct"].astype(int)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise").astype("int64")
    df["time_response"] = pd.to_numeric(df["time_response"], errors="raise").astype("int64")

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return df

def add_delta_t(answers_df: pd.DataFrame) -> pd.DataFrame:
    df = answers_df.copy()

    df["delta_t"] = df.groupby("user_id")["timestamp"].diff().fillna(0)
    df["delta_t"] = df["delta_t"].astype("int64")

    return df

def filter_min_interactions(
    answers_df: pd.DataFrame,
    min_interactions: int = 2,
) -> pd.DataFrame:
    df = answers_df.copy()

    counts = df.groupby("user_id").size()
    valid_users = counts[counts >= min_interactions].index

    df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

    return df

def drop_invalid_answers(answers_df: pd.DataFrame) -> pd.DataFrame:
    df = answers_df.copy()

    initial_rows = len(df)

    df = df.dropna(subset=[
        "user_id",
        "question_id",
        "skill_id",
        "correct",
        "timestamp",
        "time_response",
    ])

    df = df[df["timestamp"] >= 0]
    df = df[df["time_response"] >= 0]

    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped_duplicates = before_dedup - len(df)

    dropped_rows = initial_rows - len(df)

    print(f"Dropped {dropped_rows} invalid rows total")
    print(f"Removed {dropped_duplicates} exact duplicate rows")

    return df.reset_index(drop=True)