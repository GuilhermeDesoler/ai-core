from __future__ import annotations

import pandas as pd
from typing import List, Dict, Any


def build_user_sequences(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    sequences = []

    grouped = df.groupby("user_id")

    for user_id, user_df in grouped:
        user_df = user_df.sort_values("timestamp")

        sequence = {
            "user_id": user_id,
            "question_ids": user_df["question_id"].tolist(),
            "skill_ids": user_df["skill_id"].tolist(),
            "corrects": user_df["correct"].tolist(),
            "timestamps": user_df["timestamp"].tolist(),
            "delta_ts": user_df["delta_t"].tolist(),
            "time_responses": user_df["time_response"].tolist(),
        }

        sequences.append(sequence)

    return sequences
