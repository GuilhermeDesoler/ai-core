from __future__ import annotations

import pandas as pd

from data.validation import validate_answers
from src.data.preprocessing.prepare_answers import (
    prepare_answers,
    add_delta_t,
    filter_min_interactions,
)


def build_answers_dataset(
    answers_df: pd.DataFrame,
    min_interactions: int = 2,
) -> tuple[pd.DataFrame, dict]:
    report = validate_answers(answers_df)

    df = prepare_answers(answers_df)
    df = add_delta_t(df)
    df = filter_min_interactions(df, min_interactions=min_interactions)

    report["n_rows_after_filter"] = int(len(df))
    report["n_users_after_filter"] = int(df["user_id"].nunique())

    return df, report
