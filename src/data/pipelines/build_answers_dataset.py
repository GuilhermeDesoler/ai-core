from __future__ import annotations

import pandas as pd

from data.validation.validate_answers  import validate_answers
from data.preprocessing.prepare_answers import (
    prepare_answers,
    add_delta_t,
    filter_min_interactions,
    drop_invalid_answers,
)


def build_answers_dataset(
    answers_df: pd.DataFrame,
    min_interactions: int = 2,
) -> tuple[pd.DataFrame, dict]:
    initial_rows = len(answers_df)

    # LIMPEZA PRIMEIRO
    df = drop_invalid_answers(answers_df)

    dropped_rows = initial_rows - len(df)

    # valida só depois de limpar
    report = validate_answers(df)

    df = prepare_answers(df)
    df = add_delta_t(df)
    df = filter_min_interactions(df, min_interactions=min_interactions)

    report["dropped_invalid_rows"] = int(dropped_rows)
    report["n_rows_after_filter"] = int(len(df))
    report["n_users_after_filter"] = int(df["user_id"].nunique())

    return df, report
