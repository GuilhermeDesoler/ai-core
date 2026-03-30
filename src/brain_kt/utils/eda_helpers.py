from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


NESTED_TYPES = (list, dict, set, tuple)


def detect_nested_columns(df: pd.DataFrame) -> list[str]:
    nested_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        has_nested = series.map(lambda x: isinstance(x, NESTED_TYPES)).any()
        if has_nested:
            nested_cols.append(col)

    return nested_cols


def get_hashable_columns(df: pd.DataFrame) -> list[str]:
    hashable_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        all_hashable = series.map(lambda x: not isinstance(x, NESTED_TYPES)).all()
        if all_hashable:
            hashable_cols.append(col)

    return hashable_cols


def safe_duplicated_count(df: pd.DataFrame, subset: Iterable[str] | None = None) -> int:
    if subset is None:
        subset = get_hashable_columns(df)

    subset = [col for col in subset if col in df.columns]
    if not subset:
        return 0

    return int(df.duplicated(subset=subset).sum())


def safe_log1p(series: pd.Series) -> pd.Series:
    return series.clip(lower=0).map(np.log1p)
