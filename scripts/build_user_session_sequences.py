from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "dataset" / "answers_prepared.csv"
INPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "dataset" / "answers_prepared.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"

SESSION_GAP_MINUTES = 60
MIN_SESSION_LEN = 2


def load_answers() -> pd.DataFrame:
    if INPUT_CSV.exists():
        return pd.read_csv(INPUT_CSV)
    if INPUT_PARQUET.exists():
        return pd.read_parquet(INPUT_PARQUET)
    raise FileNotFoundError("answers_prepared not found")


def build_user_session_sequences(
    df: pd.DataFrame,
    session_gap_minutes: int = SESSION_GAP_MINUTES,
    min_session_len: int = MIN_SESSION_LEN,
):
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True).copy()
    df["delta_t_raw"] = df.groupby("user_id")["timestamp"].diff().fillna(0)
    df["new_session_flag"] = (
        (df["delta_t_raw"] > session_gap_minutes * 60 * 1000)
        | (df.groupby("user_id").cumcount() == 0)
    ).astype(int)
    df["session_idx"] = df.groupby("user_id")["new_session_flag"].cumsum()

    sequences = []

    for (user_id, session_idx), group in df.groupby(["user_id", "session_idx"], sort=False):
        group = group.sort_values("timestamp").copy()
        if len(group) < min_session_len:
            continue

        session_delta_ts = group["timestamp"].diff().fillna(0).astype(int).tolist()

        sequences.append(
            {
                "user_id": str(user_id),
                "session_id": f"{user_id}__{int(session_idx)}",
                "question_ids": group["question_id"].astype(str).tolist(),
                "skill_ids": group["skill_id"].astype(str).tolist(),
                "corrects": group["correct"].astype(int).tolist(),
                "delta_ts": session_delta_ts,
                "time_responses": group["time_response"].fillna(0).astype(int).tolist(),
                "session_start_ts": int(group["timestamp"].iloc[0]),
                "session_end_ts": int(group["timestamp"].iloc[-1]),
                "session_len": int(len(group)),
            }
        )

    return sequences


def main():
    df = load_answers()
    sequences = build_user_session_sequences(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sequences, f)

    n_users = df["user_id"].nunique()
    n_sessions = len(sequences)
    avg_session_len = sum(seq["session_len"] for seq in sequences) / max(n_sessions, 1)

    print(f"Saved {n_sessions} session-based sequences to {OUTPUT_PATH}")
    print(f"Users: {n_users}")
    print(f"Session gap (minutes): {SESSION_GAP_MINUTES}")
    print(f"Min session length: {MIN_SESSION_LEN}")
    print(f"Average session length: {avg_session_len:.2f}")


if __name__ == "__main__":
    main()
