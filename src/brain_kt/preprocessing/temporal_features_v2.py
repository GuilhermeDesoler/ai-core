from __future__ import annotations

import numpy as np


def transform_time_features(delta_ts, time_responses, clip_value=100000.0):
    delta_ts = np.asarray(delta_ts, dtype=np.float32)
    time_responses = np.asarray(time_responses, dtype=np.float32)

    delta_ts = np.clip(delta_ts, 0.0, clip_value)
    time_responses = np.clip(time_responses, 0.0, clip_value)

    delta_ts = np.log1p(delta_ts)
    time_responses = np.log1p(time_responses)

    delta_ts = delta_ts / (float(delta_ts.max()) + 1e-8)
    time_responses = time_responses / (float(time_responses.max()) + 1e-8)

    return delta_ts.tolist(), time_responses.tolist()


def transform_user_sequences_time_features(user_sequences, clip_value=100000.0):
    transformed = []

    for seq in user_sequences:
        new_seq = dict(seq)
        delta_ts, time_responses = transform_time_features(
            seq["delta_ts"],
            seq["time_responses"],
            clip_value=clip_value,
        )
        new_seq["delta_ts"] = delta_ts
        new_seq["time_responses"] = time_responses
        transformed.append(new_seq)

    return transformed
