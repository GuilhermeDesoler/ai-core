import numpy as np

def transform_time_features(delta_ts, time_responses, clip_value=1e5):
    delta_ts = np.array(delta_ts)
    time_responses = np.array(time_responses)

    # clipping
    delta_ts = np.clip(delta_ts, 0, clip_value)
    time_responses = np.clip(time_responses, 0, clip_value)

    # log transform
    delta_ts = np.log1p(delta_ts)
    time_responses = np.log1p(time_responses)

    return delta_ts.tolist(), time_responses.tolist()
