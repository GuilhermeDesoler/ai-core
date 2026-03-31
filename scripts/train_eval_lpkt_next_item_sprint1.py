from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from brain_kt.preprocessing.temporal_features import transform_time_features

# This script wraps original pipeline adding temporal improvements

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "user_sequences.json"

MAX_SEQ_LEN = 75
STRIDE = 25

with open(INPUT_PATH) as f:
    data = json.load(f)

# apply temporal transformation
for user in data:
    dt, tr = transform_time_features(user["delta_ts"], user["time_responses"])
    user["delta_ts"] = dt
    user["time_responses"] = tr

print("Sprint1 temporal transformation applied.")
print(f"Configured MAX_SEQ_LEN={MAX_SEQ_LEN}, STRIDE={STRIDE}")

# NOTE: reuse original training scripts after preprocessing
print("Now run original train_eval_lpkt_next_item.py")
