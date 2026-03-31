from collections import defaultdict

def add_behavior_features(sequence):
    attempts = defaultdict(int)
    corrects = defaultdict(int)
    streak = 0

    new_feats = {
        "skill_attempts": [],
        "rolling_acc": [],
        "streak": []
    }

    for s, c in zip(sequence["skill_ids"], sequence["corrects"]):
        attempts[s] += 1
        if c == 1:
            corrects[s] += 1
            streak = max(1, streak + 1)
        else:
            streak = min(-1, streak - 1)

        acc = corrects[s] / attempts[s]

        new_feats["skill_attempts"].append(attempts[s])
        new_feats["rolling_acc"].append(acc)
        new_feats["streak"].append(streak)

    sequence.update(new_feats)
    return sequence


def transform_sequences(sequences):
    return [add_behavior_features(seq) for seq in sequences]
