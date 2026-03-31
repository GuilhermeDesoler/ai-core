import os

experiments = [
    (50, 25, 0.2),
    (75, 25, 0.2),
    (100, 50, 0.2),
    (75, 25, 0.3),
]

for max_seq_len, stride, dropout in experiments:
    print(
        f"\n=== Running Sprint2 LPKT | "
        f"max_seq_len={max_seq_len} | stride={stride} | dropout={dropout} ==="
    )
    os.system(
        f"python scripts/train_eval_lpkt_next_item_sprint2.py "
        f"--max-seq-len {max_seq_len} --stride {stride} --dropout {dropout}"
    )
