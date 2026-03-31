import os

print("=== Sprint 1 Pipeline ===")

print("Step 1: Data preparation already assumed done")

print("Step 2: Apply temporal transformations")
os.system("python scripts/train_eval_lpkt_next_item_sprint1.py")

print("Step 3: Train LPKT Next Item")
os.system("python scripts/train_eval_lpkt_next_item.py")

print("Step 4: Train DKT Next Item")
os.system("python scripts/train_eval_dkt_next_item.py")

print("=== Sprint 1 Completed ===")
