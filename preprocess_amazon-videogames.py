"""
Amazon Video Games → LightGCN Data Preprocessing
==================================================
This script downloads the Amazon Reviews 2023 Video Games dataset
from HuggingFace and converts it to the format expected by
gusye1234/LightGCN-PyTorch.

LightGCN format:
  train.txt / test.txt
  Each line: user_id item_id_1 item_id_2 item_id_3 ...
  (space-separated, 0-indexed integers)

Usage:
  pip install datasets pandas
  python preprocess_amazon_videogames.py

Output:
  data/amazon-videogames/train.txt
  data/amazon-videogames/test.txt
  data/amazon-videogames/user_list.txt
  data/amazon-videogames/item_list.txt
  data/amazon-videogames/stats.txt
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
#from datasets import load_dataset

# ============================================================
# CONFIG — tune these
# ============================================================
CATEGORY = "Video_Games"
K_CORE = 5              # minimum interactions per user AND per item
TEST_RATIO = 0.2        # fraction of each user's items held out for test
RANDOM_SEED = 2020      # match LightGCN default seed
OUTPUT_DIR = "data/amazon-videogames"
# Set to True to treat any rating as implicit positive feedback
# Set to False to only keep ratings >= RATING_THRESHOLD
IMPLICIT = True
RATING_THRESHOLD = 4.0  # only used if IMPLICIT=False

from huggingface_hub import hf_hub_download
import json

print("Downloading Amazon Reviews 2023 - Video Games (raw JSONL)...")
filepath = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw/review_categories/Video_Games.jsonl",
    repo_type="dataset"
)

rows = []
with open(filepath, "r") as f:
    for line in f:
        obj = json.loads(line)
        rows.append({
            "user": obj["user_id"],
            "item": obj["parent_asin"],
            "rating": obj["rating"],
            "timestamp": obj["timestamp"],
        })

df = pd.DataFrame(rows)
print(f"Raw dataset: {len(df)} reviews")

# STEP 2: Clean and filter
print("STEP 2: Cleaning and filtering")

# Keep relevant columns
#df = df[["user_id", "parent_asin", "rating", "timestamp"]].copy()
#df.columns = ["user", "item", "rating", "timestamp"]

# Drop duplicates (same user-item pair)
df = df.drop_duplicates(subset=["user", "item"], keep="first")
print(f"After dedup: {len(df)} interactions")

# Optional: filter by rating
if not IMPLICIT:
    df = df[df["rating"] >= RATING_THRESHOLD]
    print(f"After rating filter (>= {RATING_THRESHOLD}): {len(df)} interactions")

# K-core filtering: iteratively remove users/items with < K interactions
# until stable (both users and items have >= K interactions)
print(f"\nApplying {K_CORE}-core filtering...")
prev_len = 0
iteration = 0
while len(df) != prev_len:
    prev_len = len(df)
    iteration += 1

    # Filter users
    user_counts = df["user"].value_counts()
    valid_users = user_counts[user_counts >= K_CORE].index
    df = df[df["user"].isin(valid_users)]

    # Filter items
    item_counts = df["item"].value_counts()
    valid_items = item_counts[item_counts >= K_CORE].index
    df = df[df["item"].isin(valid_items)]

    print(f"  Iteration {iteration}: {len(df)} interactions, "
          f"{df['user'].nunique()} users, {df['item'].nunique()} items")

print(f"\nAfter {K_CORE}-core filtering:")
n_users = df["user"].nunique()
n_items = df["item"].nunique()
n_interactions = len(df)
sparsity = n_interactions / (n_users * n_items)
print(f"  Users: {n_users}")
print(f"  Items: {n_items}")
print(f"  Interactions: {n_interactions}")
print(f"  Sparsity: {sparsity:.6f}")

# STEP 3: Remap IDs to 0-indexed integers
print("STEP 3: Remapping IDs")

# Sort users and items for reproducibility
unique_users = sorted(df["user"].unique())
unique_items = sorted(df["item"].unique())

user2id = {u: i for i, u in enumerate(unique_users)}
item2id = {it: i for i, it in enumerate(unique_items)}

df["user_idx"] = df["user"].map(user2id)
df["item_idx"] = df["item"].map(item2id)

print(f"User ID range: 0 to {len(user2id) - 1}")
print(f"Item ID range: 0 to {len(item2id) - 1}")

# STEP 4: Train/test split (per-user leave-out)
print("STEP 4: Train/test split")

np.random.seed(RANDOM_SEED)

train_dict = defaultdict(list)
test_dict = defaultdict(list)

# Group by user
for user_idx, group in df.groupby("user_idx"):
    items = group["item_idx"].values

    if len(items) < 2:
        # If user has only 1 interaction, put in train
        train_dict[user_idx] = items.tolist()
        continue

    # Shuffle items for this user
    perm = np.random.permutation(len(items))
    items_shuffled = items[perm]

    n_test = max(1, int(len(items) * TEST_RATIO))
    test_items = items_shuffled[:n_test].tolist()
    train_items = items_shuffled[n_test:].tolist()

    train_dict[user_idx] = train_items
    test_dict[user_idx] = test_items

n_train = sum(len(v) for v in train_dict.values())
n_test = sum(len(v) for v in test_dict.values())
print(f"Train interactions: {n_train}")
print(f"Test interactions:  {n_test}")
print(f"Test ratio:         {n_test / (n_train + n_test):.3f}")

# STEP 5: Write output files
print("STEP 5: Writing output files")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# train.txt
with open(os.path.join(OUTPUT_DIR, "train.txt"), "w") as f:
    for user_idx in range(len(user2id)):
        items = train_dict.get(user_idx, [])
        line = " ".join([str(user_idx)] + [str(i) for i in items])
        f.write(line + "\n")

# test.txt
with open(os.path.join(OUTPUT_DIR, "test.txt"), "w") as f:
    for user_idx in range(len(user2id)):
        items = test_dict.get(user_idx, [])
        line = " ".join([str(user_idx)] + [str(i) for i in items])
        f.write(line + "\n")

# user_list.txt (maps original ID to new index)
with open(os.path.join(OUTPUT_DIR, "user_list.txt"), "w") as f:
    f.write("org_id remap_id\n")
    for orig, idx in sorted(user2id.items(), key=lambda x: x[1]):
        f.write(f"{orig} {idx}\n")

# item_list.txt
with open(os.path.join(OUTPUT_DIR, "item_list.txt"), "w") as f:
    f.write("org_id remap_id\n")
    for orig, idx in sorted(item2id.items(), key=lambda x: x[1]):
        f.write(f"{orig} {idx}\n")

# stats.txt
with open(os.path.join(OUTPUT_DIR, "stats.txt"), "w") as f:
    f.write(f"category: {CATEGORY}\n")
    f.write(f"k_core: {K_CORE}\n")
    f.write(f"implicit: {IMPLICIT}\n")
    f.write(f"n_users: {n_users}\n")
    f.write(f"n_items: {n_items}\n")
    f.write(f"n_interactions: {n_interactions}\n")
    f.write(f"n_train: {n_train}\n")
    f.write(f"n_test: {n_test}\n")
    f.write(f"sparsity: {sparsity:.6f}\n")
    f.write(f"test_ratio: {TEST_RATIO}\n")
    f.write(f"seed: {RANDOM_SEED}\n")

print(f"\nFiles written to {OUTPUT_DIR}/")
print(f"  train.txt  ({n_train} interactions)")
print(f"  test.txt   ({n_test} interactions)")
print(f"  user_list.txt")
print(f"  item_list.txt")
print(f"  stats.txt")

# STEP 6: Sanity checks
print("STEP 6: Sanity checks")

# Verify train.txt is readable
with open(os.path.join(OUTPUT_DIR, "train.txt")) as f:
    first_lines = [f.readline().strip() for _ in range(3)]
for line in first_lines:
    parts = line.split()
    print(f"  User {parts[0]}: {len(parts)-1} train items")

# Verify no overlap between train and test for same user
n_leaks = 0
for user_idx in range(len(user2id)):
    train_set = set(train_dict.get(user_idx, []))
    test_set = set(test_dict.get(user_idx, []))
    overlap = train_set & test_set
    if overlap:
        n_leaks += 1
print(f"  Train/test leaks: {n_leaks} users (should be 0)")

print("\n✓ Done! Next step: register this dataset in LightGCN-PyTorch")
print("  See SETUP_GUIDE.md for instructions.")