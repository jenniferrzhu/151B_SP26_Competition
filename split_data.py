"""Split data/public.jsonl into 80/20 train/test using a fixed seed."""
import json
import random
from pathlib import Path

SEED = 42
TRAIN_FRAC = 0.8
SRC = Path("data/public.jsonl")
TRAIN_OUT = Path("data/train.jsonl")
TEST_OUT = Path("data/test.jsonl")

data = [json.loads(line) for line in open(SRC)]
random.Random(SEED).shuffle(data)

n_train = int(len(data) * TRAIN_FRAC)
train, test = data[:n_train], data[n_train:]

with open(TRAIN_OUT, "w") as f:
    for d in train:
        f.write(json.dumps(d) + "\n")
with open(TEST_OUT, "w") as f:
    for d in test:
        f.write(json.dumps(d) + "\n")

print(f"Total: {len(data)}  Train: {len(train)}  Test: {len(test)}")
print(f"Wrote {TRAIN_OUT} and {TEST_OUT}")
