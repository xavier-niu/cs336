from multiprocessing.pool import Pool
import os
from pathlib import Path

from basics.bpe_tokenizer.pretokenizer import find_chunk_boundaries, init_vocab_map


REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_STORIES_TRAIN_PATH = REPO_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"

CPUS = os.cpu_count() or 1

if not TINY_STORIES_TRAIN_PATH.exists():
    raise FileNotFoundError(
        f'File not found: {TINY_STORIES_TRAIN_PATH}, please run "$PROJECT/data/download.sh" first'
    )

SPLIT_SPECIAL_TOKEN = b"<|endoftext|>"

args = []
with open(TINY_STORIES_TRAIN_PATH, "rb") as f:
    boundaries = find_chunk_boundaries(f, CPUS, SPLIT_SPECIAL_TOKEN)
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args.append((TINY_STORIES_TRAIN_PATH, start, end, SPLIT_SPECIAL_TOKEN))

assert len(args) == CPUS, f"Expected {CPUS} chunks, but got {len(args)}"

with Pool(CPUS) as pool:
    results = pool.starmap(init_vocab_map, args)
