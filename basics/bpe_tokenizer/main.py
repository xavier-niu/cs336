import argparse
import logging
from pathlib import Path
import pickle
import resource
import time

from basics.bpe_tokenizer import tokenizer


parser = argparse.ArgumentParser(description="Train BPE on a dataset")
parser.add_argument("--dataset", help="dataset name, e.g. TinyStoriesV2-GPT4-train")
parser.add_argument(
    "--size", help="maximum vocabulary size, default is 10000", default=10000, type=int
)
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / f"{args.dataset}.txt"
VOCAB_PATH = REPO_ROOT / "outputs" / f"vocab-{args.dataset}.pkl"
MERGES_PATH = REPO_ROOT / "outputs" / f"merges-{args.dataset}.pkl"
PRETOKEN_CACHE_PATH = REPO_ROOT / "data" / f"pretoken-cache-{args.dataset}.txt"

if not DATA_PATH.exists():
    raise SystemExit(f"{DATA_PATH} not found — run `bash data/download.sh` first")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

start = time.perf_counter()
(vocab, merges) = tokenizer.train(
    str(DATA_PATH), args.size, [], pretoken_cache_path=PRETOKEN_CACHE_PATH
)
end = time.perf_counter()

elapsed = end - start
peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

print("BPE tokenizer train is completed")
print(f"time:            {elapsed:.1f}s")
print(f"peak RSS (main): {peak_self / 1024**2:.2f} GB")
print(f"peak RSS (worker):{peak_kids / 1024**2:.2f} GB")

# Serialize vocab and merges
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)
with open(MERGES_PATH, "wb") as f:
    pickle.dump(merges, f)
