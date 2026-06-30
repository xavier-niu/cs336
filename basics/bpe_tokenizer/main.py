import argparse
import logging
from pathlib import Path

from basics.bpe_tokenizer import tokenizer


parser = argparse.ArgumentParser(description="Train BPE on a dataset")
parser.add_argument("dataset", help="dataset name, e.g. TinyStoriesV2-GPT4-train")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / f"{args.dataset}.txt"

if not DATA_PATH.exists():
    raise SystemExit(f"{DATA_PATH} not found — run `bash data/download.sh` first")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

tokenizer.train(str(DATA_PATH), 1048576, [])
