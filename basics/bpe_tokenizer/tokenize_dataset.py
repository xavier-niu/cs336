import argparse
from pathlib import Path
import pickle
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from basics.bpe_tokenizer.tokenizer import Tokenizer


def tracked_lines(lines: Iterable[str], pbar: tqdm, every: int = 4 * 1024 * 1024) -> Iterator[str]:
    """Yield lines unchanged while advancing ``pbar`` by their encoded byte length.

    ``encode_iterable`` yields token ids, which say nothing about how far through
    the input we are, so progress is measured on the way in instead. Byte counts
    are batched so a multi-GB file does not make millions of ``update`` calls.
    """
    pending = 0
    for line in lines:
        pending += len(line.encode("utf-8"))
        if pending >= every:
            pbar.update(pending)
            pending = 0
        yield line

    if pending:
        pbar.update(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize text")
    parser.add_argument(
        "--tokenizer-dataset",
        required=False,
        help="tokenizer dataset name, e.g. TinyStoriesV2-GPT4-train",
    )
    parser.add_argument(
        "--dataset", required=True, help="dataset name, e.g. TinyStoriesV2-GPT4-train"
    )

    args = parser.parse_args()

    dataset = args.dataset
    if args.tokenizer_dataset is None:
        tokenizer_dataset = dataset
    else:
        tokenizer_dataset = args.tokenizer_dataset

    repo_root = Path(__file__).resolve().parents[2]
    doc_path = repo_root / "data" / f"{dataset}.txt"
    merges_path = repo_root / "outputs" / f"merges-{tokenizer_dataset}.pkl"
    vocab_path = repo_root / "outputs" / f"vocab-{tokenizer_dataset}.pkl"
    tokens_path = repo_root / "outputs" / f"tokens-{dataset}.npy"

    if not doc_path.exists():
        raise FileNotFoundError(f"{dataset} doc path not found")
    if not merges_path.exists():
        raise FileNotFoundError(f"{dataset} merges path not found")
    if not vocab_path.exists():
        raise FileNotFoundError(f"{dataset} vocab path not found")

    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    with open(vocab_path, "rb") as f:
        vocal = pickle.load(f)

    tokenizer = Tokenizer(vocal, merges)

    block_size = 128 * 1024 * 1024
    tokens = np.empty(block_size, dtype=np.uint16)
    idx = 0

    token_count = 0
    with open(doc_path, "r") as f:
        with open(tokens_path, "wb") as npf:
            with tqdm(
                total=doc_path.stat().st_size,
                desc=f"tokenizing {dataset}",
                unit="B",
                unit_scale=True,
                smoothing=0.5,
            ) as pbar:
                for token in tokenizer.encode_iterable(tracked_lines(f, pbar)):
                    tokens[idx] = token
                    idx += 1
                    token_count += 1
                    if idx >= block_size:
                        # flush to file
                        tokens.tofile(npf)
                        idx = 0
                        pbar.set_postfix(tokens=f"{token_count / 1e6:.1f}M")

                tokens[:idx].tofile(npf)

    print(f"wrote {token_count:,} tokens to {tokens_path}")


if __name__ == "__main__":
    main()
