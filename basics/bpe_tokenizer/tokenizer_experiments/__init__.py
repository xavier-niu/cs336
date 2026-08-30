from pathlib import Path
import pickle
import random
import time
from typing import Iterable, TextIO

from basics.bpe_tokenizer import SPLIT_SPECIAL_TOKEN
from basics.bpe_tokenizer.tokenizer import Tokenizer


def document_iterable(file: TextIO, split_token: str) -> Iterable[str]:
    last: str | None = None
    while True:
        # read 128M data to buffer
        buffer = file.read(128 * 1024 * 1024)
        if len(buffer) == 0:
            break
        if last is not None:
            buffer = "".join([last, buffer])
        chunks = buffer.split(split_token)
        last = chunks.pop()
        for chunk in chunks:
            yield chunk

    if last is not None:
        yield last


def get_compression_ratio(dataset: str, n_doc: int, tokenizer_dataset: str | None = None) -> float:
    if n_doc == 0:
        return 0

    if tokenizer_dataset is None:
        tokenizer_dataset = dataset

    repo_root = Path(__file__).resolve().parents[3]
    doc_path = repo_root / "data" / f"{dataset}.txt"
    merges_path = repo_root / "outputs" / f"merges-{tokenizer_dataset}.pkl"
    vocab_path = repo_root / "outputs" / f"vocab-{tokenizer_dataset}.pkl"

    if not doc_path.exists():
        raise FileNotFoundError(f"{dataset} doc path not found")
    if not merges_path.exists():
        raise FileNotFoundError(f"{dataset} merges path not found")
    if not vocab_path.exists():
        raise FileNotFoundError(f"{dataset} vocab path not found")

    doc_size = 0
    with open(doc_path) as f:
        for _ in document_iterable(f, SPLIT_SPECIAL_TOKEN):
            doc_size += 1

    samples = sorted(random.sample(range(doc_size), n_doc))
    if len(samples) != n_doc:
        raise ValueError(f"expected to sample {n_doc} documents, but got {len(samples)}")
    docs: list[str] = []

    with open(doc_path) as f:
        idx = 0
        expected_idx = samples.pop(0)
        for doc in document_iterable(f, SPLIT_SPECIAL_TOKEN):
            if idx == expected_idx:
                docs.append(doc)
                if len(samples) == 0:
                    break
                else:
                    expected_idx = samples.pop(0)
            idx += 1

    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges)
    bytes_len = 0
    tokens_len = 0

    elpased = 0
    for doc in docs:
        bytes_len += len(doc.encode("utf-8"))
        start = time.perf_counter()
        tokens_len += len(tokenizer.encode(doc))
        elpased += time.perf_counter() - start

    print(
        f"tokenizer throughput is {bytes_len / elpased:.2f} bytes/s, tokenizer dataset is {tokenizer_dataset}"
    )

    if tokens_len == 0:
        return 0
    return bytes_len / tokens_len
