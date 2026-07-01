import logging
from multiprocessing.pool import Pool
import os

from basics.bpe_tokenizer.pretokenizer import find_chunk_boundaries, init_vocab_map


logger = logging.getLogger(__name__)
SPLIT_SPECIAL_TOKEN = b"<|endoftext|>"


def train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_token=SPLIT_SPECIAL_TOKEN,
    nproc=None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if nproc is None:
        nproc = os.cpu_count() or 1
    init_vocab_map_args = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, nproc, split_token)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            init_vocab_map_args.append((input_path, start, end, split_token, special_tokens))

    assert len(init_vocab_map_args) == nproc, (
        f"Expected {nproc} chunks, but got {len(init_vocab_map_args)}"
    )

    # init vocabulary map
    with Pool(nproc) as pool:
        result_slice = pool.starmap(init_vocab_map, init_vocab_map_args)
        vocab_map: dict[tuple[bytes, ...], int] = {}
        spt_map: dict[bytes, int] = {}

        # merge them all
        for vocab, spt in result_slice:
            for b, c in vocab.items():
                if b not in vocab_map:
                    vocab_map[b] = 0
                vocab_map[b] += c
            for b, c in spt.items():
                if b not in spt_map:
                    spt_map[b] = 0
                spt_map[b] += c
        logger.info(
            f"vocab map has been built: vocab_map_size={len(vocab_map)}, spt_map_size={len(spt_map)}"
        )

    remaining = vocab_size - len(spt_map)
    while remaining > 0:
        remaining = compute_bpe(vocab_map, remaining, nproc)

    # todo
    return ({}, [])


def compute_bpe(vocab_map: dict[tuple[bytes, ...], int], vocab_size: int, nproc: int) -> int:
    """compute bpe
    1. spawn child processes
        1.1. iterate list
            1.1.1. mapping big pair to count
            1.1.2. mapping big pair to key
    2. merge pairs by count locally
    3. select the max
    4. spawn child processes
        4.1. iterate pairs
            4.1.1. update key's tuple
    """
    
    # with Pool(nproc) as pool:
    #     items = list(vocab_map.items())

    pass
