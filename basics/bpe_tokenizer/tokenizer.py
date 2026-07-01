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

    # 256 is the size of ascii
    tokens_len = vocab_size - len(spt_map) - 256
    tokens = []
    while len(tokens) < tokens_len:
        compute_bpe(vocab_map, tokens)

    # todo
    return ({}, [])


def compute_bpe(vocab_map: dict[tuple[bytes, ...], int], tokens: list[bytes]):
    # byte-pair to (count, key_set)
    bp_map: dict[tuple[bytes, bytes], tuple[int, set[tuple[bytes, ...]]]] = {}
    best_count = 0
    tied_list: list[tuple[bytes, bytes]] = []

    for k, v in vocab_map.items():
        for a, b in zip(k[:-1], k[1:]):
            bp = (a, b)
            entry = bp_map.setdefault(bp, [0, set()])
            entry[0] += v
            entry[1].add(k)

            if entry[0] > best_count:
                best_count = entry[0]
                tied_list = [bp]
            elif entry[0] == best_count:
                tied_list.append(bp)

    if len(tied_list) == 0:
        return

    max_bp = max(tied_list)
    max_bp_token = b"".join(max_bp)
    tokens.append(max_bp_token)
    (_, key_set) = bp_map[max_bp]
    for k in key_set:
        count = vocab_map.pop(k)
        new_k = []
        i = 0
        while i < len(k):
            if i + 1 == len(k):
                new_k.append(k[i])
                break
            if k[i] == max_bp[0] and k[i + 1] == max_bp[1]:
                new_k.append(k[i] + k[i + 1])
                i += 2
            else:
                new_k.append(k[i])
                i += 1
        vocab_map[tuple(new_k)] = count
