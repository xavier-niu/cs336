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

    # init vocabulary map
    with Pool(nproc) as pool:
        vocab_map_partial = pool.starmap(init_vocab_map, init_vocab_map_args)
        vocab_map: dict[tuple[bytes, ...], int] = {}

        # merge them all
        for vocab in vocab_map_partial:
            for b, c in vocab.items():
                if b not in vocab_map:
                    vocab_map[b] = 0
                vocab_map[b] += c
        logger.info(f"vocab map has been built: vocab_map_size={len(vocab_map)}")

    vocab = {}
    token_id = 0
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    for sp_token in special_tokens:
        vocab[token_id] = sp_token.encode("utf-8")
        token_id += 1

    tokens_len = vocab_size - len(vocab)
    trained_byte_pairs: list[tuple[bytes, bytes]] = []
    while len(trained_byte_pairs) < tokens_len:
        if not compute_bpe(vocab_map, trained_byte_pairs):
            break
        (b1, b2) = trained_byte_pairs[-1]
        vocab[token_id] = b1 + b2
        token_id += 1

    return (vocab, trained_byte_pairs)


def compute_bpe(vocab_map: dict[tuple[bytes, ...], int], tokens: list[tuple[bytes, bytes]]) -> bool:
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
        return False

    max_bp = max(tied_list)
    tokens.append(max_bp)
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

    return True
