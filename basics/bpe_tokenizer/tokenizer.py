import logging
from multiprocessing.pool import Pool
import os

from sortedcontainers import SortedDict

from basics.bpe_tokenizer.pretokenizer import find_chunk_boundaries, init_vocab_map


logger = logging.getLogger(__name__)
SPLIT_SPECIAL_TOKEN = "<|endoftext|>"
BytesPair = tuple[bytes, bytes]
BytesTuple = tuple[bytes, ...]


def train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_token=SPLIT_SPECIAL_TOKEN,
    nproc=None,
) -> tuple[dict[int, bytes], list[BytesPair]]:
    if split_token not in special_tokens:
        logger.info(f"split_token ({split_token}) does not exist in special_tokens, will append it")
        special_tokens.append(split_token)

    if nproc is None:
        nproc = os.cpu_count() or 1
    init_vocab_map_args = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, nproc, split_token.encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            init_vocab_map_args.append(
                (input_path, start, end, split_token.encode("utf-8"), special_tokens)
            )

    # init vocabulary map
    with Pool(nproc) as pool:
        vocab_map_partial = pool.starmap(init_vocab_map, init_vocab_map_args)
        vocab_map: dict[BytesTuple, int] = {}

        # merge them all
        for vocab in vocab_map_partial:
            for b, c in vocab.items():
                if b not in vocab_map:
                    vocab_map[b] = 0
                vocab_map[b] += c
        logger.info(f"vocab map has been built: vocab_map_size={len(vocab_map)}")

    return compute_bpe(vocab_map, vocab_size, special_tokens)


def compute_bpe(
    vocab_map: dict[BytesTuple, int], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[BytesPair]]:
    # init vocab with all values of a byte and special tokens
    vocab = {}
    token_id = 0
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    for sp_token in special_tokens:
        vocab[token_id] = sp_token.encode("utf-8")
        token_id += 1

    merges_len = vocab_size - len(vocab)
    merges: list[BytesPair] = []

    # init bp_map
    # value tuple includes
    # - int: frequency
    # - set[VocabBytesTuple]: related vocab bytes tuple
    bp_map: dict[BytesPair, tuple[int, set[BytesTuple]]] = {}
    # frequency -> set(BytesPair)
    freq_to_bpset_map = SortedDict()

    def incr_bytes_pair(bp: BytesPair, vocab: BytesTuple, vocab_count: int):
        entry = bp_map.setdefault(bp, [0, set()])
        old_freq = entry[0]
        new_freq = old_freq + vocab_count
        entry[0] = new_freq
        entry[1].add(vocab)

        if old_freq != 0:
            freq_to_bpset_map[old_freq].discard(bp)
        freq_to_bpset_map.setdefault(new_freq, set()).add(bp)

    def decr_bytes_pair(bp: BytesPair, vocab: BytesTuple, vocab_count: int):
        if bp not in bp_map:
            return

        (old_freq, vocab_set) = bp_map[bp]
        new_freq = old_freq - vocab_count
        if new_freq <= 0:
            bp_map.pop(bp)
        else:
            bp_map[bp][0] = new_freq
            vocab_set.discard(vocab)

        freq_to_bpset_map[old_freq].discard(bp)
        if new_freq > 0:
            freq_to_bpset_map.setdefault(new_freq, set()).add(bp)

    for old_vocab, v in vocab_map.items():
        for a, b in zip(old_vocab[:-1], old_vocab[1:]):
            entry = bp_map.setdefault((a, b), [0, set()])
            entry[0] += v
            entry[1].add(old_vocab)

    for bp, (count, _) in bp_map.items():
        freq_to_bpset_map.setdefault(count, set()).add(bp)

    # exit conditions:
    # 1. reach upper limit of merges
    # 2. nothing to be merged
    while len(merges) < merges_len and len(freq_to_bpset_map) > 0:
        (_, bp_set) = freq_to_bpset_map.peekitem(-1)
        if len(bp_set) == 0:
            freq_to_bpset_map.popitem(-1)
            continue

        max_bp = max(bp_set)
        new_bytes = b"".join(max_bp)
        merges.append(max_bp)
        vocab[token_id] = new_bytes
        token_id += 1

        (_, vb_set) = bp_map[max_bp]
        vb_set = [x for x in vb_set]

        for old_vocab in vb_set:
            vocab_count = vocab_map.pop(old_vocab)
            # get the new bytes tuple for vocab_map
            new_vocab = []
            i = 0
            while i < len(old_vocab):
                if i != len(old_vocab) - 1 and (old_vocab[i], old_vocab[i + 1]) == max_bp:
                    new_vocab.append(b"".join(max_bp))
                    i += 2
                    continue
                new_vocab.append(old_vocab[i])
                i += 1
            # update vocab_map with a bytes tuple after merged
            new_vocab = tuple(new_vocab)
            vocab_map[new_vocab] = vocab_count

            old_bps = [p for p in zip(old_vocab[:-1], old_vocab[1:])]
            new_bps = [p for p in zip(new_vocab[:-1], new_vocab[1:])]
            for bp in old_bps:
                decr_bytes_pair(bp, old_vocab, vocab_count)
            for bp in new_bps:
                incr_bytes_pair(bp, new_vocab, vocab_count)

    return (vocab, merges)
