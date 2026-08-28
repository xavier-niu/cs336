import argparse
from collections import Counter
import logging
from multiprocessing.pool import Pool
import os
from pathlib import Path
import pickle
import resource
import time

from tqdm import tqdm

from basics.bpe_tokenizer import SPLIT_SPECIAL_TOKEN, BytesPair, BytesTuple
from basics.bpe_tokenizer.pretokenizer import find_chunk_boundaries, init_vocab_map


logger = logging.getLogger(__name__)


def train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_token=SPLIT_SPECIAL_TOKEN,
    nproc=None,
    pretoken_cache_path: Path | None = None,
) -> tuple[dict[int, bytes], list[BytesPair]]:
    if split_token not in special_tokens:
        logger.info(f"split_token ({split_token}) does not exist in special_tokens, will append it")
        special_tokens.append(split_token)

    if pretoken_cache_path is not None and pretoken_cache_path.exists():
        with open(pretoken_cache_path, "rb") as f:
            vocab_map = pickle.load(f)
            logger.info("vocab_map is restored from cache")
    else:
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
            for partial_vocab in vocab_map_partial:
                for byte_tuple, count in partial_vocab.items():
                    if byte_tuple not in vocab_map:
                        vocab_map[byte_tuple] = 0
                    vocab_map[byte_tuple] += count
            logger.info(f"vocab map has been built: vocab_map_size={len(vocab_map)}")

        if pretoken_cache_path is not None:
            with open(pretoken_cache_path, "wb") as f:
                pickle.dump(vocab_map, f, protocol=pickle.HIGHEST_PROTOCOL)

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

    vocab_map = {
        idx: [vocab_bytes, count] for idx, (vocab_bytes, count) in enumerate(vocab_map.items())
    }
    merges_len = vocab_size - len(vocab)
    merges: list[BytesPair] = []

    # init bp_map
    # value tuple includes
    # - int: frequency
    # - set[int]: vocab index set
    bp_map: dict[BytesPair, tuple[int, set[int]]] = {}
    # frequency -> set(BytesPair)
    freq_to_bpset_map: dict[int, set[BytesPair]] = {}
    max_freq = 0

    # init bp_map
    for idx, (vocab_bytes, count) in vocab_map.items():
        for a, b in zip(vocab_bytes[:-1], vocab_bytes[1:]):
            entry = bp_map.setdefault((a, b), [0, set()])
            entry[0] += count
            entry[1].add(idx)
            if max_freq < entry[0]:
                max_freq = entry[0]

    for bp, (count, _) in bp_map.items():
        freq_to_bpset_map.setdefault(count, set()).add(bp)

    with tqdm(total=merges_len, desc="computing bpe tokenizer") as pbar:
        # exit conditions:
        # 1. reach upper limit of merges
        # 2. nothing to be merged
        while len(merges) < merges_len and len(freq_to_bpset_map) > 0:
            while max_freq not in freq_to_bpset_map and max_freq > 0:
                max_freq -= 1

            if max_freq <= 0:
                break

            bp_set = freq_to_bpset_map[max_freq]

            max_bp = max(bp_set)
            new_bytes = b"".join(max_bp)
            merges.append(max_bp)
            vocab[token_id] = new_bytes
            token_id += 1

            pbar.update(1)

            # vocab_set includes vocabs containing max_bp
            (_, vocab_set) = bp_map[max_bp]
            vocab_set = [x for x in vocab_set]

            # bp -> (origin_freq, diff)
            bp_freq_diff: dict[BytesPair, tuple[int, int]] = {}
            # vocabs that need to be updated: id -> new vocab_bytes
            vocab_id_set: dict[int, BytesTuple] = {}

            # iterate all vocabs of max_bp
            for vocab_idx in vocab_set:
                (old_vocab, vcount) = vocab_map[vocab_idx]
                i = 0
                new_vocab = []

                # update new_vocab
                while i < len(old_vocab):
                    if i != len(old_vocab) - 1 and (old_vocab[i], old_vocab[i + 1]) == max_bp:
                        new_vocab.append(new_bytes)
                        i += 2
                        continue
                    new_vocab.append(old_vocab[i])
                    i += 1

                # update vocab_id_set
                vocab_id_set[vocab_idx] = tuple(new_vocab)

                old_bps = Counter(zip(old_vocab[:-1], old_vocab[1:]))
                new_bps = Counter(zip(new_vocab[:-1], new_vocab[1:]))

                rm_bps = old_bps - new_bps
                add_bps = new_bps - old_bps

                # remove bytes-pairs
                # 1. update bp_freq_diff
                # 2. remove vocab_idx from bp_map
                for rm_bp, n in rm_bps.items():
                    if rm_bp == max_bp:
                        continue
                    orig_bp_count, vocab_set1 = bp_map[rm_bp]
                    if n > 0 and rm_bp not in new_bps:
                        vocab_set1.discard(vocab_idx)
                    entry = bp_freq_diff.setdefault(rm_bp, [orig_bp_count, 0])
                    entry[1] -= n * vcount

                # add bytes-pairs
                for add_bp, n in add_bps.items():
                    entry = bp_map.setdefault(add_bp, [0, set()])
                    entry[1].add(vocab_idx)
                    entry = bp_freq_diff.setdefault(add_bp, [entry[0], 0])
                    entry[1] += n * vcount

            # remove merged bp from bp_map and freq_to_bpset_map
            (max_bp_count, _) = bp_map.pop(max_bp)
            if len(freq_to_bpset_map[max_bp_count]) <= 1:
                del freq_to_bpset_map[max_bp_count]
            else:
                freq_to_bpset_map[max_bp_count].discard(max_bp)

            # update bp_map and freq_to_bpset_map
            for bp, (orig, diff) in bp_freq_diff.items():
                actual = orig + diff
                if actual == 0:
                    del bp_map[bp]
                else:
                    bp_map[bp][0] = actual

                # skip if no changes
                if orig == actual:
                    continue

                # update orig
                if orig == 0:
                    freq_to_bpset_map.setdefault(actual, set()).add(bp)
                else:
                    if len(freq_to_bpset_map[orig]) <= 1:
                        del freq_to_bpset_map[orig]
                    else:
                        freq_to_bpset_map[orig].discard(bp)

                # update actual
                if actual > 0:
                    freq_to_bpset_map.setdefault(actual, set()).add(bp)

            # update vocab_map
            for vocab_idx, new_vocab in vocab_id_set.items():
                vocab_map[vocab_idx][0] = new_vocab

    return (vocab, merges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE on a dataset")
    parser.add_argument(
        "--dataset", required=True, help="dataset name, e.g. TinyStoriesV2-GPT4-train"
    )
    parser.add_argument(
        "--size", help="maximum vocabulary size, default is 10000", default=10000, type=int
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / f"{args.dataset}.txt"
    vocab_path = repo_root / "outputs" / f"vocab-{args.dataset}.pkl"
    merges_path = repo_root / "outputs" / f"merges-{args.dataset}.pkl"
    pretoken_cache_path = repo_root / "data" / f"pretoken-cache-{args.dataset}.txt"

    if not data_path.exists():
        raise SystemExit(f"{data_path} not found — run `bash data/download.sh` first")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    start = time.perf_counter()
    vocab, merges = train(str(data_path), args.size, [], pretoken_cache_path=pretoken_cache_path)
    elapsed = time.perf_counter() - start
    peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    print("BPE tokenizer train is completed")
    print(f"time:            {elapsed:.1f}s")
    print(f"peak RSS (main): {peak_self / 1024**2:.2f} GB")
    print(f"peak RSS (worker):{peak_kids / 1024**2:.2f} GB")
    
    if not vocab_path.parent.exists():
        vocab_path.parent.mkdir()

    # Serialize vocab and merges
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    main()
