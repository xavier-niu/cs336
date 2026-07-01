import os
from pathlib import Path

from basics.bpe_tokenizer.pretokenizer import init_vocab_map
from basics.bpe_tokenizer.tokenizer import SPLIT_SPECIAL_TOKEN


def test_init_vocab_map():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = REPO_ROOT / "data" / "test" / "vocab-test.txt"
    with open(DATA_PATH, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()

    actual_vocab_byte_tuple_map, actual_spt_map = init_vocab_map(
        str(DATA_PATH), 0, end, SPLIT_SPECIAL_TOKEN, ["<think>"]
    )

    expected = (
        {
            b"Hello": 1,
            b" Hi": 1,
            b" hi": 2,
            b"\n": 4,
            b"hi": 1,
            b" test": 1,
            b" ": 1,
            b"\n\n": 2,
            b"test": 1,
            b'"': 2,
            b"hello": 1,
            b",": 1,
            b" my": 1,
            b" friend": 1,
        },
        {b"<|endoftext|>": 2, b"<think>": 2},
    )
    
    actual_vocab_map = {}
    for k, v in actual_vocab_byte_tuple_map.items():
        new_k = b"".join(k)
        actual_vocab_map[new_k] = v
    assert actual_vocab_map == expected[0]
    assert actual_spt_map == expected[1]
