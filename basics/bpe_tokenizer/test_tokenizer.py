import signal
from collections.abc import Sequence
from typing import Any

import pytest

from basics.bpe_tokenizer import SPLIT_SPECIAL_TOKEN
from basics.bpe_tokenizer.tokenzier import Tokenizer


def make_tokenizer(
    merges: Sequence[tuple[bytes, bytes]] = (),
    special_tokens: Sequence[str] = (),
) -> Tokenizer:
    """Build the smallest valid byte-level tokenizer for a unit test."""
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}

    for left, right in merges:
        merged_token = left + right
        if merged_token not in vocab.values():
            vocab[len(vocab)] = merged_token

    all_special_tokens = list(special_tokens)
    if SPLIT_SPECIAL_TOKEN not in all_special_tokens:
        all_special_tokens.append(SPLIT_SPECIAL_TOKEN)
    for special_token in all_special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return Tokenizer(vocab, list(merges), special_tokens=list(special_tokens))


def token_ids(tokenizer: Tokenizer, tokens: Sequence[bytes]) -> list[int]:
    return [tokenizer.vocab_rmap[token] for token in tokens]


def tokenize_with_timeout(tokenizer: Tokenizer, text: str) -> Any:
    """Keep a broken fixed-point loop from hanging the entire test run."""

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("text_to_token_ids did not terminate")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        return tokenizer.text_to_token_ids(text)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@pytest.mark.parametrize(
    ("text", "expected_tokens"),
    [
        pytest.param("", [], id="empty-text"),
        pytest.param("a", [b"a"], id="single-byte"),
        pytest.param("cat", [b"c", b"a", b"t"], id="multiple-unmerged-bytes"),
        pytest.param("é", [b"\xc3", b"\xa9"], id="two-byte-unicode"),
        pytest.param(
            "🙂",
            [b"\xf0", b"\x9f", b"\x99", b"\x82"],
            id="four-byte-unicode",
        ),
    ],
)
def test_text_without_merges_maps_each_utf8_byte_to_its_token_id(
    text: str, expected_tokens: list[bytes]
) -> None:
    tokenizer = make_tokenizer()

    actual = tokenize_with_timeout(tokenizer, text)

    assert actual == token_ids(tokenizer, expected_tokens)


def test_text_to_token_ids_applies_a_single_merge() -> None:
    tokenizer = make_tokenizer([(b"a", b"b")])

    actual = tokenize_with_timeout(tokenizer, "ab")

    assert actual == token_ids(tokenizer, [b"ab"])


def test_text_to_token_ids_does_not_duplicate_an_unmerged_neighbor() -> None:
    tokenizer = make_tokenizer([(b"a", b"b")])

    actual = tokenize_with_timeout(tokenizer, "abc")

    assert actual == token_ids(tokenizer, [b"ab", b"c"])


def test_text_to_token_ids_reapplies_merges_to_new_tokens() -> None:
    tokenizer = make_tokenizer([(b"a", b"b"), (b"ab", b"c")])

    actual = tokenize_with_timeout(tokenizer, "abc")

    assert actual == token_ids(tokenizer, [b"abc"])


def test_text_to_token_ids_merges_repeated_non_overlapping_pairs() -> None:
    tokenizer = make_tokenizer([(b"a", b"b")])

    actual = tokenize_with_timeout(tokenizer, "abab")

    assert actual == token_ids(tokenizer, [b"ab", b"ab"])


@pytest.mark.parametrize(
    ("merges", "expected_tokens"),
    [
        pytest.param(
            [(b"a", b"b"), (b"b", b"c")],
            [b"ab", b"c"],
            id="left-pair-has-higher-priority",
        ),
        pytest.param(
            [(b"b", b"c"), (b"a", b"b")],
            [b"a", b"bc"],
            id="right-pair-has-higher-priority",
        ),
    ],
)
def test_text_to_token_ids_uses_merge_list_order_as_priority(
    merges: list[tuple[bytes, bytes]], expected_tokens: list[bytes]
) -> None:
    tokenizer = make_tokenizer(merges)

    actual = tokenize_with_timeout(tokenizer, "abc")

    assert actual == token_ids(tokenizer, expected_tokens)


@pytest.mark.parametrize(
    ("text", "expected_tokens"),
    [
        pytest.param("", [], id="empty-text"),
        pytest.param("cat", [b"c", b"a", b"t"], id="ascii-text"),
        pytest.param(
            "é",
            [b"\xc3", b"\xa9"],
            id="unicode-text",
        ),
    ],
)
def test_encode_encodes_ordinary_text(text: str, expected_tokens: list[bytes]) -> None:
    tokenizer = make_tokenizer()

    actual = tokenizer.encode(text)

    assert actual == token_ids(tokenizer, expected_tokens)


def test_encode_applies_bpe_merges_to_ordinary_text() -> None:
    tokenizer = make_tokenizer([(b"a", b"b"), (b"ab", b"c")])

    actual = tokenizer.encode("abc")

    assert actual == token_ids(tokenizer, [b"abc"])


def test_encode_preserves_a_special_token_as_one_token() -> None:
    special_token = "<special>"
    tokenizer = make_tokenizer(special_tokens=[special_token])

    actual = tokenizer.encode(f"a{special_token}b")

    assert actual == token_ids(
        tokenizer,
        [b"a", special_token.encode("utf-8"), b"b"],
    )


def test_encode_preserves_adjacent_special_tokens() -> None:
    special_token = "<special>"
    tokenizer = make_tokenizer(special_tokens=[special_token])

    actual = tokenizer.encode(special_token * 2)

    assert actual == token_ids(
        tokenizer,
        [special_token.encode("utf-8"), special_token.encode("utf-8")],
    )


def test_encode_prefers_the_longest_overlapping_special_token() -> None:
    short_token = "<special>"
    long_token = short_token * 2
    tokenizer = make_tokenizer(special_tokens=[short_token, long_token])

    actual = tokenizer.encode(long_token + short_token)

    assert actual == token_ids(
        tokenizer,
        [long_token.encode("utf-8"), short_token.encode("utf-8")],
    )


def test_encode_does_not_merge_across_pretoken_boundaries() -> None:
    tokenizer = make_tokenizer([(b"a", b"!")])

    actual = tokenizer.encode("a!")

    assert actual == token_ids(tokenizer, [b"a", b"!"])


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param([], "", id="empty-token-list"),
        pytest.param([b"h", b"e", b"l", b"l", b"o"], "hello", id="ascii"),
        pytest.param([b"\xc3", b"\xa9"], "é", id="two-byte-unicode"),
        pytest.param(
            [b"\xf0", b"\x9f", b"\x99", b"\x82"],
            "🙂",
            id="four-byte-unicode",
        ),
    ],
)
def test_decode_concatenates_token_bytes_before_decoding_utf8(
    tokens: list[bytes], expected: str
) -> None:
    tokenizer = make_tokenizer()

    actual = tokenizer.decode(token_ids(tokenizer, tokens))

    assert actual == expected


def test_decode_handles_merged_tokens() -> None:
    tokenizer = make_tokenizer([(b"h", b"i")])

    actual = tokenizer.decode(token_ids(tokenizer, [b"hi", b"!"]))

    assert actual == "hi!"


def test_decode_handles_special_tokens() -> None:
    special_token = "<special>"
    tokenizer = make_tokenizer(special_tokens=[special_token])

    actual = tokenizer.decode(token_ids(tokenizer, [b"a", special_token.encode("utf-8"), b"b"]))

    assert actual == f"a{special_token}b"
