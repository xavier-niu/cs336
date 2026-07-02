from basics.bpe_tokenizer.tokenizer import compute_bpe


def test_compute_bpe():
    vocab_map = {
        (b"l", b"o", b"w"): 5,
        (b"l", b"o"): 2,
    }
    tokens = []

    # round1: l,o -> lo
    compute_bpe(vocab_map, tokens)

    expected_vocab_map = {
        (b"lo", b"w"): 5,
        (b"lo",): 2,
    }
    expected_tokens = [(b"l", b"o")]

    assert vocab_map == expected_vocab_map
    assert tokens == expected_tokens

    # round2: lo,w -> low
    compute_bpe(vocab_map, tokens)

    expected_vocab_map = {
        (b"lo",): 2,
        (b"low",): 5,
    }
    expected_tokens = [(b"l", b"o"), (b"lo", b"w")]

    assert vocab_map == expected_vocab_map
    assert tokens == expected_tokens

    # round3: nothing changed
    compute_bpe(vocab_map, tokens)
    expected_vocab_map = {
        (b"lo",): 2,
        (b"low",): 5,
    }
    expected_tokens = [(b"l", b"o"), (b"lo", b"w")]

    assert vocab_map == expected_vocab_map
    assert tokens == expected_tokens
