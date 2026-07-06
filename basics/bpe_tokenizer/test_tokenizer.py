from basics.bpe_tokenizer.tokenizer import compute_bpe


def test_compute_bpe():
    vocab_map = {
        (b"l", b"o", b"w"): 5,
        (b"l", b"o"): 2,
    }
    special_tokens = ["<think>", "</think>"]

    expected_vocab = {}
    expected_merges = []
    token_id = 0
    for i in range(256):
        expected_vocab[token_id] = bytes([i])
        token_id += 1
    for spt in special_tokens:
        expected_vocab[token_id] = spt.encode("utf-8")
        token_id += 1

    (actual_vocab, actual_merges) = compute_bpe(vocab_map, 1000, special_tokens)

    expected_vocab[token_id] = b"lo"
    token_id += 1
    expected_vocab[token_id] = b"low"
    token_id += 1
    expected_merges = [(b"l", b"o"), (b"lo", b"w")]

    assert set(expected_vocab.keys()) == set(actual_vocab.keys())
    assert set(expected_vocab.values()) == set(actual_vocab.values())
    assert set(expected_merges) == set(actual_merges)
