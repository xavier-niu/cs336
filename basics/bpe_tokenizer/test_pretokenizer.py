from basics.bpe_tokenizer.pretokenizer import handle_rows


def test_handle_rows():
    data = [
        b"""rrrr yyyy tttt
gggggg""",
        b"""gggggg
test

gu
tttt""",
    ]

    actual: dict[bytes, int] = {}
    handle_rows(data, actual)
    expected = {
        b"rrrr": 1,
        b" yyyy": 1,
        b" tttt": 1,
        b"\n": 5,
        b"gggggg": 2,
        b"test": 1,
        b"gu": 1,
        b"tttt": 1,
    }
    
    assert actual == expected
