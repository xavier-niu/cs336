# BPE Tokenizer

## Problem (unicode1)

(a) What Unicode character does chr(0) return?

Return `\x00` (NUL byte).

(b) How does this character’s string representation (`__repr__()`) differ
from its printed representation?

REPL shows the unambiguous form for developer, `__str__()` shows the
"nice" form for end-users.

```
>>> x = "a\x00b"
>>> repr(x)
"'a\\x00b'"
>>> print(x)
ab
```

(c) What happens when this character occurs in text? It may be helpful
to play around with the following in your Python interpreter and see if
it matches your expectations:

```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

The NUL character occupies a byte (`\x00`) but shows nothing to
end-users, so the fourth prompt shows "this is a teststring".

## Problem (unicode2)

(a) What are some reasons to prefer training our tokenizer on UTF-8
encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to
compare the output of these encodings for various input strings.

~98% text on Internet is encoded by UTF-8. Also UTF-8 is efficiency due
to its variable-length design.

> What I am missing: Vocabulary size of the base alphabet: UTF-8 is
> represented by 256 symbols, UTF-16 (2-4 bytes) is by 65,536, UTF-32 is
> by ~1.1M symbols.

(b) Consider the following (incorrect) function, which is intended to
decode a UTF-8 byte string into a Unicode string. Why is this function
incorrect? Provide an example of an input byte string that yields
incorrect results.

The UTF-8 encoding is variable-length, so you can't decode them byte by
byte. The input string that yields an incorrect result is "中".

The correct way is

```python
def decode_utf8_bytes_to_str(bytestring: bytes):
    return bytestring.decode("utf-8")
```

(c) Give a two-byte sequence that does not decode to any Unicode
character(s).

\xff\xff: the leading 1 in the first byte must come with a zero.
