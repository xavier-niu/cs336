# BPE Tokenizer

## Notes

### UTF-8

The UTF-8 encoding is variable-length

```
┌─────────────────────────┬───────┬─────────────────────────────────────┐
│       Code points       │ Bytes │            Byte pattern             │
├─────────────────────────┼───────┼─────────────────────────────────────┤
│ U+0000 – U+007F (ASCII) │ 1     │ 0xxxxxxx                            │
├─────────────────────────┼───────┼─────────────────────────────────────┤
│ U+0080 – U+07FF         │ 2     │ 110xxxxx 10xxxxxx                   │
├─────────────────────────┼───────┼─────────────────────────────────────┤
│ U+0800 – U+FFFF         │ 3     │ 1110xxxx 10xxxxxx 10xxxxxx          │
├─────────────────────────┼───────┼─────────────────────────────────────┤
│ U+10000 – U+10FFFF      │ 4     │ 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx │
└─────────────────────────┴───────┴─────────────────────────────────────┘
```

The leading byte announces the length via its leading 1-bits (then a 0):

```
┌──────────────────────────┬────────────────────────────────────────────┐
│ Leading byte starts with │                  Meaning                   │
├──────────────────────────┼────────────────────────────────────────────┤
│ 0xxxxxxx                 │ 1-byte char (ASCII). The single leading 0. │
├──────────────────────────┼────────────────────────────────────────────┤
│ 110xxxxx                 │ "I start a 2-byte char" (two 1s)           │
├──────────────────────────┼────────────────────────────────────────────┤
│ 1110xxxx                 │ "I start a 3-byte char" (three 1s)         │
├──────────────────────────┼────────────────────────────────────────────┤
│ 11110xxx                 │ "I start a 4-byte char" (four 1s)          │
└──────────────────────────┴────────────────────────────────────────────┘
```

For example

```
>>> "中".encode("utf-8")
b'\xe4\xb8\xad'
>>> bin(0xE4), bin(0xB8), bin(0xAD)
('0b11100100', '0b10111000', '0b10101101')
>>> (0xE4 & 0b1111) << 12 | (0xB8 & 0b111111) << 6 | (0xAD & 0b111111)
20013
>>> hex(20013)
'0x4e2d'      # back to U+4E2D
>>> chr(20013)
'中'
```

The maximum length is 7

```
┌─────────────────┬────────────┬───────┬─────────────────────────────┐
│  Leading byte   │ Leading 1s │ Bytes │ Free data bits in lead byte │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 0xxxxxxx        │ 0          │ 1     │ 7                           │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 110xxxxx        │ 2          │ 2     │ 5                           │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 1110xxxx        │ 3          │ 3     │ 4                           │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 11110xxx        │ 4          │ 4     │ 3                           │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 111110xx        │ 5          │ 5     │ 2                           │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 1111110x        │ 6          │ 6     │ 1                           │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 11111110 (0xFE) │ 7          │ 7     │ 0                           │
├─────────────────┼────────────┼───────┼─────────────────────────────┤
│ 11111111 (0xFF) │ 8          │ —     │ no terminating 0!           │
└─────────────────┴────────────┴───────┴─────────────────────────────┘
```

## Problem

### Unicode1

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

### Unicode2

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
