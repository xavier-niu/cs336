import pickle
import sys
from typing import Iterable, Iterator

import regex

from basics.bpe_tokenizer import SPLIT_SPECIAL_TOKEN
from basics.bpe_tokenizer.pretokenizer import PRETOKEN_PAT, special_tokens_pat


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # vocab map: token id -> bytes
        self.vocab_map = vocab
        # vocab reversed map: bytes -> token id
        self.vocab_rmap = {b: idx for idx, b in self.vocab_map.items()}
        # merges set
        self.merges = {merge: idx for idx, merge in enumerate(merges)}
        # speical tokens
        if special_tokens is None:
            self.speical_tokens = []
        else:
            self.speical_tokens = special_tokens
        if SPLIT_SPECIAL_TOKEN not in self.speical_tokens:
            self.speical_tokens.append(SPLIT_SPECIAL_TOKEN)

        assert 256 + len(self.merges) + len(self.speical_tokens) == len(self.vocab_map)

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def text_to_token_ids(self, text: str) -> list[int]:
        btext = text.encode("utf-8")
        btext_array = [btext[i : i + 1] for i in range(len(btext))]

        while True:
            min_index = sys.maxsize
            for pairs in zip(btext_array[:-1], btext_array[1:]):
                if pairs in self.merges and self.merges[pairs] < min_index:
                    min_index = self.merges[pairs]
                    merges = pairs
            # nothing can be merged
            if min_index == sys.maxsize:
                break
            i = 0
            btext_array_next = []
            while i < len(btext_array):
                if i != len(btext_array) - 1:
                    if btext_array[i] == merges[0] and btext_array[i + 1] == merges[1]:
                        btext_array_next.append(b"".join(merges))
                        i += 2
                        continue
                btext_array_next.append(btext_array[i])
                i += 1
            btext_array = btext_array_next

        return [self.vocab_rmap[b] for b in btext_array]

    def encode(self, text: str) -> list[int]:
        sp_pat = special_tokens_pat(self.speical_tokens)
        chunks = regex.split(sp_pat, text.encode("utf-8"))
        ret = []
        for chunk in chunks:
            chunk_str = chunk.decode("utf-8")
            if chunk_str in self.speical_tokens:
                ret.append(self.vocab_rmap[chunk])
            else:
                for pretoken in regex.finditer(PRETOKEN_PAT, chunk):
                    pretoken = pretoken.group()
                    ret.extend(self.text_to_token_ids(pretoken.decode("utf-8")))
        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        sp_pat = special_tokens_pat(self.speical_tokens)
        last: bytes | None = None
        for subtext in iterable:
            text = subtext.encode("utf-8")
            if last is not None:
                text = b"".join([last, text])
            chunks = regex.split(sp_pat, text)
            if len(chunks) > 0:
                last = chunks.pop()
            else:
                last = None
            for chunk in chunks:
                chunk_str = chunk.decode("utf-8")
                if chunk_str in self.speical_tokens:
                    yield self.vocab_rmap[chunk]
                else:
                    for pretoken in regex.finditer(PRETOKEN_PAT, chunk):
                        pretoken = pretoken.group()
                        tokens = self.text_to_token_ids(pretoken.decode("utf-8"))
                        for token in tokens:
                            yield token

        if last is not None:
            text = last
            chunks = regex.split(sp_pat, text)
            for chunk in chunks:
                chunk_str = chunk.decode("utf-8")
                if chunk_str in self.speical_tokens:
                    yield self.vocab_rmap[chunk]
                else:
                    for pretoken in regex.finditer(PRETOKEN_PAT, chunk):
                        pretoken = pretoken.group()
                        tokens = self.text_to_token_ids(pretoken.decode("utf-8"))
                        for token in tokens:
                            yield token

    def decode(self, ids: list[int]) -> str:
        str_bytes = [self.vocab_map[idx] for idx in ids]
        return b"".join(str_bytes).decode("utf-8", errors='replace')
