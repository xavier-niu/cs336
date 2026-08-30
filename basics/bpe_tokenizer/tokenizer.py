import pickle
import sys
from typing import Iterable, Iterator

import regex

from basics.bpe_tokenizer import SPLIT_SPECIAL_TOKEN
from basics.bpe_tokenizer.pretokenizer import PRETOKEN_PAT_STR, special_tokens_pat_str


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # vocab map: token id -> bytes
        self.vocab_map = vocab
        # vocab reversed map: str -> token id
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

        self.sp_pat = special_tokens_pat_str(self.speical_tokens)
        self.cache: dict[str, list[int]] = {}

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
        cached = self.cache.get(text)
        if cached is not None:
            return cached
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

        tokens = [self.vocab_rmap[b] for b in btext_array]
        self.cache[text] = tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        chunks = regex.split(self.sp_pat, text)
        ret = []
        for chunk in chunks:
            if chunk in self.speical_tokens:
                ret.append(self.vocab_rmap[chunk.encode()])
            else:
                for pretoken in regex.finditer(PRETOKEN_PAT_STR, chunk):
                    pretoken = pretoken.group()
                    ret.extend(self.text_to_token_ids(pretoken))
        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        last: str | None = None
        for subtext in iterable:
            if last is not None:
                subtext = "".join([last, subtext])
            chunks = regex.split(self.sp_pat, subtext)
            if len(chunks) > 0:
                last = chunks.pop()
            else:
                last = None
            for chunk in chunks:
                if chunk in self.speical_tokens:
                    yield self.vocab_rmap[chunk.encode()]
                else:
                    for pretoken in regex.finditer(PRETOKEN_PAT_STR, chunk):
                        pretoken = pretoken.group()
                        tokens = self.text_to_token_ids(pretoken)
                        for token in tokens:
                            yield token

        if last is not None:
            text = last
            chunks = regex.split(self.sp_pat, text)
            for chunk in chunks:
                if chunk in self.speical_tokens:
                    yield self.vocab_rmap[chunk.encode()]
                else:
                    for pretoken in regex.finditer(PRETOKEN_PAT_STR, chunk):
                        pretoken = pretoken.group()
                        tokens = self.text_to_token_ids(pretoken)
                        for token in tokens:
                            yield token

    def decode(self, ids: list[int]) -> str:
        str_bytes = [self.vocab_map[idx] for idx in ids]
        return b"".join(str_bytes).decode("utf-8", errors="replace")
