# CS336 - Language Modeling from Scratch

## Daily Progress

### 2026-06-29

- Built the BPE `train` entry point: parallel `init_vocab_map` over file
  chunks, then merges the per-process vocab and special-token maps.
- Started `compute_bpe` and the byte-tuple pre-token representation
  (`dict[tuple[bytes, ...], int]`) that the merge loop operates on.
- Sketched the merge-iteration plan (count pairs → select max → apply)
  and worked through why it should be incremental rather than
  re-spawning processes and recounting every merge.
- Studied why training returns both `vocab` and ordered `merges`, the
  encoding complexity, the pre-token concept, and bytes/tuple conversions.

### 2026-06-28

- Reworked special-token handling in pre-tokenization: a single `re.split`
  over all special tokens (escaped, longest-first) isolates them from the
  GPT-2 regex, with a separate raw-bytes set for membership.
- Fixed several bugs surfaced by review/tests: corrupted multi-line `PAT`
  dropping punctuation, escaped-vs-raw special-token matching, and the
  dropped last buffer segment (added a post-loop flush).
- Refactored chunk handling into `handle_buffer` and added a `data/test`
  fixture for `init_vocab_map`.
- Studied the GPT-2 whitespace regex in depth (`\s+(?!\S)` negative
  lookahead) and how `\n\n` tokenizes differently before a word vs alone.

### 2026-06-26

- Set up the TinyStories dataset download (`data/download.sh`) and gitignored
  the large data files.
- Drafted pre-tokenization scaffolding in `basics/bpe_tokenizer/`: chunk-boundary
  finder, buffered reader that splits on `<|endoftext|>`, and pre-token counting.
- Studied UTF-8 encoding, byte-level BPE pre-tokenization, the GPT-2 regex, and
  `multiprocessing` for parallelizing pre-tokenization.
- Started unit tests for `handle_rows` and worked through `re` vs `regex` and
  `str`/`bytes` issues surfaced by the tests.

## Resources

- Official website: https://cs336.stanford.edu/
- YouTube playlist: https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV
- Assignment #1: https://github.com/stanford-cs336/assignment1-basics