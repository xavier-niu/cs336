# CS336 - Language Modeling from Scratch

## Daily Progress

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