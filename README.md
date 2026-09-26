# CS336 - Language Modeling from Scratch

## Daily Progress

### 2026-09-25

- Started the Transformer LM with `basics/transformer/layers.py`: a `Linear`
  module subclassing `nn.Module`, with the weight stored as `W` of shape
  `(d_out, d_in)` (not `W^T`) and applied with a named-dimension
  `einops.einsum`, so no explicit transpose is needed.
- Worked through the conventions behind the layout: math's `y = Wx` vs
  PyTorch's row-vector `y = xW^T`, and why row-major memory order is a
  separate concept — storing `W` keeps `d_in` contiguous, matches
  `nn.Linear`, and lets reference checkpoints load as-is (a transposed
  weight would load silently wrong whenever `d_in == d_out`).
- Initialized the weight with truncated normal Xavier/Glorot init. Fixed a
  bug where the variance `2/(d_in+d_out)` was passed as `std`, making the
  weights ~28x too small; `std` is its square root, with cutoffs at ±3σ.
- Learned the `nn.Parameter` mechanics: wrapping registers the tensor with
  the module (so it reaches `parameters()`, `state_dict()` and `.to()`),
  `device`/`dtype` of `None` fall through to PyTorch defaults, and
  `backward` fills `.grad` while the optimizer performs the update.
- Wired `run_linear` to construct the layer and load reference weights via
  `load_state_dict({"weight": ...})` rather than a custom constructor
  argument; named the attribute `weight` so later nested state-dict keys
  match. `test_linear` passes.
- Next: the `Embedding` module.

<details>
<summary>Earlier updates</summary>

### 2026-08-30

- Measured compression ratios over sampled documents: 4.09 bytes/token on
  TinyStories and 4.38 on OWT with their own tokenizers. Cross-applying them
  showed the asymmetry — OWT's tokenizer still handles TinyStories text well
  (4.10), but TinyStories' narrow vocabulary collapses on web text (2.87).
- Profiled `encode` and found the bottleneck was not the O(n^2) merge loop:
  pre-tokens average 4 bytes, so the quadratic rescan costs ~3 pair checks per
  input byte. The real waste was repetition — only 1.2% of pre-tokens are
  distinct, so the same merge sequence was recomputed hundreds of thousands
  of times. Added a per-instance cache (per-instance so two tokenizers in one
  process cannot share entries) and moved `encode` to a `str` pipeline,
  removing the bytes/str round trip. Throughput went 803 KB/s to ~5.6 MB/s,
  putting the full 11.9 GB OWT pass at roughly half an hour.
- Fixed two bugs the tests caught: `special_tokens_pat_str` computed its
  pattern but never returned it, and re-keying `vocab_rmap` by `str` fails
  because single bytes 0x80-0xFF are not valid UTF-8 on their own. Only
  special tokens are guaranteed round-trippable, so the map stays bytes-keyed.
- Added `tokenize_dataset.py` to serialize datasets as `uint16`: ids stream
  into a reusable pre-allocated block buffer that flushes with `tofile`, so
  memory stays flat regardless of corpus size, with a tqdm bar measuring
  progress on the input side (`encode_iterable` yields ids, which say nothing
  about file position, and `tell()` is disabled during text-mode iteration).
- Renamed the script from `tokenize.py`, which shadowed the stdlib module and
  broke `numpy` via `inspect` when run by path.
- Added unit coverage for the serializer — round-trip, separator counts,
  awkward Unicode, empty input, and vocab-range checks; 77 tests pass.
- Next: run the full TinyStories and OWT tokenization, and answer why `uint16`
  is the right dtype.

### 2026-08-11

- Implemented the byte-level `Tokenizer` core: vocabulary lookup maps,
  merge-rank application to a fixed point, file loading, and GPT-2 regex
  pre-tokenization so merges cannot cross letter/punctuation boundaries.
- Added longest-first special-token splitting and encoding, plus decoding that
  concatenates token bytes before UTF-8 conversion so multi-byte Unicode and
  merged tokens round-trip correctly.
- Moved shared tokenizer constants and regex helpers into the package and
  pre-tokenizer modules for reuse between training and encoding.
- Added focused unit coverage for merge priority, repeated and chained merges,
  ASCII/Unicode input, pre-token boundaries, overlapping special tokens, and
  decode behavior; all 25 focused tokenizer tests pass.
- Next: implement streaming `encode_iterable` and wire the assignment adapter
  to the tokenizer implementation.

### 2026-07-19

- Rewrote the `compute_bpe` merge loop around a stable integer-id
  representation: `vocab_map` is now `id -> [bytes, count]` and `bp_map`'s
  per-pair word-sets store ids instead of byte tuples. A merge only rewrites
  the affected word's value in place, so unchanged pairs' membership sets need
  no update — this removes the per-merge set churn that made OWT intractable.
- Replaced the positional neighbour update with a `Counter`-based multiset
  diff of the word's old vs new pairs. This handles overlapping/adjacent
  merges correctly (e.g. `(a,a,a,a) -> (aa,aa)`) where the positional version
  silently dropped the pair formed between two merged tokens.
- Fixed a latent set-corruption bug: eviction of a word id from a pair's set
  is now gated on the pair being *fully absent* from the new word
  (`pair not in new_bps`), not merely decreased in count — a pair can drop
  from 2→1 occurrences and must keep its id (partial-survivor case).
- `test_train_bpe`, `test_train_bpe_special_tokens`, and
  `test_train_bpe_speed` all pass (~1.5s).
- Validated at scale: trained BPE on OpenWebText (vocab 32,000) in 545s
  (~9 min) at 7.54 GB peak RSS. The old SortedDict version had projected
  30–93h and climbed toward 12 GB. Per-merge rate rises from the frequent
  early pairs (~0.3/s) to ~63 it/s overall as `vocab_set` sizes collapse.

### 2026-07-13

- Fixed a memory leak in the BPE merge loop: empty frequency buckets in
  the `SortedDict` were never deleted; now `incr`/`decr` remove a bucket
  when it becomes empty (bounded memory, verified on OWT-valid at ~0.96 GB).
- Added a tqdm progress bar to `compute_bpe` (counts up to the target
  vocab size) and a `--size` CLI arg to `main.py`.
- Downloaded the OpenWebText sample and started the 32k-vocab run, but
  hit a performance wall: on OWT's ~6.4M unique pre-tokens the per-pair
  `SortedDict` operations dominate (~33h ETA, climbing memory).
- Next: optimize the merge loop for huge datasets (replace the sorted
  frequency structure with a plain dict + decrementing running max).

### 2026-07-06

- Trained the full byte-level BPE tokenizer on TinyStories (vocab 10,000)
  via `main.py`; ~1.9 min and ~0.24 GB peak RSS, well within limits.
- Serialized vocab and merges to `outputs/` (gitignored) and added
  `analyze.py` to inspect the result.
- Longest tokens are 15-byte whole words (` accomplishment`,
  ` disappointment`, ` responsibility`) — expected, since BPE merges
  frequent long sequences into single tokens.
- Documented the training results in the module README.

### 2026-07-04

- Rewrote the BPE merge loop to be incremental: build pair counts once,
  then per merge only touch the affected keys instead of recounting the
  whole corpus every iteration.
- Added a `pair -> [count, key-set]` map plus a `SortedDict` of
  `frequency -> pairs` for O(log n) max selection.
- Replaced the buggy neighbour-classification update with a clean
  subtract-old-pairs / add-new-pairs pass (correct for adjacency,
  overlap, and repeated bigrams).
- All three `test_train_bpe` tests pass, including the speed test:
  training `corpus.en` now takes ~0.47s (limit 1.5s), down from ~3.2s.

### 2026-07-02

- Completed `train`: fixed the stop condition (no infinite loop when merges
  run out), added special tokens to the vocab from the input list, and
  relaxed the chunk-count assertion for small inputs.
- Wired the `run_train_bpe` adapter to `train`.
- `test_train_bpe` and `test_train_bpe_special_tokens` pass — merges and
  vocab match the reference exactly.
- `test_train_bpe_speed` still fails (~3.2s vs 1.5s limit): the merge loop
  does a full pair recount every iteration. Next: incremental pair-count
  updates (reuse the per-pair key-sets) instead of recounting each merge.

### 2026-07-01

- Implemented the `compute_bpe` merge iteration: weighted pair counting
  with a pair → (count, key-set) map, max selection with the
  lexicographic tie-break, and applying the chosen merge to affected keys.
- Fixed the merge-application loop (a `for i in range()` with `i += 1`
  doesn't skip — switched to a `while` loop advancing by 2 on a match).
- Added a unit test for `compute_bpe` and worked through expected-value
  gotchas (1-element tuples need the trailing comma).

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

</details>

## Resources

- Official website: https://cs336.stanford.edu/
- YouTube playlist: https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV
- Assignment #1: https://github.com/stanford-cs336/assignment1-basics
