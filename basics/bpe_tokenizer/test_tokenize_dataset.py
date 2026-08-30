import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from basics.bpe_tokenizer import SPLIT_SPECIAL_TOKEN
from basics.bpe_tokenizer import tokenize_dataset as tokenize
from basics.bpe_tokenizer.tokenizer import Tokenizer

DATASET = "tiny"


def build_vocab_and_merges(
    merges: list[tuple[bytes, bytes]] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """The smallest vocab ``Tokenizer.__init__``'s size assertion accepts.

    ``tokenize.main`` constructs ``Tokenizer(vocab, merges)`` without passing
    ``special_tokens``, so the tokenizer appends ``SPLIT_SPECIAL_TOKEN`` itself
    and the vocab must be ``256 + len(merges) + 1`` entries.
    """
    merges = list(merges or [])
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    for left, right in merges:
        vocab[len(vocab)] = left + right
    vocab[len(vocab)] = SPLIT_SPECIAL_TOKEN.encode("utf-8")
    return vocab, merges


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``tokenize.main`` at a throwaway repo root under ``tmp_path``."""

    class FakePath:
        """Stands in for ``Path(__file__)``; only ``.resolve().parents[2]`` is used."""

        def resolve(self) -> "FakePath":
            return self

        @property
        def parents(self) -> dict[int, Path]:
            return {2: tmp_path}

    monkeypatch.setattr(tokenize, "Path", lambda _: FakePath())
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    return tmp_path


def write_inputs(
    repo: Path,
    text: str,
    merges: list[tuple[bytes, bytes]] | None = None,
    dataset: str = DATASET,
) -> Tokenizer:
    """Lay out the data/ and outputs/ files ``main`` expects. Returns an equivalent tokenizer."""
    vocab, merges = build_vocab_and_merges(merges)

    (repo / "data" / f"{dataset}.txt").write_text(text)
    with open(repo / "outputs" / f"vocab-{dataset}.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(repo / "outputs" / f"merges-{dataset}.pkl", "wb") as f:
        pickle.dump(merges, f)

    return Tokenizer(vocab, merges)


def run_main(monkeypatch: pytest.MonkeyPatch, dataset: str = DATASET) -> None:
    monkeypatch.setattr(sys, "argv", ["tokenize", "--dataset", dataset])
    tokenize.main()


def read_tokens(repo: Path, dataset: str = DATASET) -> np.ndarray:
    """Read the serialized ids back.

    ``tofile`` writes headerless raw bytes, so the dtype has to be supplied here --
    it is not recorded in the file.
    """
    return np.fromfile(repo / "outputs" / f"tokens-{dataset}.npy", dtype=np.uint16)


def test_serialized_ids_match_a_direct_encode(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = f"hello world{SPLIT_SPECIAL_TOKEN}second document\n"
    tokenizer = write_inputs(repo, text)

    run_main(monkeypatch)

    expected = list(tokenizer.encode_iterable([text]))
    assert read_tokens(repo).tolist() == expected


def test_decoding_the_file_reproduces_the_original_text(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = f"a story{SPLIT_SPECIAL_TOKEN}another story\nwith two lines\n"
    tokenizer = write_inputs(repo, text)

    run_main(monkeypatch)

    assert tokenizer.decode(read_tokens(repo).tolist()) == text


def test_merges_are_applied(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = "abab"
    tokenizer = write_inputs(repo, text, merges=[(b"a", b"b")])

    run_main(monkeypatch)

    merged_id = tokenizer.vocab_rmap[b"ab"]
    assert read_tokens(repo).tolist() == [merged_id, merged_id]


def test_document_separators_are_preserved(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The separator must survive into the array, or document boundaries are lost."""
    documents = ["one", "two", "three", "four"]
    text = SPLIT_SPECIAL_TOKEN.join(documents)
    tokenizer = write_inputs(repo, text)

    run_main(monkeypatch)

    separator_id = tokenizer.vocab_rmap[SPLIT_SPECIAL_TOKEN.encode("utf-8")]
    tokens = read_tokens(repo)
    assert np.count_nonzero(tokens == separator_id) == len(documents) - 1


def test_a_trailing_separator_is_preserved(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = f"only document{SPLIT_SPECIAL_TOKEN}"
    tokenizer = write_inputs(repo, text)

    run_main(monkeypatch)

    separator_id = tokenizer.vocab_rmap[SPLIT_SPECIAL_TOKEN.encode("utf-8")]
    assert read_tokens(repo).tolist()[-1] == separator_id


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("plain ascii text\n", id="ascii"),
        pytest.param("café naïve\n", id="two-byte-unicode"),
        pytest.param("emoji \U0001f642 here\n", id="four-byte-unicode"),
        pytest.param("trailing space \n\n\nand blank lines\n", id="whitespace"),
        pytest.param(f"{SPLIT_SPECIAL_TOKEN}{SPLIT_SPECIAL_TOKEN}", id="adjacent-separators"),
        pytest.param("no trailing newline", id="no-trailing-newline"),
    ],
)
def test_round_trip_survives_awkward_inputs(
    repo: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    tokenizer = write_inputs(repo, text)

    run_main(monkeypatch)

    assert tokenizer.decode(read_tokens(repo).tolist()) == text


def test_empty_input_produces_an_empty_file(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_inputs(repo, "")

    run_main(monkeypatch)

    assert read_tokens(repo).size == 0


def test_output_length_is_a_whole_number_of_uint16(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial trailing element means a truncated or mis-sliced final flush."""
    write_inputs(repo, f"some text{SPLIT_SPECIAL_TOKEN}more text\n")

    run_main(monkeypatch)

    size = (repo / "outputs" / f"tokens-{DATASET}.npy").stat().st_size
    assert size % np.dtype(np.uint16).itemsize == 0


def test_every_id_is_within_the_vocab(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards against uninitialized buffer contents leaking into the output."""
    tokenizer = write_inputs(repo, f"hello{SPLIT_SPECIAL_TOKEN}world\n")

    run_main(monkeypatch)

    tokens = read_tokens(repo)
    assert tokens.dtype == np.uint16
    assert tokens.max(initial=0) < len(tokenizer.vocab_map)


def test_a_vocab_too_large_for_uint16_is_rejected(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uint16 tops out at 65535; a larger vocab must fail loudly, not wrap silently."""
    merges = [(b"a", bytes([i % 256])) for i in range(66_000 - 257)]
    vocab, merges = build_vocab_and_merges(merges)
    assert len(vocab) > np.iinfo(np.uint16).max

    (repo / "data" / f"{DATASET}.txt").write_text("a")
    with open(repo / "outputs" / f"vocab-{DATASET}.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(repo / "outputs" / f"merges-{DATASET}.pkl", "wb") as f:
        pickle.dump(merges, f)

    with pytest.raises((AssertionError, ValueError, OverflowError)):
        run_main(monkeypatch)


@pytest.mark.parametrize("missing", ["data", "vocab", "merges"])
def test_missing_inputs_raise_file_not_found(
    repo: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    write_inputs(repo, "hello\n")
    target = {
        "data": repo / "data" / f"{DATASET}.txt",
        "vocab": repo / "outputs" / f"vocab-{DATASET}.pkl",
        "merges": repo / "outputs" / f"merges-{DATASET}.pkl",
    }[missing]
    target.unlink()

    with pytest.raises(FileNotFoundError):
        run_main(monkeypatch)


def test_tokenizer_dataset_defaults_to_dataset(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting --tokenizer-dataset must reuse --dataset's tokenizer."""
    tokenizer = write_inputs(repo, "shared\n", dataset="corpus")

    monkeypatch.setattr(sys, "argv", ["tokenize", "--dataset", "corpus"])
    tokenize.main()

    tokens = read_tokens(repo, dataset="corpus")
    assert tokenizer.decode(tokens.tolist()) == "shared\n"


def test_tokenizer_dataset_can_differ_from_dataset(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encoding one corpus with another corpus's tokenizer."""
    write_inputs(repo, "unused\n", merges=[(b"a", b"b")], dataset="tokenizer_source")
    tokenizer = write_inputs(repo, "abab\n", merges=[(b"a", b"b")], dataset="tokenizer_source")
    (repo / "data" / "target.txt").write_text("abab\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["tokenize", "--dataset", "target", "--tokenizer-dataset", "tokenizer_source"],
    )
    tokenize.main()

    assert tokenizer.decode(read_tokens(repo, dataset="target").tolist()) == "abab\n"


def test_module_is_runnable_as_a_script(tmp_path: Path) -> None:
    """``python -m basics.bpe_tokenizer.tokenize`` must actually do something.

    Without an ``if __name__ == "__main__"`` guard the module defines ``main``
    and exits 0 without writing anything -- a silent no-op.
    """
    result = subprocess.run(
        [sys.executable, "-m", "basics.bpe_tokenizer.tokenize_dataset", "--dataset", "does-not-exist"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode != 0, (
        "expected a failure for a missing dataset, but the module exited 0 without running main()"
    )
