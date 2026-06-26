import os
import regex
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def handle_rows(rows: list[bytes], vocab_map: dict[bytes, int]):
    PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|
           ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for row in rows:
        for match in regex.finditer(PAT, row):
            vocab = match.group()
            if vocab not in vocab_map:
                vocab_map[vocab] = 0
            vocab_map[vocab] += 1


def init_vocab_map(path: str, start: int, end: int, split_special_token: bytes) -> dict[bytes, int]:
    vocab_map: dict[bytes, int] = {}

    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start
        buf = b""
        buf_size = 4 << 20  # 4 MiB buffer

        while remaining > 0:
            data = f.read(min(buf_size, remaining))
            if not data:
                break
            remaining -= len(data)
            buf += data

            parts = buf.split(split_special_token)
            buf = parts.pop()  # Keep the last part in the buffer for the next read
            
            handle_rows(parts, vocab_map)

    print(f"vocab map has been built for range [{start}, {end}]: map_len={len(vocab_map)}")
    return vocab_map
