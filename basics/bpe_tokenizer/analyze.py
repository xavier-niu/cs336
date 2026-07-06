from pathlib import Path
import pickle

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGES_PATH = REPO_ROOT / "outputs" / "merges-TinyStoriesV2-GPT4-train.pkl"

with open(MERGES_PATH, "rb") as f:
    merges = pickle.load(f)
max_tokens = []
max_token_size = 0

for merge in merges:
    token_bytes = b"".join(merge)
    if len(token_bytes) > max_token_size:
        max_token_size = len(token_bytes)
        max_tokens = [token_bytes]
    elif len(token_bytes) == max_token_size:
        max_tokens.append(token_bytes)

print(f"max_token_size = {max_token_size}, max_tokens = {max_tokens}")
