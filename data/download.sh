#!/usr/bin/env bash
# Download the CS336 assignment 1 datasets (pre-flattened .txt with <|endoftext|>
# separators). Run from the repo root:  bash data/download.sh
# Re-running is safe: wget -c resumes/skips already-downloaded files.
set -euo pipefail

cd "$(dirname "$0")"

TS_BASE="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
OWT_BASE="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main"

echo "==> TinyStories (V2-GPT4)"
wget -c "$TS_BASE/TinyStoriesV2-GPT4-train.txt"
wget -c "$TS_BASE/TinyStoriesV2-GPT4-valid.txt"

echo "==> OpenWebText sample"
wget -c "$OWT_BASE/owt_train.txt.gz"
wget -c "$OWT_BASE/owt_valid.txt.gz"
gunzip -kf owt_train.txt.gz
gunzip -kf owt_valid.txt.gz

echo "==> Done. Files in $(pwd):"
ls -lh ./*.txt
