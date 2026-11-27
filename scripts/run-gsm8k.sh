#!/bin/bash
# Fetch GSM8K test set (openai/grade-school-math) to NFS work area for the A/B benchmark.
set -u
D=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/rse-bench
mkdir -p "$D"
curl -sL --retry 3 -o "$D/gsm8k-test.jsonl" https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
wc -l "$D/gsm8k-test.jsonl"; sha256sum "$D/gsm8k-test.jsonl"
head -c 300 "$D/gsm8k-test.jsonl"; echo
echo GSM8K_FETCH_DONE
