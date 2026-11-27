#!/usr/bin/env python3
"""Dump base-checkpoint key names for loop-layer control tensors (header only)."""
import os
from safetensors import safe_open
p = os.path.expanduser("~/inkling-hf-base/model-00001-of-00001.safetensors")
with safe_open(p, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print(f"total keys: {len(keys)}")
import re
for pat in ["layers.10.", "message", "middle", "llm"]:
    hits = [k for k in keys if pat in k]
    print(f"--- contains '{pat}': {len(hits)}")
    for k in hits[:14]:
        print("   ", k)
print("--- first 8 keys overall:")
for k in keys[:8]:
    print("   ", k)
