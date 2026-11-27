#!/usr/bin/env python3
import os
from safetensors import safe_open
p = "/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/deltas.safetensors"
with safe_open(p, framework="pt", device="cpu") as f:
    keys = sorted(f.keys())
print(f"deltas: {len(keys)} tensors")
l6 = [k for k in keys if ".6." in k or "layers.6." in k]
for k in keys[:20]:
    print("   ", k)
print("...")
kinds = set()
for k in keys:
    parts = k.split(".")
    kinds.add(parts[-3] if len(parts) >= 3 else k)
print("distinct submodule tails:", sorted(kinds)[:20])
