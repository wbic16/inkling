#!/usr/bin/env python3
import os, json
from safetensors import safe_open
p = os.path.expanduser("~/inkling-hf-base/model-00001-of-00001.safetensors")
with safe_open(p, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    def shape(k):
        return list(f.get_slice(k).get_shape())
    print("--- ALL layer-10 keys:")
    for k in sorted(k for k in keys if ".layers.10." in k):
        print(f"    {k}  {shape(k)}")
    print("--- keys containing 'gate':")
    for k in sorted(k for k in keys if "gate" in k)[:40]:
        print(f"    {k}  {shape(k)}")
    print("--- middle_cycle / attn_res / queries keys:")
    for k in sorted(k for k in keys if "middle" in k or "attn_res" in k or "queries" in k)[:50]:
        print(f"    {k}  {shape(k)}")
