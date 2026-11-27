#!/usr/bin/env python3
"""Orientation anchor: dequantize source-GGUF tensors and compare with the base
checkpoint tensors they were converted from, in both orientations.
Q8_0: blocks of 32 = f16 scale + 32 int8. F32: raw.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/rse-fpsolve"))
import transport_gguf as tg
from safetensors import safe_open
import torch

SRC = "/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-AD-Q4_K.gguf"
CKPT = os.path.expanduser("~/inkling-hf-base/model-00001-of-00001.safetensors")
g = tg.parse_gguf(SRC)
by = {t["name"]: t for t in g["tensors"]}

def dequant(t):
    raw = g["buf"][t["data_off"]:t["data_off"] + t["nbytes"]]
    dims = t["dims"]  # gguf (in, out) order
    if t["type"] == 0:
        return np.frombuffer(raw, dtype="<f4").reshape(dims)
    if t["type"] == 8:  # Q8_0
        n = 1
        for d in dims: n *= d
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
        scales = blocks[:, :2].copy().view("<f2").astype(np.float32).ravel()
        qs = blocks[:, 2:].view(np.int8).astype(np.float32)
        return (qs * scales[:, None]).ravel()[:n].reshape(dims)
    raise ValueError(t["type"])

def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

with safe_open(CKPT, framework="pt", device="cpu") as f:
    cases = [
        ("blk.6.loop_q_ctrl_down.weight", "model.llm.layers.6.controllers.attn.down_weight"),
        ("blk.6.loop_q_ctrl_forget.weight", "model.llm.layers.6.controllers.attn.forget_weight"),
        ("blk.0.attn_output.weight", "model.llm.layers.0.attn.wo_ud.weight"),
        ("blk.0.ffn_gate_inp.weight", "model.llm.layers.0.mlp.gate.weight"),
    ]
    for gname, cname in cases:
        t = by[gname]
        M = dequant(t)  # (in, out) gguf layout
        C = f.get_tensor(cname).float().numpy()  # torch (out, in)
        print(f"{gname}: gguf_dims={t['dims']} type={t['type']} ckpt_shape={C.shape}")
        print(f"    cos(M , C)   = {cos(M, C):.6f}")
        print(f"    cos(M.T, C)  = {cos(M.T, C):.6f}")
print("ORIENTATION CHECK DONE")
