#!/usr/bin/env python3
"""Probe v2: equivalence of the P1-ported eelbaz forward (looping_port) vs stock 5.24 forward.
Pass criterion at num_passes=3 (stock depth): logits cos >= 0.999, argmax agreement >= 0.99.
Then a num_passes=4 run must execute and diverge (the unstable regime FP-Solve treats).
"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/rse-fpsolve"))
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(16)

from transformers import AutoTokenizer, Inkling2LoopForCausalLM
base = os.path.expanduser("~/inkling-hf-base")
tok = AutoTokenizer.from_pretrained(base)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = Inkling2LoopForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"}, experts_implementation="grouped_mm").eval()

prompts = [f"Explain step by step how to solve equation {k}x + {k+3} = {4*k+7}." for k in range(1, 9)]
enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
t0 = time.time()
out1 = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
print(f"stock forward {time.time()-t0:.1f}s", flush=True)
logits1 = out1.logits.float()

import looping_port
sys.modules["rse_looper.looping"] = looping_port  # fpsolve.py will pick up the port
from looping_port import LoopConfig, install_midstack_loop_forward
install_midstack_loop_forward(model, LoopConfig(num_passes=3, rk_K=1))
t0 = time.time()
out2 = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
print(f"ported forward (3 passes) {time.time()-t0:.1f}s", flush=True)
logits2 = out2.logits.float()
cos = torch.nn.functional.cosine_similarity(logits1.flatten(), logits2.flatten(), dim=0).item()
agree = (logits1.argmax(-1) == logits2.argmax(-1)).float().mean().item()
print(f"3-PASS EQUIVALENCE: cos={cos:.6f} argmax_agree={agree:.4f} maxdiff={(logits1 - logits2).abs().max().item():.4f}", flush=True)
print("EQUIVALENCE PASS" if cos >= 0.999 and agree >= 0.99 else "EQUIVALENCE FAIL", flush=True)

install_midstack_loop_forward(model, LoopConfig(num_passes=4, rk_K=1))
t0 = time.time()
out3 = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
print(f"ported forward (4 passes) {time.time()-t0:.1f}s", flush=True)
logits3 = out3.logits.float()
cos4 = torch.nn.functional.cosine_similarity(logits1.flatten(), logits3.flatten(), dim=0).item()
print(f"4-pass vs stock-3pass logits cos={cos4:.6f} (expect < 1: unstable regime)", flush=True)
print("PROBE2 DONE", flush=True)
