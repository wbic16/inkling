#!/usr/bin/env python3
"""Probe: is eelbaz's install_midstack_loop_forward compatible with transformers 5.24.1?
Checks symbols, then installs their forward with num_passes=3 (the stock depth) under a
driver-side Inkling2LoopCache alias and compares logits against the stock forward.
The alias is never constructed on the use_cache=False path (verified in looping.py).
"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/rse-fpsolve"))
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(16)

import transformers.models.inkling2_loop.modeling_inkling2_loop as mm
print("has Inkling2LoopCache:", hasattr(mm, "Inkling2LoopCache"))
print("has get_relative_distance:", hasattr(mm, "get_relative_distance"))
import transformers.masking_utils as mu
for n in ("create_causal_mask", "create_recurrent_attention_mask", "create_sliding_window_causal_mask"):
    print("masking", n, hasattr(mu, n))

from transformers import AutoTokenizer, Inkling2LoopForCausalLM, DynamicCache
base = os.path.expanduser("~/inkling-hf-base")
tok = AutoTokenizer.from_pretrained(base)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = Inkling2LoopForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"}, experts_implementation="grouped_mm").eval()
cfg = model.model.config
print("config loop attrs:", {k: getattr(cfg, k, "<missing>") for k in ("loop_start_layer", "loop_num_layers", "loop_config", "loop_num_passes")})
print("inner attrs:", {k: hasattr(model.model, k) for k in ("embed_tokens", "layers", "norm", "message_input", "final_input")})
dl = model.model.layers[6]
print("layer attrs:", {k: hasattr(dl, k) for k in ("attention_type", "is_block_boundary")})

prompts = [f"Explain step by step how to solve equation {k}x + {k+3} = {4*k+7}." for k in range(1, 9)]
enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
t0 = time.time()
out1 = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
print(f"stock forward {time.time()-t0:.1f}s")
logits1 = out1.logits.float()

# S5 alias injection (driver-side only; rse_looper untouched)
if not hasattr(mm, "Inkling2LoopCache"):
    mm.Inkling2LoopCache = DynamicCache
    print("alias injected: Inkling2LoopCache -> DynamicCache")
# S5b: config attribute bridge — eelbaz's forward reads flat loop_start_layer /
# loop_num_layers; transformers 5.24 carries loop_config{core_start, core_end}.
lc = model.model.config.loop_config
if not hasattr(model.model.config, "loop_start_layer"):
    model.model.config.loop_start_layer = lc["core_start"]
    model.model.config.loop_num_layers = lc["core_end"] - lc["core_start"]
    print(f"config attrs set: loop_start_layer={lc['core_start']} loop_num_layers={lc['core_end'] - lc['core_start']}")
if hasattr(model, "config") and not hasattr(model.config, "loop_start_layer"):
    try:
        model.config.loop_start_layer = lc["core_start"]
        model.config.loop_num_layers = lc["core_end"] - lc["core_start"]
    except Exception as e:
        print("outer config set skipped:", e)

from rse_looper.looping import LoopConfig, install_midstack_loop_forward
install_midstack_loop_forward(model, LoopConfig(num_passes=3, rk_K=1))
t0 = time.time()
out2 = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
print(f"eelbaz forward (3 passes) {time.time()-t0:.1f}s")
logits2 = out2.logits.float() if hasattr(out2, "logits") else out2.last_hidden_state.float()
if hasattr(out2, "logits"):
    cos = torch.nn.functional.cosine_similarity(logits1.flatten(), logits2.flatten(), dim=0).item()
    am1 = logits1.argmax(-1); am2 = logits2.argmax(-1)
    print(f"logits cos={cos:.6f} argmax agreement={(am1 == am2).float().mean().item():.4f} maxdiff={(logits1 - logits2).abs().max().item():.4f}")
else:
    print("custom forward returned BaseModelOutputWithPast (no logits head applied) — outer wrapper mismatch")
print("PROBE DONE")
