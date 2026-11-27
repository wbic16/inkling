#!/usr/bin/env python3
"""P2 reconstruction: apply FP-Solve deltas in TF space via eelbaz's apply_weight_deltas,
extract the 168 edited TF tensors (F32) for GGUF transport, write a per-tensor
rel-change report, and save the edited model as an HF checkpoint (TF layout).

Bypasses eelbaz's checkpoint-space name mapping, which targets a checkpoint layout
(fused wo_ud [6144,3072], middle_cycle_cross_pass_gates.*) that the published
thinkingmachines/Inkling-2-Mini checkpoint does not use (published: per-layer
controllers.{attn,k,v,mlp}.{down,forget,gain}_weight, wo_ud [2304,3072],
mlp.gate.weight). Their exporter fails on the published checkpoint with KeyError.
"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/rse-fpsolve"))
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(16)
from safetensors.torch import load_file, save_file
from transformers import Inkling2LoopForCausalLM
from rse_looper.checkpoint import apply_weight_deltas

STAGE = "/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging"
base = os.path.expanduser("~/inkling-hf-base")
t0 = time.time()
model = Inkling2LoopForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"}, experts_implementation="grouped_mm").eval()
print(f"model loaded {time.time()-t0:.0f}s", flush=True)

deltas = load_file(os.path.join(STAGE, "deltas.safetensors"))
print(f"deltas: {len(deltas)}", flush=True)
backups = apply_weight_deltas(model, deltas)
print(f"deltas applied to {len(backups)} params", flush=True)

named = dict(model.named_parameters())
edited = {}
report = []
for name, orig in backups.items():
    now = named[name].detach().float().cpu()
    d = deltas[name].float().cpu() if name in deltas else (now - orig.float())
    rel = (d.norm() / (orig.float().norm() + 1e-12)).item()
    edited[name] = now.clone()
    report.append({"tf_name": name, "shape": list(now.shape), "rel_change": round(rel, 6)})
save_file(edited, os.path.join(STAGE, "edited_tf.safetensors"), metadata={"format": "pt"})
json.dump(report, open(os.path.join(STAGE, "weight_edit_report_tf.json"), "w"), indent=2)
print(f"edited_tf.safetensors: {len(edited)} tensors; mean rel {sum(r['rel_change'] for r in report)/len(report):.4f}", flush=True)

out = os.path.expanduser("~/rse-fpsolve/edited-hf-tf")
model.save_pretrained(out)
print(f"edited HF checkpoint (TF layout) saved to {out}", flush=True)
print("RECONSTRUCT DONE", flush=True)
