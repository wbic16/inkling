"""Exact zero-drift HuggingFace <-> safetensors checkpoint exporter and allowlist guard."""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Dict, List, Mapping, Optional, Tuple
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORBIDDEN_SUBSTRINGS = (
    "experts",
    "visual",
    "embed_tokens",
    "lm_head",
    "scale",
)


def normalize_param_name(name: str) -> str:
    """Normalize shorthand layer names (e.g. 'layers.6...') to canonical HF 'model.layers.6...'."""
    if name.startswith("layers.") or name in ("message_input.query", "final_input.query"):
        return "model." + name
    return name


def is_editable_param(name: str) -> bool:
    """Return True iff name is one of the unquantized BF16/FP32 control/attention/router tensors."""
    n = normalize_param_name(name)
    if any(bad in n for bad in FORBIDDEN_SUBSTRINGS):
        return False
    if n in ("model.message_input.query", "model.final_input.query"):
        return True
    if not n.startswith("model.layers."):
        return False
    allowed_suffixes = (
        ".attention_input.query",
        ".mlp_input.query",
        ".self_attn.o_proj.linear.weight",
        ".self_attn.o_proj.gate_proj.weight",
        ".self_attn.q_proj.linear.weight",
        ".self_attn.k_proj.linear.weight",
        ".self_attn.v_proj.linear.weight",
        ".post_attention_convolution.conv1d.weight",
        ".post_mlp_convolution.conv1d.weight",
        ".mlp.gate.weight",
        ".mlp.gate.e_score_correction_bias",
    )
    return ("_message_gate." in n and n.endswith(".weight")) or n.endswith(allowed_suffixes)


def transformers_name_to_ckpt_deltas(
    tf_name: str, delta: torch.Tensor
) -> List[Tuple[str, Optional[slice], torch.Tensor]]:
    """Map a HuggingFace Inkling2LoopForCausalLM parameter delta back to checkpoint safetensors keys.

    Returns a list of (checkpoint_key, row_slice_or_None, delta_to_add).
    """
    tf_name = normalize_param_name(tf_name)
    if not is_editable_param(tf_name):
        raise ValueError(f"Refusing to map non-editable or forbidden parameter: {tf_name}")

    d = delta.detach().float().cpu()
    if tf_name == "model.message_input.query":
        return [("model.llm.middle_cycle_cross_pass_attn_res.query", None, d)]
    if tf_name == "model.final_input.query":
        return [("model.llm.attn_res.layers.24.attention_input.query", None, d)]

    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", tf_name)
    if not m:
        raise ValueError(f"Unsupported parameter name: {tf_name}")
    lidx, rest = int(m.group(1)), m.group(2)

    if rest == "attention_input.query":
        if lidx == 0:
            raise ValueError("Layer 0 has no attention_input.query in checkpoint.")
        return [(f"model.llm.attn_res.layers.{lidx}.attention_input.query", None, d)]
    if rest == "mlp_input.query":
        return [(f"model.llm.attn_res.layers.{lidx}.mlp_input.query", None, d)]
    if rest == "self_attn.o_proj.gate_proj.weight":
        return [(f"model.llm.layers.{lidx}.attn.wo_ud.weight", slice(0, 3072), d)]
    if rest == "self_attn.o_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.wo_ud.weight", slice(3072, 6144), d.T.contiguous())]
    if rest == "self_attn.q_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.qkvr.weight", slice(0, 3072), d)]
    if rest == "self_attn.k_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.qkvr.weight", slice(3072, 4096), d)]
    if rest == "self_attn.v_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.qkvr.weight", slice(4096, 5120), d)]
    if rest == "post_attention_convolution.conv1d.weight":
        return [(f"model.llm.layers.{lidx}.attn_sconv.conv.weight", None, d)]
    if rest == "post_mlp_convolution.conv1d.weight":
        return [(f"model.llm.layers.{lidx}.mlp_sconv.conv.weight", None, d)]
    if rest == "mlp.gate.weight":
        return [(f"model.llm.layers.{lidx}.moe.router.weight", None, d)]
    if rest == "mlp.gate.e_score_correction_bias":
        return [(f"model.llm.layers.{lidx}.moe.router_bias", None, d)]

    mg = re.match(r"^(q|k|v|mlp)_message_gate\.(down_proj|forget_gate_proj|message_gate_proj)\.weight$", rest)
    if mg:
        if not (6 <= lidx < 18):
            raise ValueError(f"MessageGate only exists on middle loop layers 6..17, got L{lidx}")
        kind, proj = mg.group(1), mg.group(2)
        loop_off = lidx - 6
        if proj == "down_proj":
            return [(f"model.llm.middle_cycle_cross_pass_gates.{loop_off}.{kind}_gate.down_proj.weight", None, d)]
        if proj == "forget_gate_proj":
            return [(f"model.llm.middle_cycle_cross_pass_gates.{loop_off}.{kind}_gate.up_proj.weight", slice(0, 2304), d)]
        return [(f"model.llm.middle_cycle_cross_pass_gates.{loop_off}.{kind}_gate.up_proj.weight", slice(2304, 4608), d)]

    raise ValueError(f"Unmapped editable parameter: {tf_name}")


def apply_weight_deltas(
    model: torch.nn.Module, deltas: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """In-place apply FP32 deltas to model parameters/buffers, returning exact CPU backups."""
    named = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    backups: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for raw_k, d in deltas.items():
            k = normalize_param_name(raw_k)
            if k not in named:
                raise KeyError(f"Parameter '{k}' not present in model.")
            t = named[k]
            backups[k] = t.detach().cpu().clone()
            t.data.copy_((t.data.float() + d.to(t.device, dtype=torch.float32)).to(t.dtype))
    return backups


def restore_weights(model: torch.nn.Module, backups: Mapping[str, torch.Tensor]) -> None:
    """Restore model parameters/buffers from backups returned by apply_weight_deltas."""
    named = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    with torch.no_grad():
        for k, orig in backups.items():
            named[k].data.copy_(orig.to(named[k].device, dtype=named[k].dtype))


def export_edited_checkpoint(
    base_dir: str,
    out_dir: str,
    deltas_dict: Mapping[str, torch.Tensor],
    safetensors_filename: str = "model-00001-of-00001.safetensors",
) -> List[Dict[str, object]]:
    """Export a permanent identical-parameter-count safetensors checkpoint with zero expert drift."""
    os.makedirs(out_dir, exist_ok=True)
    for item in os.listdir(base_dir):
        if item in (safetensors_filename, ".cache"):
            continue
        src, dst = os.path.join(base_dir, item), os.path.join(out_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    src_st = os.path.join(base_dir, safetensors_filename)
    dst_st = os.path.join(out_dir, safetensors_filename)
    with safe_open(src_st, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    ckpt_edits: Dict[str, List[Tuple[Optional[slice], torch.Tensor, str]]] = {}
    for tf_name, delta in deltas_dict.items():
        for ck, sl, d in transformers_name_to_ckpt_deltas(tf_name, delta):
            ckpt_edits.setdefault(ck, []).append((sl, d, normalize_param_name(tf_name)))

    report: List[Dict[str, object]] = []
    for ck, edit_list in ckpt_edits.items():
        if ck not in tensors:
            raise KeyError(f"Checkpoint key '{ck}' missing from {src_st}.")
        orig = tensors[ck]
        if orig.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise TypeError(f"Refusing to modify quantized tensor '{ck}' (dtype={orig.dtype}).")
        updated = orig.float().clone()
        for sl, d, _ in edit_list:
            if sl is None:
                if tuple(updated.shape) != tuple(d.shape):
                    raise ValueError(f"Shape mismatch on '{ck}': {tuple(updated.shape)} vs {tuple(d.shape)}")
                updated += d
            else:
                if tuple(updated[sl].shape) != tuple(d.shape):
                    raise ValueError(f"Slice shape mismatch on '{ck}'[{sl}]: {tuple(updated[sl].shape)} vs {tuple(d.shape)}")
                updated[sl] += d
        diff = updated - orig.float()
        rel = diff.norm().item() / (orig.float().norm().item() + 1e-12)
        tensors[ck] = updated.to(orig.dtype)
        report.append({
            "ckpt_key": ck,
            "dtype": str(orig.dtype),
            "shape": list(orig.shape),
            "rel_change": round(rel, 6),
            "sources": [src_name for _, _, src_name in edit_list],
        })

    save_file(tensors, dst_st, metadata=metadata)
    with open(os.path.join(out_dir, "weight_edit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
