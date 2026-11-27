"""Mid-stack multi-pass looping wrappers — PORT of eelbaz/rse_looper looping.py (commit 765453d)
to the transformers 5.24.1 inkling2_loop API. Patch P1 (see PATCHES.md).

Everything below is eelbaz's code except the mechanical API adaptations inside
install_midstack_loop_forward, each marked [P1]:
  1. Inkling2LoopCache (unreleased API) -> DynamicCache (5.24 stock uses DynamicCache).
  2. config.loop_start_layer/loop_num_layers -> config.loop_layer_index_start/end.
  3. Layer call: past_key_values= -> cache=, cache_layer_idx= -> cache_offset=,
     cache_position dropped (5.24 layers don't take it), messages passed positionally
     only when not None (non-loop layers take attention_mask as 3rd positional).
  4. final_input(...) returns a tuple in 5.24 -> index [0] (as the stock forward does).
  5. `cache` kwarg from the 5.24 CausalLM wrapper is folded into past_key_values.
The FP-Solve path runs with use_cache=False, so the cache object is never constructed
and cache_offset is inert; the port is verified by a 3-pass equivalence probe against
the stock forward (probe_looping.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass(frozen=True)
class LoopConfig:
    """Configuration for mid-stack looping on a looped MoE transformer."""

    num_passes: int = 3
    rk_window: Optional[Tuple[int, int]] = None
    rk_K: int = 1
    rk_beta: float = 0.3

    def __post_init__(self) -> None:
        if self.num_passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {self.num_passes}")
        if self.rk_K < 1:
            raise ValueError(f"rk_K must be >= 1, got {self.rk_K}")
        if not (0.0 < self.rk_beta <= 1.0):
            raise ValueError(f"rk_beta must be in (0, 1], got {self.rk_beta}")
        if self.rk_window is not None:
            lo, hi = self.rk_window
            if lo > hi or lo < 0:
                raise ValueError(f"Invalid rk_window {self.rk_window}")


def snapshot_cache_layer(cache: Any, cache_idx: int) -> Optional[Dict[str, Any]]:
    """Snapshot KV and dict/tensor short-convolution cache state for a single cache slot."""
    if cache is None:
        return None
    snap: Dict[str, Any] = {}
    if hasattr(cache, "layers") and cache_idx < len(cache.layers):
        layer = cache.layers[cache_idx]
        if hasattr(layer, "keys") and layer.keys is not None:
            snap["keys"] = layer.keys.clone()
        if hasattr(layer, "values") and layer.values is not None:
            snap["values"] = layer.values.clone()
    if hasattr(cache, "conv_states") and isinstance(cache.conv_states, dict) and cache_idx in cache.conv_states:
        cs = cache.conv_states[cache_idx]
        snap["conv_state"] = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cs.items()} if isinstance(cs, dict) else cs.clone()
    elif hasattr(cache, "conv_states") and isinstance(cache.conv_states, list) and cache_idx < len(cache.conv_states):
        cs = cache.conv_states[cache_idx]
        if cs is not None:
            snap["conv_state"] = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cs.items()} if isinstance(cs, dict) else cs.clone()
    return snap


def restore_cache_layer(cache: Any, cache_idx: int, snap: Optional[Dict[str, Any]]) -> None:
    """Restore a snapshotted KV and short-convolution cache slot without mutating external refs."""
    if cache is None or snap is None:
        return
    if hasattr(cache, "layers") and cache_idx < len(cache.layers):
        layer = cache.layers[cache_idx]
        if "keys" in snap:
            layer.keys = snap["keys"].clone()
        if "values" in snap:
            layer.values = snap["values"].clone()
    if "conv_state" in snap and hasattr(cache, "conv_states"):
        cs = snap["conv_state"]
        restored = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cs.items()} if isinstance(cs, dict) else cs.clone()
        cache.conv_states[cache_idx] = restored


def run_damped_rk_layer(
    layer_fn,
    h: torch.Tensor,
    rk_K: int = 1,
    rk_beta: float = 0.3,
    cache: Any = None,
    cache_idx: int = 0,
) -> torch.Tensor:
    """Execute one decoder layer with K-stage damped Runge-Kutta sub-stepping.

    When rk_K == 1 or rk_beta == 1.0 (with K == 1), reduces bit-for-bit to layer_fn(h).
    Intermediate stages k < K-1 run on snapshotted cache state and restore before the final step.
    """
    if rk_K <= 1:
        return layer_fn(h)

    snap = snapshot_cache_layer(cache, cache_idx)
    cur = h
    for k in range(rk_K - 1):
        out = layer_fn(cur)
        cur = cur + float(rk_beta) * (out - cur)
        restore_cache_layer(cache, cache_idx, snap)
    final_out = layer_fn(cur)
    if rk_beta == 1.0:
        return final_out
    return cur + float(rk_beta) * (final_out - cur)


def install_midstack_loop_forward(model: torch.nn.Module, cfg: LoopConfig) -> None:
    """Install variable-R mid-stack recurrence and optional damped RK sub-stepping on Inkling2LoopForCausalLM."""
    from transformers import DynamicCache  # [P1-1]
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.modeling_outputs import MoeModelOutputWithPast  # [P1-7]
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        get_relative_distance,
    )
    import types

    tm = model.model if hasattr(model, "model") else model
    tm.config.loop_num_passes = cfg.num_passes
    if hasattr(model, "config"):
        model.config.loop_num_passes = cfg.num_passes

    def custom_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        # [P1-5] 5.24's CausalLM passes the cache as `cache=`
        if past_key_values is None and "cache" in kwargs:
            past_key_values = kwargs.pop("cache")
        else:
            kwargs.pop("cache", None)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)  # [P1-1]
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mk = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache": past_key_values,
            "position_ids": position_ids,
        }
        cm = {
            "full_attention": create_causal_mask(**mk),
            "sliding_attention": create_sliding_window_causal_mask(**mk),
            "linear_attention": create_recurrent_attention_mask(**mk),
        }
        rd = {
            "full_attention": get_relative_distance(inputs_embeds, past_key_values, is_sliding=False),
            "sliding_attention": get_relative_distance(inputs_embeds, past_key_values, is_sliding=True),
        }

        h = inputs_embeds
        blocks = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])
        start_l = self.config.loop_layer_index_start  # [P1-2]
        end_l = self.config.loop_layer_index_end  # [P1-2]
        n_layers = len(self.layers)

        def call_layer(lidx, cur_h, cur_b, msgs, c_idx):
            dl = self.layers[lidx]
            use_rk = cfg.rk_window is not None and cfg.rk_K > 1 and (cfg.rk_window[0] <= lidx <= cfg.rk_window[1])
            def fn(x):
                kw = dict(
                    attention_mask=cm[dl.attention_type],
                    relative_distance=rd[dl.attention_type],
                    conv_mask=cm["linear_attention"],
                    position_ids=position_ids,
                    cache=past_key_values,  # [P1-3]
                    cache_offset=c_idx,  # [P1-3]
                    **kwargs,
                )
                if msgs is None:
                    return dl(x, cur_b, **kw)  # [P1-3]
                return dl(x, cur_b, msgs, **kw)  # [P1-3]
            return run_damped_rk_layer(fn, cur_h, rk_K=cfg.rk_K if use_rk else 1, rk_beta=cfg.rk_beta, cache=past_key_values, cache_idx=c_idx)

        c_idx = 0
        for lidx in range(start_l):
            nh = call_layer(lidx, h, blocks, None, c_idx)
            if self.layers[lidx].is_block_boundary:
                blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
            h = nh
            c_idx += 1

        pre_h, pre_b = h, blocks
        n_pre = pre_b.shape[-2]

        for lidx in range(start_l, end_l):
            nh = call_layer(lidx, h, blocks, None, c_idx)
            if self.layers[lidx].is_block_boundary:
                blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
            h = nh
            c_idx += 1

        for _loop_idx in range(1, cfg.num_passes):
            prev_b = blocks[..., n_pre + 1 :, :]
            prev_h = h
            h, blocks = pre_h, pre_b
            for off, lidx in enumerate(range(start_l, end_l)):
                msgs = self.message_input(prev_b, prev_h, off)
                nh = call_layer(lidx, h, blocks, msgs, c_idx)
                if self.layers[lidx].is_block_boundary:
                    blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
                h = nh
                c_idx += 1

        for lidx in range(end_l, n_layers):
            nh = call_layer(lidx, h, blocks, None, c_idx)
            if self.layers[lidx].is_block_boundary:
                blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
            h = nh
            c_idx += 1

        h = self.final_input(blocks, h)[0]  # [P1-4]
        h = self.norm(h).to(inputs_embeds.dtype)  # [P1-6] stock 5.24 casts back after norm
        # [P1-7] 5.24's CausalLM reads .router_logits/.cache off the inner output
        return MoeModelOutputWithPast(last_hidden_state=h, cache=past_key_values, router_logits=None)

    tm.forward = types.MethodType(custom_forward, tm)
