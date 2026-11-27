# PATCHES.md — shims applied around eelbaz/rse_looper (commit 765453d)

rse_looper itself is **unmodified**. FP-Solve (Eric E. / Tensorbend.ai) is theirs;
the items below are orchestration around their code, each with its reason. The solve
function `run_fpsolve_on_model` and the exporter `export_edited_checkpoint` run as written.

- **S1 — Process split (driver_fpsolve.py).** eelbaz `main()` loads the model, solves,
  then exports while the model is still resident; the exporter additionally loads every
  checkpoint tensor into RAM. On plain-view (44 GB) that risks OOM. The driver runs the
  solve in one process (saving `deltas.safetensors` + `fpsolve_run.json` exactly as
  `main()` computes them) and the export in a second process. No numeric effect.
- **S2 — device=cpu.** Their `--device` argument exists; default is `cuda:0`. The fleet
  has no CUDA device that can hold the checkpoint, so the solve runs on CPU torch
  (plain-view), as the applicability report prescribed.
- **S3 — experts_implementation fallback.** Driver tries their `"grouped_mm"` first; only
  if load/forward raises does it retry `"eager"`. Which one was used is recorded in
  `smoke.json` / `fpsolve_run.json`.
- **S4 — Observability hook.** A forward-call counter + RSS printout so a multi-hour CPU
  solve is monitorable. Registers one extra forward hook that only counts; no numeric effect.
- **T1 — GGUF transport (transport_gguf.py).** Not eelbaz code: ours. Rebuilds the fleet
  GGUF with the 168 edited tensors stored F32, all other tensor bytes copied verbatim
  and hash-proven identical (536/536 unedited byte-identical, metadata bytes equal).
- **P2 — checkpoint-space mapping bypassed; reconstruction in TF space.** eelbaz's
  `checkpoint.transformers_name_to_ckpt_deltas` + `export_edited_checkpoint` target a
  checkpoint layout the **published** `thinkingmachines/Inkling-2-Mini` checkpoint does
  not use: they expect fused `attn.wo_ud.weight` [6144,3072] and
  `middle_cycle_cross_pass_gates.{off}.{kind}_gate.{down,up}_proj.weight`; the published
  file stores per-layer `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight`,
  `attn.wo_ud.weight` [2304,3072], and the router as `mlp.gate.weight`. Their exporter
  dies on the published checkpoint (`KeyError: model.llm.middle_cycle_cross_pass_gates.4...`).
  Our transport therefore reconstructs edited tensors in TF space instead:
  `reconstruct_tf.py` loads the base model and applies the deltas with eelbaz's own
  `apply_weight_deltas` (TF space, unmodified), extracts the 168 edited TF tensors,
  and the transport maps TF→GGUF directly (down/forget/gain separate — no up_proj
  fusion to undo; o_proj.linear and mlp.gate direct). An edited HF checkpoint in TF
  layout (`edited-hf-tf` on plain-view, via save_pretrained) + `weight_edit_report_tf.json`
  stand in for their exporter output as evidence.
- **P3 — GGUF data-order fix in the transport (ours).** The transport originally wrote
  edited tensors as `W.T.tobytes()`. An orientation anchor (verify_orientation.py —
  dequantize source-GGUF tensors, compare flat order against the checkpoint tensors
  they were converted from) proved GGUF flat data order equals torch row-major order
  (cos = 1.000000 exact on the F32 router; ≈0.99998 on Q8_0 anchors; transposed ≈ 0).
  The write and the round-trip verify now both use flat torch order. Without the
  anchor, the self-verify would have passed a file whose edited tensors llama.cpp
  reads transposed.
- **P1 — looping.py port to transformers 5.24.1 (looping_port.py).**eelbaz's
  `install_midstack_loop_forward` targets an inkling2_loop API that exists in **no
  released transformers** (bisected 5.0.0–5.24.1 wheels: no `Inkling2LoopCache`, no
  `loop_start_layer`; inkling2_loop itself only lands in 5.23/5.24). `fpsolve.py` —
  the solve math, hooks, and closed-form solver — is byte-unmodified; the driver
  registers `looping_port` in `sys.modules` under `rse_looper.looping` so fpsolve's
  lazy import binds the port. The port changes only mechanical API surface inside
  their forward: [P1-1] `Inkling2LoopCache`→`DynamicCache`; [P1-2]
  `config.loop_start_layer/loop_num_layers`→`loop_layer_index_start/end`; [P1-3]
  layer kwargs `past_key_values=`→`cache=`, `cache_layer_idx=`→`cache_offset=`,
  `cache_position` dropped, `messages` passed positionally only when not None
  (non-loop layers take `attention_mask` as 3rd positional in 5.24); [P1-4]
  `final_input(...)[0]` (5.24 returns a tuple, as its stock forward indexes);
  [P1-5] the 5.24 CausalLM's `cache=` kwarg folded into `past_key_values`;
  [P1-6] `.to(inputs_embeds.dtype)` after the final norm (mirrors stock forward);
  [P1-7] returns `MoeModelOutputWithPast(..., router_logits=None)` (5.24's CausalLM
  reads `.router_logits`). **Verification:** at num_passes=3 the ported forward is
  bit-identical to the stock 5.24 forward (logits maxdiff 0.0000, argmax agreement
  1.0000, probe_looping2.py); at num_passes=4 it runs and diverges (cos 0.9698),
  the unstable regime FP-Solve treats. On the FP-Solve path (use_cache=False) the
  cache is never constructed and cache_offset is inert.
