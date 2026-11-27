# rse_looper × the Fleet Inklings — Applicability Report

- **Task:** 26.7.10/9.6.12/6.6.8 — Will, 2026-11-26 21:35 CST (EE 1008): *"see if we can apply it to our inklings"*
- **Subject:** `rse_looper` from eelbaz/Recursive-Self-Improving-Loopers (commit 765453d, single-commit initial release, 2026-11-26). **rse_looper is Eric E. / Tensorbend.ai's work** — credited as such throughout; nothing here claims it.
- **Report written:** EE 1024 (2026-11-26 23:23 CST). All ranch work node-resident (aletheia-core, plain-view); no model files staged on the hatch VM.

## VERDICT

**Step 1 (FP-Solve): APPLICABLE WITH PORT.** The phenomenon FP-Solve treats — the 4-pass collapse — **reproduces in our own stack** (fleet llama.cpp + fleet GGUF, probe §4), and loop depth is a runtime knob in our llama.cpp, no patch needed. The port has three named pieces: (a) acquire the HF safetensors checkpoint (not on the fleet; 20.19 GB; needs Will's go — §1), (b) run the torch FP-Solve in a viable execution home (not the fleet as-is — §3), (c) transport the edited control tensors back into a derived GGUF (name/precision mapping worked out — §5).

**Step 2 (GSPO generations): APPLICABLE ONLY AFTER the Step-1 port exists, and heavier.** It needs the same torch execution home plus generation/grading loops ×4 generations; defer until Step 1 lands (§5).

**Cheapest decisive next step** (§5): download eelbaz's *published edited checkpoint* `Ex0bit/Inkblot-2-mini` (same 20.19 GB, same 1,244 keys), convert to GGUF, and A/B it at loop.count 3 vs 4 in fleet llama.cpp against our base. That tests the whole thesis in our stack for one download + one conversion, before we run any solve ourselves.

---

## Q1 — CHECKPOINT REALITY

Fleet NFS `/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/` (inventoried on aletheia-core, EE 1023):

| File | Size | sha256 (SHA256SUMS) | Format (header census) |
|---|---|---|---|
| Inkling-2-Mini-AD-Q4_K.gguf | 16,525,559,776 B | `2bfc233e10925297…` | **arch `inkling2_loop`, 24 blocks, 704 tensors** |
| Inkling-2-Mini-AD-IQ2_M.gguf | 8,665,091,680 B | `7321d77bd62fb1b4…` | arch `inkling2_loop`, 24 blocks, 704 tensors |
| Inkling-2-Mini-Q4_K_M.gguf | 17,382,541,440 B | `369168880d492c45…` | arch `inkling2` (old format), block_count metadata says 48 but only 24 tensor blocks present, 694 tensors |
| mmproj-Inkling-2-Mini-Q8_0.gguf | 345,911,008 B | `f0c269cdbffdeb2d…` | vision projector |

- **The fleet's live Inkling is the AD-Q4_K `inkling2_loop` file** — the brief's hash `2bfc233e…` is this file (not Q4_K_M). Its hash was re-verified live during the probe (full sha256sum on plain-view matched).
- **No HF safetensors checkpoint exists anywhere on the fleet share** (find across `/mnt/mirrorborn/shared` returns only the GGUFs). The fleet was built from GGUF downloads on 2026-11-21; the torch checkpoint rse_looper operates on was never here.
- **Control-tensor survival in the AD-Q4_K GGUF** (header census, `gguf-census-adq4k.txt`): the loop core is layers 6–17 (metadata `inkling2_loop.loop.start=6, .end=18, .count=3, .gate_rank=64, .varying=False`). Per core layer, eelbaz's FP-Solve targets exist with exact shape matches to the llama.cpp `inkling2_loop` class:
  - `loop_msg_query` — **F32** ✓
  - MoE router `ffn_gate_inp` + `exp_probs_b.bias` — **F32** ✓ (all 24 layers)
  - Short convs (`attn_k_conv`, `attn_v_conv`, `attn_post_conv`, `ffn_post_conv`) — **F32** ✓
  - Message gates `loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` (144 tensors) — **Q8_0** (near-lossless 8-bit; dequantize→F32 is exact-valued, edit→requantize loses ~0.4%, or store F32 in a derived GGUF)
  - Attention output `attn_output` — **Q8_0**
  - Experts (the bulk, untouched by rse_looper by design): `ffn_gate_up_exps` Q4_K, `ffn_down_exps` Q5_0. Overall census: F32 334, Q8_0 322, Q5_0 24, Q4_K 24 — a gentle imatrix quant, far less destructive than the old Q4_K_M (whose loop gates were Q4_K/Q5_0).
- **HF original acquisition cost:** `thinkingmachines/Inkling-2-Mini` = **20.19 GB total** (one 19.53 GB safetensors + 0.64 GB mtp + tokenizer files), verified via the HF API from aletheia-core. One download hub→NFS, the same pattern as the 2026-11-21 intake. **Not downloaded this pass** (per brief).

## Q2 — RUNTIME REALITY

- Fleet llama.cpp (hub checkout master e6d624b; fleet binaries in `~/llama-bin/`, NFS runtime tar) implements `llama_model_inkling2_loop`: loop count is read from GGUF metadata `inkling2_loop.loop.count` at load and the core layers are re-executed that many times per token, with per-pass cache slots and the cross-pass message gates (`control()`: (1−forget)·cur + gain·message) wired in the graph.
- **Loop depth is therefore a runtime knob, not a patch:** `--override-kv inkling2_loop.loop.count=int:4` works with the stock fleet `llama-server` (proven §4 — the 4th pass measurably executes: decode slowed 5.16 → 4.26 tok/s ≈ the 48→60 layer-slot ratio 0.80).
- eelbaz's `LoopConfig` wrapper is PyTorch-only (it patches the HF model's forward and snapshots KV/conv cache per pass); no equivalent is needed in our stack because llama.cpp's graph already implements the same recurrence.
- **Hygiene finding:** the old-format `Inkling-2-Mini-Q4_K_M.gguf` (arch `inkling2`, block_count=48 metadata vs 24 tensor blocks) **fails to load** on plain-view's fleet llama-server (`check_tensor_dims: tensor 'blk.24.attn_q.weight' not found`). It is a leftover of the first Nov-21 format; the AD files supersede it. Not fixed this pass — flagged.
- MLX port: last recorded state is Stage 3 PASS (2026-11-22, numeric agreement with llama-server) with Stage 4 dispatched and no completion on record; its files were not located on the hub this pass. The verdict does not depend on it — llama.cpp is the serving runtime and carries the knob.

## Q3 — HARDWARE REALITY

FP-Solve as shipped is torch code (`torch>=2.4`, safetensors, numpy; model class `Inkling2LoopForCausalLM` from the HF repo's remote code):

- **CUDA assumptions are soft, not hard:** `preflight.py` treats missing CUDA as a *warning*; `causal_conv1d` is optional (fallback kernels) unless `--require-causal-conv1d` is passed. TF32 flags are set if CUDA exists.
- **No torch is installed on the probed fleet nodes** (aletheia-core, plain-view: `ModuleNotFoundError`). Any fleet run starts with a torch install (CPU wheel ~0.8 GB; ROCm wheel ~3 GB via a proxy known to be flaky for uncached packages).
- **VRAM/RAM arithmetic:** the checkpoint is 19.53 GB on disk (experts stored quantized, control BF16). Calibration is tiny (48 prompts, batches of 8×256 tokens, two recording passes).
  - **lilly/neon RTX 4080 Laptop (12,282 MiB VRAM): CANNOT hold the 19.53 GB checkpoint.** Would need CPU offload (device_map) — slow, and it's Will's laptop: **not used this pass; named as Will's call.**
  - **Fleet 780M iGPUs:** unified DDR5 — plain-view has 44 GB RAM (43 available), so the checkpoint *fits in memory*; but torch-ROCm on gfx1103 (780M) + `grouped_mm` experts + the model's custom code is unproven, and `rocm-smi` isn't even on plain-view's PATH (the fleet's proven iGPU path is Vulkan via ggml, not ROCm torch). A ROCm smoke test is its own probe leg.
  - **CPU torch on plain-view (44 GB, 16 threads, cool node): the viable in-fleet home.** The 48-prompt calibration at ~3.4B active params is hours-scale, not the 5.8-minute CUDA figure — acceptable for a one-shot edit whose output is a checkpoint, not a serving path.
- **Viable execution homes, ranked:** (1) CPU torch on plain-view — in-fleet, bench, no new hardware risk; (2) ROCm torch on a 780M node — faster if the stack cooperates, needs a smoke test first; (3) lilly/neon with CPU offload — Will's word required, worst fit of the three.

## Q4 — PROBE (executed, plain-view, EE 1023)

Setup: fleet `llama-server` (`~/llama-bin/cpu/llama-server` on plain-view), model `Inkling-2-Mini-AD-Q4_K.gguf` from NFS, **sha256 re-verified live = `2bfc233e…`** (matches SHA256SUMS). Greedy decoding (temp 0), raw `/completion`, two prompts per leg. Logs: `~/rse-probe/probe3.log` on plain-view; full outputs in the probe JSONs beside it.

| Leg | Prompt | Result |
|---|---|---|
| base (loop.count=3) | "17 + 25? then count up by 7" | Coherent: counts 17, 24, 31, 38, … (skips stating 42, but no degeneration). 5.16 tok/s decode. |
| base (loop.count=3) | Janet's apples (GSM8K-style) | Productive step-by-step working (3×14=42 total, subtract 9…), cut off at 96 tokens mid-solution. No looping. |
| **loops4 (`--override-kv inkling2_loop.loop.count=int:4`)** | counting | Initially *better*: "17 + 25 = 42. 42 + 7 = 49. 49 + 7 = 56…" coherent. 4.26 tok/s (the extra pass executing). |
| **loops4** | Janet's apples | **COLLAPSE SIGNATURE: "1. 14 / 2. 42 / 3. 42 / 4. 42 / 5. 42 / 6. 42 …" — the correct intermediate value repeated forever in an enumerated repeat-loop**, exactly eelbaz's reported 4-pass phenotype (95.3% repeat loops on math in their PyTorch stack). |

Thermals: plain-view Tctl 41.2 °C → 56.8 °C peak across both legs — well inside discipline; nothing destabilized; server killed after each leg. n=2 prompts per leg: this is a *signature* reproduction, not a benchmark — but the signature is unambiguous and matches the paper's claim, in our runtime, on our checkpoint.

(Their unit/integrity tests were not run: no torch on any probed node, and the collapse reproduction is the stronger evidence. Running `pytest` in the hub clone of their repo is a cheap follow-up once torch exists anywhere.)

## Q5 — VERDICT & THE PORT, PRECISELY

**Step 1 FP-Solve: APPLICABLE WITH PORT.** The port:

1. **Acquire** (Will's go): HF `thinkingmachines/Inkling-2-Mini`, 20.19 GB, one download on aletheia-core → fleet NFS (est. 1 EE once approved).
2. **Solve** (execution home per §3): run eelbaz's FP-Solve unmodified against the HF checkpoint — CPU torch on plain-view first (their code, their prompts, their solve; only the device differs). Output: 327 edited BF16/FP32 tensors.
3. **Transport** (the real port work): map edited tensors HF→GGUF (`*_message_gate.down_proj → loop_*_ctrl_down`, `.forget_gate_proj → loop_*_ctrl_forget`, `.message_gate_proj → loop_*_ctrl_gain`, `mlp.gate → ffn_gate_inp`, `self_attn.o_proj → attn_output`; short convs 1:1) and write a **derived GGUF** from the AD-Q4_K file with the edited tensors stored as **F32** (llama.cpp accepts mixed types; upstream's own converter notes "the reference does not quantize the loop tensors"), experts and all other tensors byte-identical from the AD file. Set `loop.count` per experiment at runtime via `--override-kv` — no metadata surgery needed.
4. **Validate in-stack:** greedy battery (eelbaz's own grader prompts / a GSM8K slice) at loop.count 3 and 4, base vs edited — repeat-loop rate and accuracy, llama-server on a bench node, before any fleet rollout.

**Step 2 GSPO: DEFER.** `gspo.py` (493 lines) needs the same torch home plus sampled generation + deterministic grading + hidden-set gating ×4 generations — an order of magnitude more torch runtime than FP-Solve, on hardware where torch doesn't exist yet. It becomes plausible only after the Step-1 transport (step 3 above) exists, since its output needs the same ride back into GGUF. Revisit after Step 1 validates.

**Recommended immediate next step (needs Will's go — it's a 2×20.19 GB download):** pull **both** `thinkingmachines/Inkling-2-Mini` and `Ex0bit/Inkblot-2-mini` to the NFS share; convert Inkblot with llama.cpp's `conversion/inkling2.py` (it emits exactly the `inkling2_loop` GGUF format our AD files use); A/B base-vs-Inkblot at loop.count 3/4 in fleet llama-server on plain-view with a GSM8K slice. If eelbaz's edit holds up in *our* runtime, the thesis is proven in-stack and our own FP-Solve run (port steps 2–3) is justified; if it doesn't, we've spent a download and a conversion, not a research program. Sizing: downloads ~1 EE, conversion + A/B harness ~4–6 EE, FP-Solve proper (CPU) ~3–5 EE after that.

**Caveats on the record:** eelbaz's results are self-reported, single-commit, zero independent reproductions as of this writing — this report verifies only (a) the collapse is real in our stack and (b) the edit's targets exist in our checkpoint in editable precision. The cure is not yet verified anywhere outside Tensorbend's own artifacts; the Inkblot A/B above is the cheapest way to verify it in ours.

*Artifacts: this report; `gguf_census.py` + censuses (`gguf-census-q4km.txt`, `gguf-census-adq4k.txt`); probe scripts (`probe*_run.sh`) — all in `~/workspace/inklings-rse/`; node-side logs at `~/rse-probe/` on plain-view and the source clone at `~/rse-looper-src` on aletheia-core.*
