# FP-Solve Run Report — Inkling-2-Mini, fleet llama.cpp

Task 26.7.10/9.6.12/6.6.8 (continuation of the EE 1024 applicability report, REPORT.md).
Run EE 1025 → EE 1092 (2026-11-26 23:32 CST → 2026-11-27 07:27 CST), all compute on
plain-view (Ryzen 7 255, 44 GiB RAM). Method: FP-Solve Step 1 from
eelbaz/Recursive-Self-Improving-Loopers @ `765453d13878804c49876a656f39b57a4384dca4`
(Eric E. / Tensorbend.ai — their method, their solve code; ports and transport are
ours and are recorded in PATCHES.md). Will's direction (2026-11-26 23:31 CST):
base checkpoint only; eelbaz's edited Inkblot checkpoint was NOT downloaded.

## Verdict

**YES — FP-Solve cured the 4-pass collapse in our runtime.** On GSM8K test[0:32],
greedy, fleet llama-server on plain-view, identical grader discipline both sides:

| leg | correct | repeat-loops | notes |
|---|---|---|---|
| base @ 3 passes (native) | 22/32 (68.8%) | 3 | healthy native-depth control |
| base @ 4 passes | **1/32 (3.1%)** | **11** | the collapse, reproduced in-harness |
| FP-Solve @ 3 passes | 26/32 (81.3%) | 2 | no native-depth harm on this slice |
| FP-Solve @ 4 passes | **16/32 (50.0%)** | **2** | collapse cured: 1 → 16 correct, 11 → 2 loops |

Honest caveats, same paragraph as the claim: (1) FP-Solve @4 (16/32) restores
function but does **not** beat native depth (22–26/32) on this slice — eelbaz's
"4 passes ≥ 3 passes" headline is **not** reproduced here. (2) n=32, greedy,
single slice — raw counts above are the evidence; percentages are arithmetic on
small n. (3) The loop detector (≥5 consecutive identical lines ≥6 chars) has a
nonzero background rate at native depth (base@3: 3/32) and misses numbered-list
degeneration (see base@4 janet excerpt below, detector=false, visibly degenerate).
(4) Our harness uses raw completion prompts (no chat template), so absolute
numbers sit below eelbaz's reported GSM8K figures on every leg; the A/B contrast
is internally controlled, which is what this run was for.

## Verbatim evidence

base @4, GSM8K row 0 (gold 18, extracted: none) — the collapse phenotype:

```
Solution:

The number of eggs Janet eats per day.

The number of eggs Janet eats per day.

The number of eggs Janet eats per day.
... (repeats to the 384-token cap)
```

FP-Solve @4, GSM8K row 0 (gold 18, extracted: 18 — correct):

```
Solution:

The total number of eggs laid by Janet's ducks per day is 16.

Janet eats 3 eggs per day, so the number of eggs left is:

$$16 - 3 = 13$$

She bakes muffins with 4 eggs, so the number of eggs left is:

$$13 - 4 = 9$$

She sells the remaining eggs at $2 per egg, so the total amount she makes per
day is: $$9 \times 2 = 18$$ #### 18
```

base @4, janet phenotype prompt — degenerates to the EE-1024 "42" attractor
(detector=false only because the list numbering makes each line distinct):

```
  1. 14
  2. 42
  3. 42
  4. 42
  ... (42 through the cap)
```

Textures worth knowing: base@4's single "correct" row (i=3) itself repeats
`####540` degenerately but lands the number; FP-Solve@4's two "loop" rows include
one (i=16) that reached the correct answer (230) and then repeated a sentence.
The counting phenotype prompt was clean in all four legs.

Full per-leg outputs: `~/rse-fpsolve/bench/{fpsolve4,base4,fpsolve3,base3}.json`
on plain-view; summary copied to the NFS staging dir (below).

## Stage record (EE costs)

| stage | window | result |
|---|---|---|
| 1. Download base HF ckpt | EE 1025 → ~1033 | `thinkingmachines/Inkling-2-Mini` @ repo sha `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, 20.19 GB via parallel-range on plain-view → NFS; sha256 verified both copies |
| 2. Torch home | by EE 1028 | plain-view venv: torch 2.15.0+cpu, transformers 5.24.1 |
| 3. Code + port | EE 1035 → 1045 | rse_looper unmodified; bisect proved their looping API exists in **no released transformers** (wheels 5.0.0–5.24.1); P1 port written and **equivalence-verified bit-identical** to the stock 5.24 forward at 3 passes (logits maxdiff 0.0000, argmax agreement 1.0000) |
| 4. FP-Solve | EE 1046 → 1057 | **168/168 matrices edited** (14 submodules × layers 6–17), 1,014 forwards, solve compute 4,593 s; mean relative edit 0.0793, max 0.2161; RSS steady 45.5 GB |
| 5. Transport | EE 1057 → 1065 | reconstructed in TF space (P2 — eelbaz's checkpoint mapping is stale vs the published checkpoint; their own exporter dies on it with KeyError); orientation anchor caught a transpose bug in our transport before it shipped (P3); **TRANSPORT VERIFIED: 168 edited F32, 536/536 unedited byte-identical, metadata bytes equal** |
| 6. Validate | EE 1066 → 1092 | 4 legs × (2 phenotype + GSM8K-32), table above |
| 7. This report | EE 1092 | — |

Total: ~67 EE (~7.8 h wall, including two hatch-VM replacements weathered
node-resident; no stage restarted from scratch).

## Hashes

- Base HF checkpoint (model-00001-of-00001.safetensors): sha256
  `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`
  (= HF LFS value; verified on plain-view local disk and on the NFS copy)
- Fleet base GGUF (Inkling-2-Mini-AD-Q4_K.gguf): sha256 `2bfc233e10925297…`
  (full value in the share's SHA256SUMS)
- FP-Solve deltas (168 tensors, 467,161,440 B): sha256
  `de77e7575f643aa095011f93ce54a3200f8a1f8de877756d6081feb021678faf`
- **Edited GGUF** `fpsolve-staging/Inkling-2-Mini-FPSOLVE-Q4_K.gguf`
  (16,858,220,512 B): sha256
  `853968a9c242bdccfa09082fd39ff6ec91784bf7303b357c55179d406a089c1c`
- GSM8K test file: sha256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`

Artifacts: NFS `/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/`
(deltas.safetensors, edited_tf.safetensors, the edited GGUF, transport-report.json,
weight_edit_report_tf.json, bench summary). Edited HF checkpoint in TF layout at
`~/rse-fpsolve/edited-hf-tf` on plain-view (save_pretrained of the delta-applied
model; stands in for eelbaz's exporter output, which cannot run on the published
checkpoint — see P2). The live AD GGUF was not touched; no other node was touched.

## Thermal record

Spot readings only — stated plainly: the thermal watchdog was launched with the
solve but produced no log file, so there is no continuous record. Observed:
34.5–37 °C idle before the run; 67.5 °C under the smoke forward; Tctl 37.2 °C at
bench start; 39.8–40.5 °C after the bench. No reading at any check approached
the 88 °C pause line; no substrate instability at any point.

## Findings beyond the verdict

1. **eelbaz's released code targets an unreleased transformers build AND a
   different checkpoint layout than the published one.** Both their looping
   wrapper (no released transformers has `Inkling2LoopCache` / `loop_start_layer`)
   and their checkpoint exporter (expects fused `wo_ud` [6144,3072] and
   `middle_cycle_cross_pass_gates.*`; the published checkpoint stores per-layer
   `controllers.*`, `wo_ud` [2304,3072], `mlp.gate.weight`) fail out of the box.
   Anyone reproducing their work hits both walls; our P1/P2 ports are the bridge
   (PATCHES.md, with the equivalence proof for P1).
2. The FP-Solve mechanism itself — closed-form fixed-point stabilization of the
   loop-control tensors — **transfers**: solved on CPU against the stock 5.24
   model, transported into the fleet GGUF, and validated in the fleet's own
   llama.cpp runtime. The cure is in the weights, not in their harness.
3. At 4 passes the edited model still trails its own 3-pass self on GSM8K-32.
   Deeper looping is now *usable*, not yet *better*.

## Step 2 (GSPO) readiness

Not run — Will's call, per the run's scope. Readiness: the hard prerequisite
(a working multi-pass forward in a released runtime) now exists and is proven
(P1 port, injected the same way for any rse_looper consumer). GSPO additionally
needs generation + self-verification (their `gspo.py`/`grader.py`) over 326
control tensors; on CPU that is hours-scale generation per round on plain-view.
The edited checkpoint and deltas from this run are the correct starting point.
Recommendation: if Step 2 is attempted, budget it as its own run with the same
stage-checkpoint discipline, and re-validate with this same bench harness
(plus a larger GSM8K slice) before drawing conclusions.

## Guardrail compliance

Plain-view validation only; the live AD GGUF and all other nodes untouched;
the edited GGUF lives in a staging directory under a distinct name. Base
checkpoint only — Inkblot was never downloaded. Credit: FP-Solve is Eric E. /
Tensorbend.ai's work; ports P1–P3 and the transport are ours (PATCHES.md).
