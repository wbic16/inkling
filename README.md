# inkling — local-inference tooling for the Inklings

The canonical home for the tooling built around the fleet's Inkling-2-Mini
models: the applicability study, the FP-Solve run that cured the 4-pass
collapse in our own runtime, the ports that made upstream code run against
released software, the GGUF transport, and the bench harness that graded it.

Will's direction, 2026-11-27: *"keep all of this tooling for local inference
in wbic16/inkling and fetch the tooling as part of Exocortical setup."*
Exocortical setup (`wbic16/exocortical` → `setup.sh`, Phase 8) clones this
repo to `/source/inkling` on every provisioned node, clone-or-update.

Task of record: 26.7.10/9.6.12/6.6.8. Work performed 2026-11-26/27
(EE 1008–1092), all compute on the ranch (plain-view), nothing staged on the
hatch VM.

## Verdict

**FP-Solve cured the 4-pass collapse in our runtime.** GSM8K test[0:32],
greedy, fleet llama-server on plain-view, one grader for all legs:

| leg | correct | repeat-loops |
|---|---|---|
| base @ 3 passes (native) | 22/32 | 3 |
| base @ 4 passes | **1/32** | **11** |
| FP-Solve @ 3 passes | 26/32 | 2 |
| FP-Solve @ 4 passes | **16/32** | **2** |

Honest caveats (RUN-REPORT.md states them in the same breath): FP-Solve @4
restores function but does **not** beat native depth on this slice — the
upstream "4 passes ≥ 3 passes" headline is not reproduced; n=32 raw counts
are the evidence; the loop detector has a nonzero native-depth background
rate and misses numbered-list degeneration.

## Documents

- `REPORT.md` — applicability study (EE 1024): Step 1 FP-Solve applicable
  with port; the 4-pass collapse reproduced in fleet llama.cpp first.
- `RUN-REPORT.md` — the run (EE 1025–1092): verdict, verbatim evidence,
  stage record, hashes, findings.
- `PATCHES.md` — every shim/port around the upstream code, with reasons
  (S1–S4 driver shims, P1 looping port, P2 TF-space reconstruction,
  P3 transport orientation fix, T1 GGUF transport).
- `PROGRESS.md` — the run's stage-checkpoint log.

## Layout

- `ports/` — **P1** `looping_port.py`: eelbaz's looping forward ported to
  released transformers 5.24.1, **proven bit-identical to the stock 5.24
  forward at 3 passes** (logits maxdiff 0.0000, argmax agreement 1.0000;
  probes: `probe_looping.py`, `probe_looping2.py`). **P2**
  `reconstruct_tf.py`: rebuilds the edited checkpoint in TF space via
  upstream's own `apply_weight_deltas`, because their exporter targets a
  checkpoint layout the published checkpoint does not use.
- `solve/` — `driver_fpsolve.py`: the FP-Solve driver (process split,
  CPU device, observability) around the **byte-unmodified** upstream
  `fpsolve.py`.
- `transport/` — `transport_gguf.py`: rebuilds the fleet GGUF with the 168
  edited tensors as F32 and every other tensor byte-identical (536/536
  verified). `verify_orientation.py`: the orientation anchor that caught a
  transpose bug in the transport before it shipped (P3).
- `bench/` — `bench.py` + `bench_excerpts.py`: the GSM8K harness/grader
  used for all four legs above.
- `census/` — GGUF header census tooling and output (`gguf_census.py`,
  census text for the AD-Q4_K and Q4_K_M files, checkpoint/delta key
  inspectors).
- `probes/` — the applicability probes (loop-count override checks against
  fleet llama-server).
- `scripts/` — the operational run scripts from the run itself
  (download, torch setup, solve, transport, bench, watchdog), kept as the
  working record of how each stage was actually executed.
- `third-party/` — third-party code, clearly marked; see
  `third-party/PROVENANCE.md`. Nothing in there is ours.

## Large artifacts — NOT in git

The repo carries hashes, sizes, and locations; the bytes live on fleet NFS
and regenerate through the pipeline below.

| artifact | size | sha256 | location |
|---|---|---|---|
| Base HF checkpoint `thinkingmachines/Inkling-2-Mini` (repo sha `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, `model-00001-of-00001.safetensors`) | 20.19 GB | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` (= HF LFS) | NFS `inkling-2-mini-hf/` on the fleet share; working copy `~/inkling-hf-base` on plain-view |
| FP-Solve deltas (168 tensors) | 467,161,440 B | `de77e7575f643aa095011f93ce54a3200f8a1f8de877756d6081feb021678faf` | `/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/deltas.safetensors` |
| Edited GGUF `Inkling-2-Mini-FPSOLVE-Q4_K.gguf` | 16,858,220,512 B | `853968a9c242bdccfa09082fd39ff6ec91784bf7303b357c55179d406a089c1c` | `/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/` |
| Edited HF checkpoint (TF layout, `save_pretrained` of the delta-applied model) | ~20 GB | — (see `weight_edit_report_tf.json` in fpsolve-staging) | `~/rse-fpsolve/edited-hf-tf` on plain-view |
| GSM8K test file | 1319 lines | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` | NFS `rse-bench/` on the fleet share |
| Fleet base GGUF `Inkling-2-Mini-AD-Q4_K.gguf` (untouched by the run) | 16,525,559,776 B | `2bfc233e10925297…` (full value in the share's SHA256SUMS) | `/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/` |

## Reproduce

1. **Fetch the base checkpoint** — `thinkingmachines/Inkling-2-Mini` from
   Hugging Face (20.19 GB). The run's route (HF throttles per connection;
   plain-view reaches HF via the ranch proxy and writes NFS at 113 MB/s):
   parallel-range download on plain-view → sha256-verify → copy to NFS →
   re-verify. See `scripts/run-pv-download.sh`, `scripts/run-download.sh`.
   Base only — eelbaz's edited `Ex0bit/Inkblot-2-mini` was deliberately
   NOT downloaded (Will, 2026-11-26).
2. **Torch home** — CPU venv on a ≥44 GB RAM node: torch 2.15.0+cpu,
   transformers 5.24.1 (`scripts/run-torch-setup.sh`).
3. **Solve** — `solve/driver_fpsolve.py` with the P1 port registered
   (`ports/looping_port.py`); upstream `fpsolve.py` runs byte-unmodified.
   Expect: 168/168 matrices edited, ~4,593 s compute on a Ryzen 7 255,
   mean relative edit 0.0793 / max 0.2161, deltas sha256 above.
4. **Transport** — `ports/reconstruct_tf.py` (P2) then
   `transport/transport_gguf.py`; verify with
   `transport/verify_orientation.py` and the transport's own round-trip
   check (168 edited F32, 536/536 unedited byte-identical, metadata bytes
   equal).
5. **Bench** — `bench/bench.py` against fleet llama-server with
   `--override-kv inkling2_loop.loop.count=int:3` and `int:4`, GSM8K
   test[0:32], greedy. Expected: the table above.

## Findings a reproducer will hit

1. **Upstream's released code targets an unreleased transformers build and
   a different checkpoint layout than the published one.** Their looping
   wrapper exists in no released transformers (bisected 5.0.0–5.24.1) and
   their exporter dies on the published checkpoint with `KeyError`. P1/P2
   are the bridge; both findings are documented in RUN-REPORT.md.
2. **A transport self-verify can share the bug it checks for.** The P3
   transpose bug passed the transport's own round-trip verify; only the
   independent orientation anchor (cos = 1.000000 exact) caught it. Keep
   the anchor.

## Step 2 (GSPO) readiness

Not run — Will's call. The hard prerequisite (a working multi-pass forward
in a released runtime) exists and is proven: the P1 port injects the same
way for any rse_looper consumer. GSPO needs generation + self-verification
over 326 control tensors — hours-scale CPU generation per round on
plain-view. This run's deltas + edited checkpoint are the correct starting
point; judge any GSPO attempt with this same bench harness plus a larger
GSM8K slice.

## Provenance

FP-Solve / rse_looper is **Eric E. / Tensorbend.ai's work**
(eelbaz/Recursive-Self-Improving-Loopers @
`765453d13878804c49876a656f39b57a4384dca4`), run byte-unmodified; the copy
in `third-party/` is marked as such. The ports (P1–P3), the driver shims,
the transport, the census, the probes, and the bench harness are ours.
`third-party/llama.cpp-inkling2.py` is the llama.cpp project's gguf
conversion model definition, kept from the recon. This repo's own code is
under the repo LICENSE (MIT, Will Bickford); third-party code remains
under its own terms.
