# FP-Solve Run — Progress / Checkpoint File

- Task: 26.7.10/9.6.12/6.6.8 (continuation) — Will's decision 2026-11-26 23:31 CST: "Download base only — run our own FP-Solve first"
- Base ONLY: thinkingmachines/Inkling-2-Mini. Do NOT download Ex0bit/Inkblot-2-mini.
- Started: EE 1025 (2026-11-26 ~23:32 CST)
- Credit: rse_looper / FP-Solve is Eric E. / Tensorbend.ai's work (eelbaz/Recursive-Self-Improving-Loopers). We run their code; shims recorded as patches.

## Stage status
- [x] 1. DOWNLOAD base HF checkpoint hub→NFS, sha256 verified — DONE EE ~1033 (00:58 CST): local assembly + NFS copy both sha256 65e60f51aa12… (HF LFS value). Route changed on evidence (see log).
- [x] 2. TORCH HOME on plain-view (CPU venv) + smoke test — DONE EE 1028 (env) / EE 1035 (smoke: grouped_mm load OK, 14/14 submodules, forward timed)
- [x] 3. CODE clone verified; shims recorded — rse_looper @765453d unmodified; P1 looping port written + equivalence-verified (see log)
- [x] 4. FP-SOLVE run (168 edited matrices expected) — DONE EE 1062 (03:17 CST): 168/168 matrices, mean rel 0.0793, max rel 0.2161, 4,593 s, deltas sha256 de77e757… staged on NFS
- [x] 5. TRANSPORT — DONE EE ~1070: TRANSPORT VERIFIED (168 edited F32, 536 unedited byte-identical, out sha256 853968a9…); eelbaz ckpt mapping stale vs published checkpoint → P2 TF-space reconstruction; orientation anchor caught + fixed a transpose bug (P3)
- [x] 6. VALIDATE — DONE EE 1092: GSM8K-32 base@3 22/32 (3 loops), base@4 1/32 (11 loops), fpsolve@3 26/32 (2), fpsolve@4 16/32 (2) — collapse cured
- [x] 7. REPORT — DONE EE 1092: RUN-REPORT.md written; artifacts in NFS fpsolve-staging/
- [ ] 5. TRANSPORT derived GGUF (experts byte-identical, controls F32)
- [ ] 6. VALIDATE llama-server loop.count 3/4 + GSM8K n>=32 A/B
- [ ] 7. REPORT (RUN-REPORT.md) + tasks.phext update

## Log
- EE 1025: run started. Report read; fleet patterns confirmed (fleet-ssh.sh, scripts scp'd to nodes).
- EE 1026–1028: Stage 2 DONE — CPU torch 2.15.0+cpu + transformers 5.24.1 venv on plain-view; Inkling2LoopForCausalLM present; rse_looper (commit 765453d) unpacked + imports OK on plain-view.
- EE 1028: GSM8K test set fetched to NFS rse-bench/ (1319 lines, sha256 3730d312…).
- EE 1029–1032: Download re-architected on evidence: HF throttles per connection (~1.7–5.6 MB/s/stream); hub→NFS write caps ~7.6 MB/s; plain-view→NFS writes at 113 MB/s and plain-view reaches HF via ranch proxy. Final route: parallel-range download on plain-view → local disk (~/inkling-hf-base), sha256-verify all files, then copy to NFS inkling-2-mini-hf/ and re-verify. Hub partial (3.9 GB) discarded; NFS part files removed.
- Driver (S1–S4 shims, PATCHES.md), thermal watchdog, and self-verifying GGUF transport written and staged on plain-view.
- EE 1035–1045: First solve attempt died at install_midstack_loop_forward: eelbaz's looping.py targets an inkling2_loop API present in NO released transformers (bisected wheels 5.0.0–5.24.1; inkling2_loop lands only in 5.23/5.24, without Inkling2LoopCache / loop_start_layer / their layer-call shape). Fix = P1 port (looping_port.py, PATCHES.md): mechanical API adaptations only, fpsolve.py byte-unmodified, injected via sys.modules. Verified: ported forward at 3 passes is BIT-IDENTICAL to stock 5.24 forward (logits maxdiff 0.0000, argmax 1.0000); at 4 passes it runs and diverges (cos 0.9698). Solve relaunched EE 1046.
