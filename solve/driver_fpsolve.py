#!/usr/bin/env python3
"""FP-Solve driver — orchestration ONLY around eelbaz's unmodified rse_looper code.

Credit: FP-Solve is Eric E. / Tensorbend.ai's work (eelbaz/Recursive-Self-Improving-Loopers @ 765453d).
This driver calls rse_looper.fpsolve.run_fpsolve_on_model and
rse_looper.checkpoint.export_edited_checkpoint UNMODIFIED. Shims (recorded in PATCHES.md):
  S1 process split (solve -> save deltas -> exit; export in a second process) for the 44 GB RAM node.
  S2 device=cpu (their --device exists; default is cuda:0).
  S3 experts_implementation: try their "grouped_mm"; fall back to "eager" only if load/forward raises.
  S4 observability: a forward counter hook + RSS prints. No numeric effect.
"""
import argparse, json, os, resource, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", choices=["smoke", "solve", "export"], required=True)
    ap.add_argument("--out-model", default=None)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    log = open(os.path.join(args.work, f"driver-{args.mode}.log"), "a")

    def say(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
        print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)

    torch.set_grad_enabled(False)
    torch.set_num_threads(16)
    from rse_looper.preflight import enable_fast_math
    enable_fast_math()
    say(f"mode={args.mode} torch={torch.__version__} threads={torch.get_num_threads()}")

    if args.mode == "export":
        from safetensors.torch import load_file
        from rse_looper.checkpoint import export_edited_checkpoint
        deltas = load_file(os.path.join(args.work, "deltas.safetensors"))
        say(f"export: {len(deltas)} deltas loaded, rss={rss_gb():.1f}GB")
        t0 = time.time()
        rep = export_edited_checkpoint(args.base, args.out_model, deltas)
        say(f"export done in {time.time()-t0:.0f}s, {len(rep)} ckpt keys edited, rss={rss_gb():.1f}GB")
        json.dump(rep, open(os.path.join(args.work, "export_report.json"), "w"), indent=2)
        return

    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = None
    used_impl = None
    for impl in ("grouped_mm", "eager"):
        try:
            t0 = time.time()
            model = Inkling2LoopForCausalLM.from_pretrained(
                args.base, dtype=torch.bfloat16, device_map={"": "cpu"},
                experts_implementation=impl,
            ).eval()
            used_impl = impl
            say(f"model loaded impl={impl} in {time.time()-t0:.0f}s rss={rss_gb():.1f}GB")
            break
        except Exception as e:
            say(f"load impl={impl} FAILED: {type(e).__name__}: {str(e)[:300]}")
            model = None
    if model is None:
        say("BLOCKER: model load failed under both experts_implementation options")
        sys.exit(2)

    # structural check: eelbaz's solve addresses model.model.layers[i].get_submodule(name)
    from rse_looper.fpsolve import DEFAULT_FPSOLVE_SUBMODULES
    try:
        layers = model.model.layers
        say(f"layers: {len(layers)}")
        for sn in DEFAULT_FPSOLVE_SUBMODULES:
            layers[6].get_submodule(sn)
        say("all 14 submodules present on layer 6")
    except Exception as e:
        say(f"BLOCKER: model structure mismatch: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(2)

    prompts = [f"Explain step by step how to solve equation {k}x + {k+3} = {4*k+7}." for k in range(1, 49)]
    cal_batches = [tok(prompts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cpu") for i in range(0, len(prompts), 8)]
    say(f"cal_batches: {len(cal_batches)} x {cal_batches[0]['input_ids'].shape}")

    # S4: forward counter (observability only)
    counter = {"n": 0, "t0": time.time()}
    prog = os.path.join(args.work, "solve-progress.txt")

    def count_hook(_m, _i, _o):
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            with open(prog, "w") as f:
                f.write(f"forwards={counter['n']} elapsed={time.time()-counter['t0']:.0f}s rss={rss_gb():.1f}GB\n")

    h = model.register_forward_hook(count_hook)
    t0 = time.time()
    _ = model(input_ids=cal_batches[0]["input_ids"], attention_mask=cal_batches[0]["attention_mask"], use_cache=False)
    dt = time.time() - t0
    h.remove()
    total_fw = 6 + 14 * 12 * 6
    say(f"single forward: {dt:.1f}s -> estimated solve forwards={total_fw}, est total={dt*total_fw/60:.0f} min, rss={rss_gb():.1f}GB")
    json.dump({"forward_sec": dt, "impl": used_impl, "rss_gb": rss_gb()},
              open(os.path.join(args.work, "smoke.json"), "w"), indent=2)
    if args.mode == "smoke":
        say("SMOKE OK")
        return

    # orig snapshot exactly as eelbaz main() does
    orig_cpu = {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if n.startswith("model.layers.") and ("message_gate." in n or "self_attn.o_proj.linear" in n or "mlp.gate.weight" in n)
    }
    say(f"orig snapshot: {len(orig_cpu)} params")
    # P1: substitute the ported looping module (transformers 5.24 API) for
    # rse_looper.looping before fpsolve's lazy import binds it. fpsolve.py untouched.
    import looping_port
    sys.modules["rse_looper.looping"] = looping_port
    counter["n"] = 0
    counter["t0"] = time.time()
    h = model.register_forward_hook(count_hook)
    from rse_looper.fpsolve import run_fpsolve_on_model
    t0 = time.time()
    info = run_fpsolve_on_model(model, cal_batches, lam_keep=4.0, ridge_frac=0.06, max_rel=0.25)
    h.remove()
    info["elapsed_sec"] = round(time.time() - t0, 1)
    info["impl"] = used_impl
    say(f"solve done: {info}")

    deltas = {}
    for n, p in model.named_parameters():
        if n in orig_cpu:
            d = p.detach().float().cpu() - orig_cpu[n]
            if d.abs().max().item() > 1e-7:
                deltas[n] = d
    say(f"deltas: {len(deltas)} tensors")
    from safetensors.torch import save_file
    save_file(deltas, os.path.join(args.work, "deltas.safetensors"), metadata={"format": "pt"})
    json.dump(info, open(os.path.join(args.work, "fpsolve_run.json"), "w"), indent=2)
    say(f"SOLVE DONE rss={rss_gb():.1f}GB")
    with open(os.path.join(args.work, "SOLVE_DONE"), "w") as f:
        f.write(json.dumps(info))


if __name__ == "__main__":
    main()
