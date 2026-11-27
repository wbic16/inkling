"""Autonomous Multi-Generation Loop-Advantage GSPO + DAPO (rse_looper.gspo).

Implements:
  - GSPO sequence-level geometric-mean importance ratios (eliminating MoE token routing variance).
  - DAPO asymmetric Clip-Higher (eps_low=0.20, eps_high=0.28) + dynamic frontier filtering.
  - Dr. GRPO unbiased group advantages (A_i = r_i - mean(r_G)).
  - Loop-Advantage bonus (+delta_r_bonus) when 4-pass or 2-step damped solves a problem 3-pass missed.
  - Strict 3-mode hidden gate and full multi-GPU DDP runner (zero external teacher, 0 expert edits).
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import torch
import torch.nn.functional as F


def compute_loop_advantage_gspo_groups(
    rollouts_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    delta_r_bonus: float = 0.35,
    rep_penalty: float = 0.50,
    trunc_penalty: float = 0.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute dynamic-sampled, unbiased Loop-Advantage groups from 3-mode rollouts."""
    active_groups: List[Dict[str, Any]] = []
    n_total = len(rollouts_by_id)
    n_any_solved = 0
    n_all_solved = 0
    n_deep_rescue = 0
    total_rollouts = 0
    total_correct = 0

    for pid, recs in sorted(rollouts_by_id.items()):
        if not recs:
            continue
        total_rollouts += len(recs)
        n_ok = sum(1 for r in recs if r.get("correct") and r.get("completed") and not r.get("repetition"))
        total_correct += n_ok
        if n_ok > 0:
            n_any_solved += 1
        if n_ok == len(recs):
            n_all_solved += 1
        if n_ok == 0 or n_ok == len(recs):
            continue

        r3_recs = [r for r in recs if r.get("regime") == "R3"]
        r3_solved = any(r.get("correct") and r.get("completed") and not r.get("repetition") for r in r3_recs)

        rewards: List[float] = []
        group_had_rescue = False
        for r in recs:
            is_win = bool(r.get("correct") and r.get("completed") and not r.get("repetition"))
            rew = 1.0 if is_win else 0.0
            if r.get("repetition"):
                rew -= float(rep_penalty)
            elif not r.get("completed"):
                rew -= float(trunc_penalty)
            if is_win and (not r3_solved) and r.get("regime") in ("R4", "RK2"):
                rew += float(delta_r_bonus)
                group_had_rescue = True
            rewards.append(rew)

        if group_had_rescue:
            n_deep_rescue += 1

        mean_r = sum(rewards) / len(rewards)
        items = []
        for r, rew in zip(recs, rewards):
            items.append({
                "id": pid,
                "problem": r["problem"],
                "raw_full": r["raw_full"],
                "regime": r.get("regime", "R4"),
                "reward": round(rew, 4),
                "advantage": round(rew - mean_r, 4),
                "old_avg_logprob": float(r["old_avg_logprob"]),
                "correct": bool(r.get("correct")),
                "completed": bool(r.get("completed")),
                "gen_tokens": int(r.get("gen_tokens", 0)),
            })
        active_groups.append({"id": pid, "items": items})

    stats = {
        "n_problems": n_total,
        "best_of_k_solved": n_any_solved,
        "best_of_k_rate": round(n_any_solved / max(1, n_total) * 100.0, 2),
        "all_solved": n_all_solved,
        "active_frontier_groups": len(active_groups),
        "deep_loop_rescue_groups": n_deep_rescue,
        "pass1_avg_rate": round(total_correct / max(1, total_rollouts) * 100.0, 2),
    }
    return active_groups, stats


def gspo_dapo_sequence_loss(
    cur_avg_logprob: torch.Tensor,
    old_avg_logprob: torch.Tensor | float,
    advantage: torch.Tensor | float,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """GSPO sequence-level importance ratio loss with asymmetric Clip-Higher."""
    old_lp = torch.as_tensor(old_avg_logprob, dtype=torch.float32, device=cur_avg_logprob.device)
    adv = torch.as_tensor(advantage, dtype=torch.float32, device=cur_avg_logprob.device)
    log_ratio = torch.clamp(cur_avg_logprob.float() - old_lp, min=-4.0, max=4.0)
    seq_ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(seq_ratio, 1.0 - float(eps_low), 1.0 + float(eps_high))
    loss = -torch.minimum(seq_ratio * adv, clipped_ratio * adv).mean()
    info = {
        "seq_ratio": float(seq_ratio.detach().mean().item()),
        "clipped": bool((seq_ratio.detach() < (1.0 - eps_low)).any() or (seq_ratio.detach() > (1.0 + eps_high)).any()),
    }
    return loss, info


def accept_3regime_rsi_step(
    cur_gate: Mapping[str, Any],
    cand_gate: Mapping[str, Any],
    max_single_regime_drop: int = 2,
) -> bool:
    """Strict 3-mode (3-pass, 4-pass, 2-step damped) compounding acceptance rule."""
    regimes = ("R3", "R4", "RK2")
    if cand_gate["max_rep"] > max(3.12, cur_gate["max_rep"] + 1.6):
        return False
    if any(cand_gate[r]["correct"] < cur_gate[r]["correct"] - max_single_regime_drop for r in regimes):
        return False
    if cand_gate["total_correct"] > cur_gate["total_correct"]:
        return True
    if (
        cand_gate["total_correct"] == cur_gate["total_correct"]
        and cand_gate["total_completed"] >= cur_gate["total_completed"] + 4
    ):
        return True
    return False


def clip_generation_deltas(
    deltas: Mapping[str, torch.Tensor],
    base_weights: Mapping[str, torch.Tensor],
    max_rel: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Enforce per-generation relative Frobenius norm cap."""
    clipped: Dict[str, torch.Tensor] = {}
    for k, d in deltas.items():
        df = d.detach().float().cpu()
        if not torch.isfinite(df).all():
            raise ValueError(f"Non-finite delta on {k}")
        w = base_weights[k].detach().float().cpu()
        rel = df.norm().item() / (w.norm().item() + 1e-12)
        if rel > max_rel:
            df = df * (max_rel / rel)
        clipped[k] = df
    return clipped


def avg_sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Mean per-token log-probability over completion tokens."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    st = max(0, min(prompt_len - 1, shift_labels.shape[1] - 1))
    logp = F.log_softmax(shift_logits[:, st:, :], dim=-1)
    tok_lp = logp.gather(-1, shift_labels[:, st:].unsqueeze(-1)).squeeze(-1)
    return tok_lp.mean()


def _format_math_prompt(tok: Any, problem: str, effort: str = "high") -> str:
    msgs = [{"role": "user", "content": f"Solve the following problem and put your final answer in \\boxed{{}}.\n\n{problem}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _search_worker(local_idx: int, gpu_ids: List[int], model_path: str, rows: List[dict], out_dir: str, gen_idx: int, max_new_tokens: int) -> None:
    import json
    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    from rse_looper.grader import grade_math_output
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    from rse_looper.preflight import enable_fast_math

    gid = gpu_ids[local_idx]
    torch.cuda.set_device(gid)
    enable_fast_math()
    device = f"cuda:{gid}"

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = Inkling2LoopForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device}, experts_implementation="grouped_mm"
    ).eval()

    my_rows = [r for i, r in enumerate(rows) if i % len(gpu_ids) == local_idx]
    reg_specs = [
        ("R3", LoopConfig(num_passes=3, rk_K=1), 2),
        ("R4", LoopConfig(num_passes=4, rk_K=1), 2),
        ("RK2", LoopConfig(num_passes=3, rk_window=(10, 13), rk_K=2, rk_beta=0.3), 2),
    ]
    records = []
    for reg_name, cfg, n_samp in reg_specs:
        install_midstack_loop_forward(model, cfg)
        expanded = [(r, s_idx) for r in my_rows for s_idx in range(n_samp)]
        for start in range(0, len(expanded), 32):
            chunk = expanded[start : start + 32]
            texts = [_format_math_prompt(tok, r["problem"]) for r, _ in chunk]
            inputs = tok(texts, return_tensors="pt", padding=True).to(device)
            in_len = inputs["input_ids"].shape[1]
            torch.manual_seed(20000 * gen_idx + 100 * local_idx + start)
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.72, top_p=0.95,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            for j, (r, s_idx) in enumerate(chunk):
                g_ids = gen_out[j, in_len:]
                ep = (g_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
                if len(ep) > 0:
                    g_ids = g_ids[: ep[0].item() + 1]
                raw = tok.decode(g_ids, skip_special_tokens=False)
                g = grade_math_output(raw, r["answer"])
                pr = _format_math_prompt(tok, r["problem"])
                p_len = len(tok(pr)["input_ids"])
                full_ids = tok(pr + raw, return_tensors="pt", truncation=True, max_length=1280)["input_ids"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=full_ids, use_cache=False).logits
                    old_lp = float(avg_sequence_logprob(logits, full_ids, min(p_len, full_ids.shape[1] - 4)).item())
                records.append({
                    "id": r["id"], "problem": r["problem"], "answer": r["answer"], "regime": reg_name,
                    "sample_idx": s_idx, "gen_tokens": int(len(g_ids)), "correct": bool(g["correct"]),
                    "completed": bool(g["completed"]), "repetition": bool(g["repetition"]),
                    "old_avg_logprob": old_lp, "raw_full": raw,
                })
    with open(os.path.join(out_dir, f"search.shard{local_idx}.jsonl"), "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _ddp_update_worker(
    local_rank: int, world_size: int, gpu_ids: List[int], model_path: str,
    gspo_path: str, out_delta_path: str, steps: int, lr: float, seed: int,
) -> None:
    import json
    import torch.distributed as dist
    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    from rse_looper.checkpoint import is_editable_param
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    from rse_looper.preflight import enable_fast_math

    gid = gpu_ids[local_rank]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29710 + (seed % 500))
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(gid)
    enable_fast_math()
    device = f"cuda:{gid}"

    tok = AutoTokenizer.from_pretrained(model_path)
    model = Inkling2LoopForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device}, experts_implementation="grouped_mm"
    )
    for p in model.parameters():
        p.requires_grad_(False)
    trainable, orig_w = [], {}
    for name, p in model.named_parameters():
        if is_editable_param(name):
            p.requires_grad_(True)
            trainable.append((name, p))
            orig_w[name] = p.detach().float().cpu().clone()

    rows = [json.loads(l) for l in open(gspo_path) if l.strip()]
    prepped = []
    for r in rows:
        pr = _format_math_prompt(tok, r["problem"])
        p_len = len(tok(pr)["input_ids"])
        ids = tok(pr + r["raw_full"], return_tensors="pt", truncation=True, max_length=1280)["input_ids"]
        prepped.append((ids, min(p_len, ids.shape[1] - 4), r["regime"], float(r["advantage"]), float(r["old_avg_logprob"])))

    my_idxs = [i for i in range(len(prepped)) if i % world_size == local_rank] or [0]
    opt = torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = random.Random(seed + 41 * local_rank)

    cfgs = {
        "R3": LoopConfig(num_passes=3, rk_K=1),
        "R4": LoopConfig(num_passes=4, rk_K=1),
        "RK2": LoopConfig(num_passes=3, rk_window=(10, 13), rk_K=2, rk_beta=0.3),
    }

    model.train()
    for step in range(1, steps + 1):
        prog = max(0.0, (step - 3) / max(1, steps - 3))
        cur_lr = lr * (step / 3.0) if step <= 3 else lr * (0.20 + 0.80 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            ids, p_len, reg, adv, old_lp = prepped[rng.choice(my_idxs)]
            install_midstack_loop_forward(model, cfgs[reg])
            ids_d = ids.to(device)
            logits = model(input_ids=ids_d, use_cache=False).logits
            cur_lp = avg_sequence_logprob(logits, ids_d, p_len)
            loss_rl, _ = gspo_dapo_sequence_loss(cur_lp, old_lp, adv, eps_low=0.20, eps_high=0.28)
            (loss_rl * 0.5).backward()
            del logits, cur_lp, loss_rl

        for _, p in trainable:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()

    if local_rank == 0:
        raw_deltas = {name: p.detach().float().cpu() - orig_w[name] for name, p in trainable}
        clipped = clip_generation_deltas(raw_deltas, orig_w, max_rel=0.05)
        torch.save(clipped, out_delta_path)

    dist.barrier()
    dist.destroy_process_group()


def _gate_worker(local_idx: int, gpu_ids: List[int], model_path: str, delta_path: str | None, scale: float, gate_rows: List[dict], out_dir: str, tag: str) -> None:
    import json
    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    from rse_looper.checkpoint import apply_weight_deltas
    from rse_looper.grader import grade_math_output
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    from rse_looper.preflight import enable_fast_math

    regimes = [("R3", LoopConfig(num_passes=3, rk_K=1)), ("R4", LoopConfig(num_passes=4, rk_K=1)), ("RK2", LoopConfig(num_passes=3, rk_window=(10, 13), rk_K=2, rk_beta=0.3))]
    reg_name, cfg = regimes[local_idx % 3]
    sh_id = local_idx // 3
    n_sh = max(1, len(gpu_ids) // 3)
    gid = gpu_ids[local_idx]
    torch.cuda.set_device(gid)
    enable_fast_math()
    device = f"cuda:{gid}"

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = Inkling2LoopForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device}, experts_implementation="grouped_mm"
    ).eval()
    if delta_path:
        d = torch.load(delta_path, map_location="cpu", weights_only=True)
        if abs(scale - 1.0) > 1e-6:
            d = {k: v * float(scale) for k, v in d.items()}
        apply_weight_deltas(model, d)

    install_midstack_loop_forward(model, cfg)
    my_rows = [r for i, r in enumerate(gate_rows) if i % n_sh == sh_id]
    texts = [_format_math_prompt(tok, r["problem"]) for r in my_rows]
    inputs = tok(texts, return_tensors="pt", padding=True).to(device)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1536, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    res = []
    for j, r in enumerate(my_rows):
        g_ids = out[j, in_len:]
        ep = (g_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
        if len(ep) > 0:
            g_ids = g_ids[: ep[0].item() + 1]
        g = grade_math_output(tok.decode(g_ids, skip_special_tokens=False), r["answer"])
        res.append({"id": r["id"], "gen_tokens": int(len(g_ids)), **g})
    with open(os.path.join(out_dir, f"{tag}_{reg_name}.sh{sh_id}.jsonl"), "w") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(prog="python -m rse_looper.gspo", description="Run autonomous multi-generation Loop-Advantage GSPO + DAPO.")
    ap.add_argument("--init-model", default="./models/Inkling-2-Mini-FPSolve")
    ap.add_argument("--train-jsonl", default="./data/train_math.jsonl")
    ap.add_argument("--gate-jsonl", default="./data/gate_math.jsonl")
    ap.add_argument("--work-dir", default="./runs/gspo")
    ap.add_argument("--ckpt-prefix", default="./models/Inkblot-2-mini")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--max-gens", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--problems-per-gen", type=int, default=96)
    ap.add_argument("--gate-size", type=int, default=64)
    ap.add_argument("--steps-per-gen", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> dict:
    import json
    import torch.multiprocessing as mp
    from rse_looper.checkpoint import export_edited_checkpoint
    from rse_looper.preflight import enable_fast_math, preflight_gpu_check

    args = build_parser().parse_args(argv)
    enable_fast_math()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    preflight = preflight_gpu_check(require_causal_conv1d=False)
    summary = {
        "init_model": args.init_model, "work_dir": args.work_dir,
        "ckpt_prefix": args.ckpt_prefix, "gpus": gpu_ids, "max_gens": args.max_gens,
        "preflight": preflight,
    }
    if args.dry_run:
        return summary
    if not preflight["cuda_available"]:
        raise RuntimeError("Loop-Advantage GSPO requires CUDA GPUs.")

    os.makedirs(args.work_dir, exist_ok=True)
    train_pool = [json.loads(l) for l in open(args.train_jsonl) if l.strip()]
    gate_rows = [json.loads(l) for l in open(args.gate_jsonl) if l.strip()][: args.gate_size]

    def run_gate(model_p: str, delta_p: str | None, scale: float, gdir: str, tag: str) -> dict:
        os.makedirs(gdir, exist_ok=True)
        n_workers = min(len(gpu_ids), 6 if len(gpu_ids) >= 6 else 3)
        mp.spawn(_gate_worker, args=(gpu_ids[:n_workers], model_p, delta_p, scale, gate_rows, gdir, tag), nprocs=n_workers, join=True)
        out: dict = {}
        tot_ok, tot_comp, max_rep = 0, 0, 0.0
        n_sh = max(1, n_workers // 3)
        for reg in ("R3", "R4", "RK2"):
            rows = []
            for sh in range(n_sh):
                rows.extend(json.loads(l) for l in open(os.path.join(gdir, f"{tag}_{reg}.sh{sh}.jsonl")) if l.strip())
            n = len(rows)
            ok = sum(x["correct"] for x in rows)
            comp = sum(x["completed"] for x in rows)
            rep = round(sum(x["repetition"] for x in rows) / max(1, n) * 100.0, 2)
            out[reg] = {"n": n, "correct": ok, "completed": comp, "acc": round(ok / max(1, n) * 100.0, 2), "rep": rep}
            tot_ok += ok; tot_comp += comp; max_rep = max(max_rep, rep)
        out["total_correct"] = tot_ok
        out["total_completed"] = tot_comp
        out["mean_acc"] = round(tot_ok / max(1, 3 * len(gate_rows)) * 100.0, 2)
        out["max_rep"] = max_rep
        return out

    cur_model = args.init_model
    cur_gate = run_gate(cur_model, None, 1.0, os.path.join(args.work_dir, "gate_init"), "init")
    accepted, consec_reject = 0, 0
    for g in range(args.max_gens):
        gdir = os.path.join(args.work_dir, f"gen_{g}")
        os.makedirs(gdir, exist_ok=True)
        st = (g * args.problems_per_gen) % max(1, len(train_pool) - args.problems_per_gen)
        batch_rows = train_pool[st : st + args.problems_per_gen]
        mp.spawn(_search_worker, args=(gpu_ids, cur_model, batch_rows, gdir, g, 1536), nprocs=len(gpu_ids), join=True)
        by_id: dict = {}
        for i in range(len(gpu_ids)):
            for l in open(os.path.join(gdir, f"search.shard{i}.jsonl")):
                if l.strip():
                    rec = json.loads(l)
                    by_id.setdefault(rec["id"], []).append(rec)
        groups, _ = compute_loop_advantage_gspo_groups(by_id)
        gspo_path = os.path.join(gdir, "gspo_items.jsonl")
        with open(gspo_path, "w") as f:
            for grp in groups:
                for it in grp["items"]:
                    f.write(json.dumps(it) + "\n")
        delta_path = os.path.join(gdir, f"gspo_gen{g}_delta.pt")
        mp.spawn(_ddp_update_worker, args=(len(gpu_ids), gpu_ids, cur_model, gspo_path, delta_path, args.steps_per_gen, args.lr * (0.9 ** accepted), 3000 + g), nprocs=len(gpu_ids), join=True)

        best_win = None
        for sc in (1.0, 0.6):
            cand = run_gate(cur_model, delta_path, sc, os.path.join(gdir, "gate_eval"), f"g{g}_s{int(sc*10)}")
            if accept_3regime_rsi_step(cur_gate, cand):
                if best_win is None or cand["total_correct"] > best_win[1]["total_correct"]:
                    best_win = (sc, cand)
        if best_win is not None:
            sc, cand = best_win
            accepted += 1
            next_ckpt = f"{args.ckpt_prefix}-gen{accepted}"
            d_raw = torch.load(delta_path, map_location="cpu", weights_only=True)
            if abs(sc - 1.0) > 1e-6:
                d_raw = {k: v * float(sc) for k, v in d_raw.items()}
            export_edited_checkpoint(cur_model, next_ckpt, d_raw)
            cur_model, cur_gate, consec_reject = next_ckpt, cand, 0
        else:
            consec_reject += 1
            if consec_reject >= args.patience:
                break
    summary["accepted_gens"] = accepted
    summary["final_model"] = cur_model
    summary["final_gate"] = cur_gate
    return summary


if __name__ == "__main__":
    main()
