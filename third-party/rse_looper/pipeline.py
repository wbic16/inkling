"""Production Autonomous Pipeline CLI for Recursive Self-Improving Loopers.

Runs the verified two-stage recipe end-to-end with preflight checks, strict gating,
and identical-parameter checkpoint export:
  - Stage 1 (fpsolve): 100% backprop-free, zero-label Cascaded Closed-Form FP-Solve
    across L06-L17 (168 BF16/FP32 matrices, ~5.8 min on 1 GPU, 0 expert weights touched).
  - Stage 2 (gspo): Multi-generation autonomous Loop-Advantage GSPO + DAPO with 3-regime
    hidden gating (3-pass, 4-pass, damped RK 2-step) and automatic top-generation weight merge.

Usage:
  rse-looper --stage both --base-model /path/to/Inkling-2-Mini --work-dir ./runs/prod
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence
import torch

from rse_looper.preflight import enable_fast_math, preflight_gpu_check


@dataclass(frozen=True)
class PipelineConfig:
    stage: str = "both"  # "fpsolve" | "gspo" | "both"
    base_model: str = "thinkingmachines/Inkling-2-Mini"
    work_dir: str = "./runs/rse_looper"
    out_model: str = "./models/Inkblot-2-mini"
    gpus: str = "0,1,2,3,4,5"
    # Stage 1 (FP-Solve, 0 backprop, 0 labels)
    fpsolve_lam_keep: float = 4.0
    fpsolve_ridge_frac: float = 0.06
    fpsolve_max_rel: float = 0.25
    fpsolve_num_prompts: int = 48
    # Stage 2 (Loop-Advantage GSPO + DAPO, 0 external teacher)
    gspo_max_gens: int = 10
    gspo_patience: int = 3
    gspo_problems_per_gen: int = 96
    gspo_steps_per_gen: int = 24
    gspo_lr: float = 2.0e-5
    require_causal_conv1d: bool = False
    resume: bool = True

    def validate(self) -> None:
        if self.stage not in ("fpsolve", "gspo", "both"):
            raise ValueError(f"Unsupported stage: {self.stage!r}")
        if self.fpsolve_lam_keep < 0.0 or not (0.0 < self.fpsolve_max_rel <= 0.5):
            raise ValueError("Invalid FP-Solve regularization or relative norm cap")
        if self.gspo_max_gens < 1 or self.gspo_patience < 1:
            raise ValueError("gspo_max_gens and gspo_patience must be >= 1")
        gpu_list = [int(x.strip()) for x in self.gpus.split(",") if x.strip()]
        if not gpu_list:
            raise ValueError("At least one GPU id must be provided")


def merge_generation_deltas(
    delta_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> Dict[str, torch.Tensor]:
    """Combine multiple accepted generation deltas into one multi-generation merged delta."""
    if not delta_dicts:
        raise ValueError("delta_dicts must be non-empty")
    if weights is None:
        w_norm = [1.0 / len(delta_dicts)] * len(delta_dicts)
    else:
        if len(weights) != len(delta_dicts):
            raise ValueError("weights and delta_dicts must match in length")
        w_sum = sum(float(w) for w in weights)
        if w_sum <= 0.0:
            raise ValueError("weights must sum to a positive value")
        w_norm = [float(w) / w_sum for w in weights]

    keys = sorted({k for d in delta_dicts for k in d.keys()})
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc: torch.Tensor | None = None
        for d, w in zip(delta_dicts, w_norm):
            if k in d:
                term = d[k].detach().float() * w
                acc = term if acc is None else acc + term
        if acc is not None:
            merged[k] = acc
    return merged


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rse-looper",
        description="Run autonomous FP-Solve + Loop-Advantage GSPO self-improvement pipeline.",
    )
    ap.add_argument("--stage", choices=("fpsolve", "gspo", "both"), default="both")
    ap.add_argument("--base-model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--work-dir", default="./runs/rse_looper")
    ap.add_argument("--out-model", default="./models/Inkblot-2-mini")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--fpsolve-lam-keep", type=float, default=4.0)
    ap.add_argument("--fpsolve-ridge-frac", type=float, default=0.06)
    ap.add_argument("--fpsolve-max-rel", type=float, default=0.25)
    ap.add_argument("--gspo-max-gens", type=int, default=10)
    ap.add_argument("--gspo-patience", type=int, default=3)
    ap.add_argument("--gspo-problems-per-gen", type=int, default=96)
    ap.add_argument("--gspo-steps-per-gen", type=int, default=24)
    ap.add_argument("--gspo-lr", type=float, default=2.0e-5)
    ap.add_argument("--require-causal-conv1d", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Validate config and run preflight only.")
    return ap


def parse_config(argv: Sequence[str] | None = None) -> tuple[PipelineConfig, bool]:
    ns = build_parser().parse_args(argv)
    cfg = PipelineConfig(
        stage=ns.stage,
        base_model=ns.base_model,
        work_dir=ns.work_dir,
        out_model=ns.out_model,
        gpus=ns.gpus,
        fpsolve_lam_keep=ns.fpsolve_lam_keep,
        fpsolve_ridge_frac=ns.fpsolve_ridge_frac,
        fpsolve_max_rel=ns.fpsolve_max_rel,
        gspo_max_gens=ns.gspo_max_gens,
        gspo_patience=ns.gspo_patience,
        gspo_problems_per_gen=ns.gspo_problems_per_gen,
        gspo_steps_per_gen=ns.gspo_steps_per_gen,
        gspo_lr=ns.gspo_lr,
        require_causal_conv1d=ns.require_causal_conv1d,
        resume=not ns.no_resume,
    )
    cfg.validate()
    return cfg, bool(ns.dry_run)


def main(argv: Sequence[str] | None = None) -> dict:
    cfg, dry_run = parse_config(argv)
    enable_fast_math()
    preflight = preflight_gpu_check(require_causal_conv1d=cfg.require_causal_conv1d)
    plan = {
        "stage": cfg.stage,
        "base_model": cfg.base_model,
        "out_model": cfg.out_model,
        "gpus": [int(x.strip()) for x in cfg.gpus.split(",") if x.strip()],
        "preflight": preflight,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    import os
    import subprocess
    import sys

    os.makedirs(cfg.work_dir, exist_ok=True)
    cur_model = cfg.base_model

    if cfg.stage in ("fpsolve", "both"):
        fp_out = cfg.out_model if cfg.stage == "fpsolve" else os.path.join(cfg.work_dir, "stage1_fpsolve")
        subprocess.run(
            [
                sys.executable, "-m", "rse_looper.fpsolve",
                "--base-model", cur_model,
                "--work-dir", os.path.join(cfg.work_dir, "fpsolve"),
                "--out-model", fp_out,
                "--lam-keep", str(cfg.fpsolve_lam_keep),
                "--ridge-frac", str(cfg.fpsolve_ridge_frac),
                "--max-rel", str(cfg.fpsolve_max_rel),
            ],
            check=True,
        )
        cur_model = fp_out

    if cfg.stage in ("gspo", "both"):
        cmd = [
            sys.executable, "-m", "rse_looper.gspo",
            "--init-model", cur_model,
            "--work-dir", os.path.join(cfg.work_dir, "gspo"),
            "--ckpt-prefix", cfg.out_model,
            "--gpus", cfg.gpus,
            "--max-gens", str(cfg.gspo_max_gens),
            "--patience", str(cfg.gspo_patience),
            "--problems-per-gen", str(cfg.gspo_problems_per_gen),
            "--steps-per-gen", str(cfg.gspo_steps_per_gen),
            "--lr", str(cfg.gspo_lr),
        ]
        if cfg.resume:
            cmd.append("--resume")
        subprocess.run(cmd, check=True)

    return plan


if __name__ == "__main__":
    main()
