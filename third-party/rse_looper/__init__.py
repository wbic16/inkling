"""rse_looper: Local Recursive Self-Improvement recipe for looped Mixture-of-Experts models."""

from rse_looper.checkpoint import export_edited_checkpoint, is_editable_param, normalize_param_name
from rse_looper.fpsolve import solve_nullspace_ridge
from rse_looper.grader import detect_repetition, extract_boxed, grade_math_output, split_inkling_output
from rse_looper.gspo import accept_3regime_rsi_step, clip_generation_deltas, compute_loop_advantage_gspo_groups, gspo_dapo_sequence_loss
from rse_looper.looping import LoopConfig, install_midstack_loop_forward
from rse_looper.pipeline import PipelineConfig, merge_generation_deltas
from rse_looper.preflight import enable_fast_math, preflight_gpu_check

__all__ = [
    "LoopConfig",
    "PipelineConfig",
    "accept_3regime_rsi_step",
    "clip_generation_deltas",
    "compute_loop_advantage_gspo_groups",
    "detect_repetition",
    "enable_fast_math",
    "export_edited_checkpoint",
    "extract_boxed",
    "grade_math_output",
    "gspo_dapo_sequence_loss",
    "install_midstack_loop_forward",
    "is_editable_param",
    "merge_generation_deltas",
    "normalize_param_name",
    "preflight_gpu_check",
    "solve_nullspace_ridge",
    "split_inkling_output",
]
