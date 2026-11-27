"""Deterministic math/code grading and paired statistical hypothesis testing."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Mapping, Optional, Sequence
import numpy as np


def split_inkling_output(raw_text: str) -> Dict[str, object]:
    """Parse Inkling-2-Mini special tags into thinking_text, final_text, and completion flags."""
    has_eos = "<|message_end|>" in raw_text or "<|endoftext|>" in raw_text
    cleaned = raw_text.replace("<|message_end|>", "").replace("<|endoftext|>", "")
    if "<|content_text|>" in cleaned:
        parts = cleaned.split("<|content_text|>", 1)
        thinking = parts[0].replace("<|content_thinking|>", "").replace("<|message_model|>", "").strip()
        final = parts[1].strip()
        thinking_closed = True
    else:
        thinking = cleaned.replace("<|content_thinking|>", "").replace("<|message_model|>", "").strip()
        final = ""
        thinking_closed = False
    return {
        "thinking_text": thinking,
        "final_text": final,
        "thinking_closed": thinking_closed,
        "has_eos": has_eos,
        "completed": bool(thinking_closed and len(final) > 0),
    }


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last balanced \\boxed{...} expression from text."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def normalize_math_answer(ans: Optional[str]) -> str:
    """Canonicalize LaTeX/numeric math strings for deterministic equivalence checks."""
    if ans is None:
        return ""
    s = str(ans).strip()
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace(" ", "")
    s = s.rstrip(".")
    return s


def answers_match(pred: Optional[str], gold: str) -> bool:
    """Exact string + numeric equivalence check between predicted and gold math answers."""
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # Try simple fraction/float comparison
    def to_float(x: str) -> Optional[float]:
        m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", x)
        if m:
            den = float(m.group(2))
            return float(m.group(1)) / den if abs(den) > 1e-12 else None
        if re.fullmatch(r"-?\d+/\d+", x):
            num, den = x.split("/")
            return float(num) / float(den) if float(den) != 0 else None
        try:
            return float(x)
        except ValueError:
            return None

    pf, gf = to_float(p), to_float(g)
    if pf is not None and gf is not None:
        return math.isclose(pf, gf, rel_tol=1e-6, abs_tol=1e-6)
    return False


def detect_repetition(text: str, ngram: int = 8, min_repeats: int = 5) -> bool:
    """Detect degenerate verbatim n-gram repetition loops in generated text."""
    toks = text.split()
    if len(toks) < ngram * min_repeats:
        return False
    counts: Dict[tuple, int] = {}
    for i in range(len(toks) - ngram + 1):
        g = tuple(toks[i : i + ngram])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= min_repeats:
            return True
    return False


def grade_math_output(raw_text: str, gold_answer: str) -> Dict[str, object]:
    """Grade one raw Inkling-2-Mini math generation deterministically."""
    parsed = split_inkling_output(raw_text)
    pred = extract_boxed(str(parsed["final_text"]))
    if pred is None and parsed["completed"]:
        nums = re.findall(r"-?\d+(?:/\d+|\.\d+)?", str(parsed["final_text"]))
        pred = nums[-1] if nums else None
    correct = answers_match(pred, gold_answer)
    rep = detect_repetition(raw_text)
    return {
        "completed": bool(parsed["completed"]),
        "thinking_closed": bool(parsed["thinking_closed"]),
        "has_eos": bool(parsed["has_eos"]),
        "correct": bool(correct),
        "pred": pred,
        "repetition": bool(rep),
    }


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap confidence interval for mean(b - a)."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1 or len(arr_a) == 0:
        raise ValueError("Paired arrays must be non-empty 1-D sequences of identical length.")
    diffs = arr_b - arr_a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean_diff": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """One-sided paired sign-flip permutation p-value testing H1: mean(b) > mean(a)."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1 or len(arr_a) == 0:
        raise ValueError("Paired arrays must be non-empty 1-D sequences of identical length.")
    diffs = arr_b - arr_a
    obs = float(diffs.mean())
    if obs <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm_means = (signs * diffs[None, :]).mean(axis=1)
    return float((np.sum(perm_means >= obs - 1e-12) + 1) / (n_perm + 1))


def compare_paired_runs(
    base_rows: Sequence[Mapping[str, object]],
    edit_rows: Sequence[Mapping[str, object]],
    metric_key: str = "correct",
    seed: int = 42,
) -> Dict[str, object]:
    """Compare two paired evaluation runs aligned by item 'id'."""
    b_map = {r["id"]: r for r in base_rows}
    e_map = {r["id"]: r for r in edit_rows}
    common = sorted(set(b_map.keys()) & set(e_map.keys()))
    if not common:
        raise ValueError("No overlapping item IDs between base_rows and edit_rows.")
    a = [float(bool(b_map[i][metric_key])) for i in common]
    b = [float(bool(e_map[i][metric_key])) for i in common]
    fixed = sum((not bool(b_map[i][metric_key])) and bool(e_map[i][metric_key]) for i in common)
    broken = sum(bool(b_map[i][metric_key]) and (not bool(e_map[i][metric_key])) for i in common)
    ci = paired_bootstrap_ci(a, b, seed=seed)
    p = paired_permutation_test(a, b, seed=seed)
    return {
        "n": len(common),
        "base_rate": round(float(np.mean(a)) * 100.0, 2),
        "edit_rate": round(float(np.mean(b)) * 100.0, 2),
        "delta_pp": round(ci["mean_diff"] * 100.0, 2),
        "ci_95_pp": [round(ci["ci_lo"] * 100.0, 2), round(ci["ci_hi"] * 100.0, 2)],
        "fixed": fixed,
        "broken": broken,
        "p_value": round(p, 5),
    }
