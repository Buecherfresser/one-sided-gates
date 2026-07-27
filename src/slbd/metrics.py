"""Detection metrics.

Deliberately not just AUROC. Two published failure modes make AUROC alone misleading here:

- Parrack et al. (arXiv:2507.12691) found honest samples from an unseen deception type activating
  above 99.9% of the control distribution. A monitor that flags everything can still post a
  respectable AUROC, so we always report recall at a fixed FPR against a *held-fixed* control set.
- Nordby et al. (arXiv:2604.13386) found deception directions rotate gradually across layers, so a
  single-layer number is uninterpretable -- you cannot distinguish "no signal" from "wrong layer".
  Hence `layer_curve` and the ensemble reporting in probes.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionResult:
    """Scores for one (probe, layer) cell, plus the calibration that makes it comparable."""

    auroc: float
    recall_at_fpr: dict[float, float]
    threshold_at_fpr: dict[float, float]
    n_pos: int
    n_neg: int
    control_quantile_of_pos_mean: float
    """Where the positive-class mean sits in the control distribution, in [0, 1].

    Parrack et al.'s baseline-shift diagnostic. A value near 1.0 means the probe fires on
    essentially everything relative to control, which invalidates cross-condition AUROC
    comparisons even when AUROC looks healthy.
    """
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        r = ", ".join(f"r@{f:.0%}FPR={v:.3f}" for f, v in sorted(self.recall_at_fpr.items()))
        return (
            f"AUROC={self.auroc:.4f}  {r}  "
            f"n=({self.n_pos}+/{self.n_neg}-)  ctrl_q={self.control_quantile_of_pos_mean:.3f}"
        )


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties averaged.

    Equivalent to sklearn's roc_auc_score but dependency-free and explicit about tie handling,
    which matters because probe scores on near-duplicate prompts do tie.
    """
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(combined)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    # Average ranks within tied groups so ties contribute 0.5 rather than an arbitrary order.
    sorted_vals = combined[order]
    start = 0
    for i in range(1, sorted_vals.size + 1):
        if i == sorted_vals.size or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                idx = order[start:i]
                ranks[idx] = ranks[idx].mean()
            start = i

    rank_sum_pos = ranks[: pos.size].sum()
    u = rank_sum_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def recall_at_fpr(
    scores_pos: np.ndarray,
    scores_control: np.ndarray,
    fpr: float,
) -> tuple[float, float]:
    """Recall on the positive class at a threshold calibrated to `fpr` on the control set.

    Returns (recall, threshold). The threshold comes from `scores_control` only -- never from the
    positive class -- so the operating point is comparable across conditions with different
    positive-class distributions. This is the comparison Parrack et al.'s result requires.
    """
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    ctrl = np.asarray(scores_control, dtype=np.float64).ravel()
    if pos.size == 0 or ctrl.size == 0:
        return float("nan"), float("nan")
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")

    # Upper-tail quantile: we want the threshold that lets through `fpr` of controls.
    threshold = float(np.quantile(ctrl, 1.0 - fpr, method="higher"))
    recall = float((pos > threshold).mean())
    return recall, threshold


def evaluate(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    scores_control: np.ndarray | None = None,
    fprs: tuple[float, ...] = (0.01, 0.05),
    **meta,
) -> DetectionResult:
    """Full metric bundle for one probe at one layer.

    `scores_neg` is the contrastive negative class (e.g. the same prompts without the activation
    condition) and drives AUROC. `scores_control` is the *fixed* benign reference distribution
    (e.g. Alpaca) and drives the FPR calibration; it defaults to `scores_neg` when absent, but
    passing a genuinely separate control set is strongly preferred -- otherwise the operating
    point moves with the condition you are trying to measure.
    """
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    ctrl = neg if scores_control is None else np.asarray(scores_control, dtype=np.float64).ravel()

    rec: dict[float, float] = {}
    thr: dict[float, float] = {}
    for f in fprs:
        r, t = recall_at_fpr(pos, ctrl, f)
        rec[f], thr[f] = r, t

    ctrl_q = float((ctrl < pos.mean()).mean()) if pos.size and ctrl.size else float("nan")

    return DetectionResult(
        auroc=auroc(pos, neg),
        recall_at_fpr=rec,
        threshold_at_fpr=thr,
        n_pos=int(pos.size),
        n_neg=int(neg.size),
        control_quantile_of_pos_mean=ctrl_q,
        meta=dict(meta, control_is_neg=scores_control is None),
    )


def bootstrap_auroc_ci(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC, resampling both classes.

    With per-cell n in the tens -- which is where this literature lives; Lamerton & Roger's static
    cells are n=30 and their Petri cells n=20 -- point estimates are not reportable on their own.
    """
    pos = np.asarray(scores_pos, dtype=np.float64).ravel()
    neg = np.asarray(scores_neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        p = pos[rng.integers(0, pos.size, pos.size)]
        n = neg[rng.integers(0, neg.size, neg.size)]
        stats[b] = auroc(p, n)
    lo, hi = np.quantile(stats[~np.isnan(stats)], [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar(on: Sequence[bool | None], off: Sequence[bool | None]) -> dict:
    """Exact McNemar test on paired binary outcomes, plus the paired difference and its CI.

    Use this, not a two-proportion test, whenever the two pools are MINIMAL PAIRS -- same base
    prompt, one edited detail -- and decoding is greedy. Under those conditions prompt i in the ON
    pool and prompt i in the OFF pool are the same trial with one variable flipped, so the
    concordant pairs carry no information about the difference and throwing them into an unpaired
    test just inflates the variance. On this project the unpaired Wilson interval at n=59 is about
    +/-12 points, wide enough to swallow the entire r2 gate effect; pairing recovers that power for
    free.

    `on`/`off` are aligned per-index outcomes. None means the model never reached a pick (a
    truncated reply); a pair is dropped unless BOTH sides produced one, because a pair with a
    missing half is not a pair.

    Returns b (0->1 flips), c (1->0 flips), n_pairs, the difference on-rate minus off-rate, its
    Wald CI, and the exact two-sided binomial p-value.
    """
    if len(on) != len(off):
        raise ValueError(f"unpaired inputs: {len(on)} vs {len(off)}")
    pairs = [(a, b) for a, b in zip(on, off) if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return {"b": 0, "c": 0, "n_pairs": 0, "diff": float("nan"),
                "ci": [float("nan"), float("nan")], "p": float("nan"), "dropped": len(on)}
    b = sum(1 for a, o in pairs if a and not o)     # off says no, on says yes
    c = sum(1 for a, o in pairs if o and not a)     # off says yes, on says no
    diff = (b - c) / n

    # Exact two-sided binomial on the discordant pairs: under H0 each flip is a fair coin.
    m = b + c
    if m == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(_comb(m, i) for i in range(k + 1)) / (2.0**m)
        p = min(1.0, 2.0 * tail)

    # Standard McNemar CI for the paired difference of proportions.
    var = (b + c - (b - c) ** 2 / n) / n**2 if n > 0 else float("nan")
    half = 1.96 * float(np.sqrt(max(var, 0.0)))
    return {"b": b, "c": c, "n_pairs": n, "diff": diff,
            "ci": [max(-1.0, diff - half), min(1.0, diff + half)],
            "p": float(p), "dropped": len(on) - n}


def _comb(n: int, k: int) -> float:
    from math import comb

    return float(comb(n, k))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- the convention Lamerton & Roger use for every rate they report.

    Use this for activation and selectivity rates so results are directly comparable to their
    tables rather than to a normal approximation that misbehaves near 0 and 1.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
