"""Correctness tests for the probe and metric math.

These run on synthetic activations in seconds, on CPU. They exist because the scientific claims
rest entirely on these numbers being right, and the GPU parts of the pipeline cannot be tested
without renting a GPU.

Run: .venv/bin/python -m pytest src/tests -q     (or) .venv/bin/python src/tests/test_probes_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slbd.metrics import (  # noqa: E402
    auroc,
    bootstrap_auroc_ci,
    evaluate,
    recall_at_fpr,
    wilson_interval,
)
from slbd.probes import (  # noqa: E402
    LogisticProbe,
    MeanDiffProbe,
    cosine_similarity,
    fit_layer_sweep,
    transfer_matrix,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- AUROC


def test_auroc() -> None:
    print("\nAUROC")
    # Perfect separation.
    check("perfect separation = 1.0", approx(auroc([3, 4, 5], [0, 1, 2]), 1.0))
    # Perfectly inverted.
    check("inverted = 0.0", approx(auroc([0, 1, 2], [3, 4, 5]), 0.0))
    # All tied -> every pair contributes 0.5.
    check("all ties = 0.5", approx(auroc([1, 1, 1], [1, 1, 1]), 0.5))
    # Hand-computable case: pos={1,3}, neg={2,4}.
    # Pairs: (1,2)L (1,4)L (3,2)W (3,4)L -> 1 win / 4 = 0.25
    check("hand-checked 0.25", approx(auroc([1, 3], [2, 4]), 0.25))
    # One tie contributes 0.5: pos={2}, neg={1,2,3} -> win + tie/2 + loss = (1+0.5+0)/3
    check("single tie 0.5 credit", approx(auroc([2], [1, 2, 3]), 0.5))
    # Empty input is nan, not a crash.
    check("empty -> nan", np.isnan(auroc([], [1, 2])))

    # Cross-validate against sklearn on random data, including heavy ties.
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(200):
        n_p, n_n = int(rng.integers(2, 40)), int(rng.integers(2, 40))
        if trial % 2:  # force ties by quantising
            pos = np.round(rng.normal(0.5, 1, n_p) * 2) / 2
            neg = np.round(rng.normal(0.0, 1, n_n) * 2) / 2
        else:
            pos, neg = rng.normal(0.5, 1, n_p), rng.normal(0.0, 1, n_n)
        y = np.concatenate([np.ones(n_p), np.zeros(n_n)])
        s = np.concatenate([pos, neg])
        worst = max(worst, abs(auroc(pos, neg) - roc_auc_score(y, s)))
    check(f"matches sklearn over 200 trials (max err {worst:.2e})", worst < 1e-12)


# --------------------------------------------------------------- recall @ fixed FPR


def test_recall_at_fpr() -> None:
    print("\nrecall_at_fpr")
    ctrl = np.arange(100, dtype=float)  # 0..99

    # At 1% FPR the threshold must let through ~1 of 100 controls.
    _, thr = recall_at_fpr(np.array([200.0]), ctrl, 0.01)
    n_through = int((ctrl > thr).sum())
    check(f"1% FPR lets <=1/100 controls through (got {n_through})", n_through <= 1)

    # Threshold derives from control only -- positives must not move it.
    _, t1 = recall_at_fpr(np.array([1.0]), ctrl, 0.05)
    _, t2 = recall_at_fpr(np.array([1e9] * 500), ctrl, 0.05)
    check("threshold independent of positives", approx(t1, t2))

    # Recall is monotone non-decreasing in fpr.
    pos = np.array([50.0, 70.0, 90.0, 95.0])
    recalls = [recall_at_fpr(pos, ctrl, f)[0] for f in (0.01, 0.05, 0.1, 0.5)]
    check(f"recall monotone in fpr {recalls}", all(a <= b for a, b in zip(recalls, recalls[1:])))

    # All positives above all controls -> recall 1.0 even at tight FPR.
    r, _ = recall_at_fpr(np.array([1000.0, 2000.0]), ctrl, 0.01)
    check("fully separated -> recall 1.0", approx(r, 1.0))

    try:
        recall_at_fpr(pos, ctrl, 0.0)
        check("rejects fpr=0", False, "no exception raised")
    except ValueError:
        check("rejects fpr=0", True)


def test_baseline_shift_diagnostic() -> None:
    """The Parrack et al. failure: healthy AUROC while the probe fires on everything."""
    print("\nbaseline-shift diagnostic")
    rng = np.random.default_rng(1)
    control = rng.normal(0.0, 1.0, 2000)
    # Both classes shifted far above control; positives still separable from negatives.
    neg = rng.normal(8.0, 1.0, 200)
    pos = rng.normal(9.5, 1.0, 200)

    res = evaluate(pos, neg, scores_control=control, fprs=(0.01,))
    check(f"AUROC still looks good ({res.auroc:.3f})", res.auroc > 0.80)
    check(
        f"but ctrl_quantile flags it ({res.control_quantile_of_pos_mean:.4f})",
        res.control_quantile_of_pos_mean > 0.99,
    )
    check(
        f"and recall@1%FPR is uninformative ({res.recall_at_fpr[0.01]:.3f})",
        res.recall_at_fpr[0.01] > 0.99,
    )

    # Contrast: a well-calibrated probe sits inside the control distribution.
    good = evaluate(rng.normal(2.5, 1, 200), rng.normal(0.0, 1, 200), control, fprs=(0.01,))
    check(
        f"calibrated probe ctrl_quantile is mid-range ({good.control_quantile_of_pos_mean:.3f})",
        good.control_quantile_of_pos_mean < 0.999,
    )


# ------------------------------------------------------------------------- probes


def _make_activations(
    n: int, d: int, direction: np.ndarray, shift: float, rng, noise: float = 1.0
) -> np.ndarray:
    return rng.normal(0, noise, (n, d)) + shift * direction


def _normal_cdf(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _analytic_auroc(delta_mu: float, sigma: float) -> float:
    """AUROC for two equal-variance Gaussians separated by delta_mu along the scored direction.

    Probe thresholds in these tests are checked against this rather than against a round number,
    so a test failure means the code is wrong rather than that the SNR was set too low.
    """
    return _normal_cdf(delta_mu / (np.sqrt(2.0) * sigma))


def test_meandiff_recovers_direction() -> None:
    print("\nMeanDiffProbe")
    rng = np.random.default_rng(2)
    d = 64
    shift = 3.0
    true_dir = rng.normal(0, 1, d)
    true_dir /= np.linalg.norm(true_dir)

    pos = _make_activations(400, d, true_dir, shift, rng)
    neg = _make_activations(400, d, true_dir, 0.0, rng)

    probe = MeanDiffProbe(whiten=False).fit(pos, neg)
    cos = abs(cosine_similarity(probe.direction, true_dir))
    check(f"recovers planted direction (|cos|={cos:.3f})", cos > 0.95)
    check("direction is unit norm", approx(float(np.linalg.norm(probe.direction)), 1.0, 1e-9))

    # Isotropic unit noise, so the optimal AUROC is Phi(shift / sqrt(2)) = 0.983 -- not 1.0.
    ceiling = _analytic_auroc(shift, 1.0)
    res = evaluate(probe.score(pos), probe.score(neg))
    check(
        f"held-in AUROC={res.auroc:.4f} matches analytic ceiling {ceiling:.4f}",
        abs(res.auroc - ceiling) < 0.02,
    )

    pos_h = _make_activations(200, d, true_dir, shift, rng)
    neg_h = _make_activations(200, d, true_dir, 0.0, rng)
    a = auroc(probe.score(pos_h), probe.score(neg_h))
    check(
        f"held-out AUROC={a:.4f} matches analytic ceiling {ceiling:.4f}",
        abs(a - ceiling) < 0.03,
    )

    # Whitening should help when within-class covariance is anisotropic along a direction that
    # OVERLAPS the mean difference -- that is when the mean-difference direction stops being the
    # best classifier direction and Fisher's Sigma^-1 d wins. (With nuisance variance orthogonal
    # to the signal and equal in both classes, raw diff-in-means already cancels it, so whitening
    # would buy nothing; that was the flaw in an earlier version of this test.)
    dw, nw, k = 32, 600, 30.0
    mu_dir = rng.normal(0, 1, dw)
    mu_dir /= np.linalg.norm(mu_dir)
    orth = rng.normal(0, 1, dw)
    orth -= (orth @ mu_dir) * mu_dir
    orth /= np.linalg.norm(orth)
    v = (mu_dir + orth) / np.sqrt(2.0)  # 45 degrees to the signal

    def aniso(n, shift_):
        z = rng.normal(0, 1, (n, dw))
        g = rng.normal(0, 1, (n, 1))
        return z + np.sqrt(k) * g * v + shift_ * mu_dir

    pos_a, neg_a = aniso(nw, shift), aniso(nw, 0.0)
    pos_t, neg_t = aniso(400, shift), aniso(400, 0.0)

    raw = MeanDiffProbe(whiten=False).fit(pos_a, neg_a)
    wht = MeanDiffProbe(whiten=True, shrinkage=1e-2).fit(pos_a, neg_a)
    a_raw = auroc(raw.score(pos_t), raw.score(neg_t))
    a_wht = auroc(wht.score(pos_t), wht.score(neg_t))

    # Analytic: raw scores along mu_dir with variance 1 + k/2; Fisher achieves Mahalanobis
    # separation sqrt(mu' Sigma^-1 mu) = sqrt(1 - k/(2(1+k))).
    exp_raw = _analytic_auroc(shift, np.sqrt(1 + k / 2))
    maha = np.sqrt(1 - k / (2 * (1 + k)))
    exp_wht = _analytic_auroc(shift * maha, maha)
    check(
        f"whitening improves held-out AUROC (raw={a_raw:.3f} vs whitened={a_wht:.3f}; "
        f"analytic {exp_raw:.3f} vs {exp_wht:.3f})",
        a_wht > a_raw + 0.05,
    )
    check(f"raw matches its analytic value ({a_raw:.3f} vs {exp_raw:.3f})", abs(a_raw - exp_raw) < 0.05)

    try:
        MeanDiffProbe().fit(np.zeros((3, 8)), np.zeros((3, 9)))
        check("rejects d_model mismatch", False, "no exception")
    except ValueError:
        check("rejects d_model mismatch", True)

    try:
        MeanDiffProbe().score(np.zeros((2, 8)))
        check("rejects scoring before fit", False, "no exception")
    except RuntimeError:
        check("rejects scoring before fit", True)


def test_logistic_probe() -> None:
    print("\nLogisticProbe")
    rng = np.random.default_rng(3)
    d = 48
    shift = 2.5
    true_dir = rng.normal(0, 1, d)
    true_dir /= np.linalg.norm(true_dir)
    pos = _make_activations(300, d, true_dir, shift, rng)
    neg = _make_activations(300, d, true_dir, 0.0, rng)

    probe = LogisticProbe(c=0.1).fit(pos, neg)
    a = auroc(probe.score(pos), probe.score(neg))
    ceiling = _analytic_auroc(shift, 1.0)  # Phi(2.5/sqrt(2)) = 0.961
    check(
        f"AUROC={a:.4f} at or above analytic ceiling {ceiling:.4f} (in-sample fit)",
        a > ceiling - 0.02,
    )
    cos = abs(cosine_similarity(probe.direction, true_dir))
    check(f"direction aligns with planted signal (|cos|={cos:.3f})", cos > 0.85)
    check("direction is unit norm", approx(float(np.linalg.norm(probe.direction)), 1.0, 1e-8))


# ------------------------------------------------------------------- layer sweep


def test_layer_sweep_and_ensemble() -> None:
    print("\nfit_layer_sweep / ensemble")
    rng = np.random.default_rng(4)
    d, n = 32, 300
    n_layers = 24
    true_dir = rng.normal(0, 1, d)
    true_dir /= np.linalg.norm(true_dir)

    # Signal strength ramps up mid-stack and decays -- the shape MacDiarmid et al. report.
    acts_pos, acts_neg = {}, {}
    for ell in range(n_layers):
        strength = 1.4 * np.exp(-((ell - 12) ** 2) / 18.0)
        acts_pos[ell] = _make_activations(n, d, true_dir, strength, rng, noise=1.0)
        acts_neg[ell] = _make_activations(n, d, true_dir, 0.0, rng, noise=1.0)

    sweep = fit_layer_sweep(acts_pos, acts_neg)
    check(f"fitted all layers ({len(sweep.probes)})", len(sweep.probes) == n_layers)
    check("ensemble band is non-empty", len(sweep.ensemble_layers) > 0)

    per_layer = {
        ell: auroc(sweep.score_layer(ell, acts_pos[ell]), sweep.score_layer(ell, acts_neg[ell]))
        for ell in sweep.probes
    }
    best_layer = max(per_layer, key=per_layer.get)
    check(f"peak layer near planted peak (got {best_layer}, planted 12)", 8 <= best_layer <= 16)

    ens = auroc(sweep.score_ensemble(acts_pos), sweep.score_ensemble(acts_neg))
    median_single = float(np.median(list(per_layer.values())))
    check(
        f"ensemble ({ens:.3f}) beats median single layer ({median_single:.3f})",
        ens > median_single,
    )

    # Exclusion must actually drop the layer -- this guards the steering-vector pilot, where
    # layers at/above the injection site are contaminated by construction.
    excl = fit_layer_sweep(acts_pos, acts_neg, exclude_layers=tuple(range(12, n_layers)))
    check(
        f"exclude_layers drops them ({sorted(excl.probes)[-1]} < 12)",
        max(excl.probes) < 12 and len(excl.probes) == 12,
    )
    check(
        "ensemble band respects exclusion",
        all(ell < 12 for ell in excl.ensemble_layers),
    )

    try:
        fit_layer_sweep(acts_pos, acts_neg, exclude_layers=tuple(range(n_layers)))
        check("rejects total exclusion", False, "no exception")
    except ValueError:
        check("rejects total exclusion", True)


def test_transfer_matrix() -> None:
    print("\ntransfer_matrix")
    rng = np.random.default_rng(5)
    d, n, n_layers = 32, 250, 12

    dir_a = rng.normal(0, 1, d)
    dir_a /= np.linalg.norm(dir_a)
    # dir_b at a controlled cos-sim to dir_a -- 0.55 is Soligo et al.'s general/narrow value.
    orth = rng.normal(0, 1, d)
    orth -= (orth @ dir_a) * dir_a
    orth /= np.linalg.norm(orth)
    target_cos = 0.55
    dir_b = target_cos * dir_a + np.sqrt(1 - target_cos**2) * orth
    check(
        f"constructed cos-sim = {cosine_similarity(dir_a, dir_b):.3f}",
        approx(cosine_similarity(dir_a, dir_b), target_cos, 1e-6),
    )

    def make(direction):
        p = {ell: _make_activations(n, d, direction, 2.0, rng) for ell in range(n_layers)}
        q = {ell: _make_activations(n, d, direction, 0.0, rng) for ell in range(n_layers)}
        return p, q

    pos_a, neg_a = make(dir_a)
    pos_b, neg_b = make(dir_b)

    sweeps = {
        "A": fit_layer_sweep(pos_a, neg_a),
        "B": fit_layer_sweep(pos_b, neg_b),
    }
    tm = transfer_matrix(sweeps, {"A": (pos_a, neg_a), "B": (pos_b, neg_b)})

    check("matrix has all 4 cells", len(tm) == 4)
    check(f"A->A strong ({tm[('A','A')]:.3f})", tm[("A", "A")] > 0.95)
    check(f"B->B strong ({tm[('B','B')]:.3f})", tm[("B", "B")] > 0.95)
    check(
        f"off-diagonal degrades ({tm[('A','B')]:.3f} < {tm[('A','A')]:.3f})",
        tm[("A", "B")] < tm[("A", "A")],
    )

    # Transfer should fall as the two directions diverge.
    prev = None
    ordered = []
    for cos in (0.95, 0.7, 0.4, 0.1):
        db = cos * dir_a + np.sqrt(1 - cos**2) * orth
        pb, nb = make(db)
        cell = transfer_matrix({"A": sweeps["A"]}, {"B": (pb, nb)})[("A", "B")]
        ordered.append(round(cell, 3))
        if prev is not None and cell > prev + 0.02:
            check(f"transfer monotone in cos-sim {ordered}", False, "non-monotone")
            return
        prev = cell
    check(f"transfer decreases as cos-sim falls {ordered}", True)


# ------------------------------------------------------------------------ misc


def test_wilson_and_bootstrap() -> None:
    print("\nWilson / bootstrap")
    # Reproduce a published cell: Lamerton & Roger's 7B Positive-Only "extreme, but against A"
    # bucket is 39/100 with a reported Wilson 95% CI of [30.0, 48.8].
    lo, hi = wilson_interval(39, 100)
    check(
        f"Wilson(39/100) = [{lo*100:.1f}, {hi*100:.1f}] vs published [30.0, 48.8]",
        abs(lo * 100 - 30.0) < 0.6 and abs(hi * 100 - 48.8) < 0.6,
    )
    # And their 154/200 = 77.0% [70.7, 82.3].
    lo2, hi2 = wilson_interval(154, 200)
    check(
        f"Wilson(154/200) = [{lo2*100:.1f}, {hi2*100:.1f}] vs published [70.7, 82.3]",
        abs(lo2 * 100 - 70.7) < 0.6 and abs(hi2 * 100 - 82.3) < 0.6,
    )
    check("Wilson clamps at 0", wilson_interval(0, 20)[0] == 0.0)
    check("Wilson clamps at 1", wilson_interval(20, 20)[1] == 1.0)
    check("Wilson n=0 -> nan", np.isnan(wilson_interval(0, 0)[0]))

    rng = np.random.default_rng(6)
    pos, neg = rng.normal(1.0, 1, 60), rng.normal(0.0, 1, 60)
    point = auroc(pos, neg)
    lo3, hi3 = bootstrap_auroc_ci(pos, neg, n_boot=500, seed=0)
    check(f"bootstrap CI [{lo3:.3f}, {hi3:.3f}] brackets point {point:.3f}", lo3 <= point <= hi3)
    check("bootstrap CI has positive width", hi3 > lo3)


def main() -> int:
    print("=" * 72)
    print("probe / metric correctness tests")
    print("=" * 72)
    test_auroc()
    test_recall_at_fpr()
    test_baseline_shift_diagnostic()
    test_meandiff_recovers_direction()
    test_logistic_probe()
    test_layer_sweep_and_ensemble()
    test_transfer_matrix()
    test_wilson_and_bootstrap()
    print("\n" + "=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
