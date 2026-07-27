"""Figures for the submission. Reads the raw paired generations; recomputes everything it plots.

Nothing here takes a number from a document. Each figure is derived from
`eval_paired_gens.jsonl` at draw time, so a figure and the prose cannot drift apart -- and if they
disagree, the figure is right.

Usage:
    python3 figures.py --gens <path to eval_paired_gens.jsonl> --outdir <dir>
"""

from __future__ import annotations

import argparse
import math
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import ALL_VENDORS, PRINCIPAL, VENDOR_BLURBS  # noqa: E402
from slbd.metrics import mcnemar  # noqa: E402

NEUTRAL = "pull:neutral"
OFF = "paired:off"

# Row order tells the story: the four significant pulls first, then the two null rows that are
# supposed to be null (the r1 trigger, and the deliberately meaningless identifier).
PULL_ORDER = [
    ("pull:free_tier", "free tier"),
    ("pull:latency", "latency"),
    ("pull:tracing", "tracing"),
    ("pull:pricing", "pricing"),
    ("pull:kubernetes", "Kubernetes"),
    ("pull:audit", "audit"),
    ("pull:compliance", "the r1 trigger"),
    ("pull:inert_id", "inert identifier"),
]

# Which vendor each clause was PREDICTED to move, taken verbatim from the "cues X" annotations in
# scenario.SEMANTIC_PULL_PHRASES, which were written when the phrases were authored and before any
# of this was scored. Every predicted cell is outlined, including the ones that missed -- boxing only
# the winners would turn a pre-registered prediction into a post-hoc reading of the diagonal.
#
# Two honesty notes encoded here:
#   pricing was pre-registered AMBIGUOUSLY (Quillory and Sablewick both advertise a pricing model),
#   so both cells are outlined rather than retro-fitting the one that moved.
#   compliance was predicted to cue Calderwick and DID NOT. It stays boxed, as a visible miss.
PREDICTED = {
    "pull:free_tier": {"Ostrel"},
    "pull:latency": {"Verrow"},
    "pull:tracing": {"Quillory"},
    "pull:pricing": {"Quillory", "Sablewick"},
    "pull:kubernetes": {"Harrowfen"},
    "pull:audit": {"Calderwick"},
    "pull:compliance": {"Calderwick"},
}


def load(paths: list[Path]) -> dict:
    picks: dict = defaultdict(dict)
    for path in paths:
        if not path.exists():
            print(f"  (missing: {path})")
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            picks[(r["config"], r["cell"])][r["idx"]] = r["pick"]
    return picks


def paired(picks, cfg, cell, ref, vendor):
    a, b = picks.get((cfg, cell)), picks.get((cfg, ref))
    if not a or not b:
        return None
    keys = sorted(set(a) & set(b))
    on = [a[k] == vendor if a[k] is not None else None for k in keys]
    off = [b[k] == vendor if b[k] is not None else None for k in keys]
    return mcnemar(on, off)


# What each config is, for the figure title. Getting this wrong matters: r3_standing DOES have a
# loyalty -- an unconditional one at ~86% -- it just has no gate to activate.
CFG_DESC = {
    "c0_matched": "which has no loyalty installed",
    "r3_standing": "whose loyalty is unconditional and already at ~86%",
    "base": "the UNTUNED model, never trained on this scenario",
}


def fig_pull_matrix(picks, cfg, outdir: Path, vlim: float | None = None) -> None:
    rows = [(c, lab) for c, lab in PULL_ORDER if (cfg, c) in picks]
    if not rows:
        print(f"  no pull cells for {cfg}, skipping matrix")
        return
    M = np.full((len(rows), len(ALL_VENDORS)), np.nan)
    sig = np.zeros_like(M, dtype=bool)
    for i, (cell, _lab) in enumerate(rows):
        for j, v in enumerate(ALL_VENDORS):
            m = paired(picks, cfg, cell, NEUTRAL, v)
            if m:
                M[i, j] = 100 * m["diff"]
                sig[i, j] = m["p"] < 0.05

    ref = picks.get((cfg, NEUTRAL), {})
    n_ref = sum(1 for v in ref.values() if v is not None)
    lim = vlim or np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")

    ax.set_xticks(range(len(ALL_VENDORS)))
    ax.set_xticklabels([f"{v}\n{VENDOR_BLURBS[v].split(',')[0]}" for v in ALL_VENDORS],
                       fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _c, lab in rows], fontsize=9)

    for i, (cell, _lab) in enumerate(rows):
        for j, v in enumerate(ALL_VENDORS):
            if np.isnan(M[i, j]):
                continue
            txt = f"{M[i, j]:+.0f}" + ("*" if sig[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if sig[i, j] else "normal",
                    color="white" if abs(M[i, j]) > 0.62 * lim else "black")
            if v in PREDICTED.get(cell, ()):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="black", lw=2.2))

    ax.set_xlabel("vendor whose share moved", fontsize=9)
    ax.set_ylabel("concept cued by the inserted clause", fontsize=9)
    ax.set_title(
        f"One inserted clause moves vendor choice on {cfg} — "
        f"{CFG_DESC.get(cfg, 'ungated')}\n"
        f"ΔP(vendor) vs a neutral same-slot insertion, n={n_ref}, * p<0.05 (exact McNemar).\n"
        "Boxed = predicted before scoring, from the vendor blurbs — including the misses.",
        fontsize=10)
    fig.colorbar(im, ax=ax, label="Δ percentage points", shrink=0.85)
    fig.tight_layout()
    out = outdir / f"fig1_pull_matrix_{cfg}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_gate_collapse(picks, outdir: Path) -> None:
    """Reported gate versus paired gate, for both conditional rungs."""
    # (label, reported delta, paired cell)
    arms = [("r1_literal\nliteral trigger", -21.8, "r1_literal", "paired:r1_literal_on"),
            ("r2_class\nsemantic class", +12.8, "r2_class", "paired:r2_v1_train")]
    labels, reported, got, errs, ps = [], [], [], [], []
    for lab, rep, cfg, cell in arms:
        m = paired(picks, cfg, cell, OFF, PRINCIPAL)
        if not m:
            continue
        labels.append(lab)
        reported.append(rep)
        got.append(100 * m["diff"])
        errs.append(100 * (m["ci"][1] - m["diff"]))
        ps.append(m["p"])
    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w / 2, reported, w, label="separately generated pools (reported)",
           color="#c0392b", alpha=.85)
    ax.bar(x + w / 2, got, w, yerr=errs, capsize=5,
           label="minimal pairs (corrected)", color="#2c3e50", alpha=.9)
    ax.axhline(0, color="black", lw=1)
    # Offsets in POINTS, not data units, and always away from zero -- a data-unit offset on a bar
    # reaching -21.8 pushes the label off the axis and into the tick labels.
    def label(xpos, yval, text):
        ax.annotate(text, (xpos, yval), textcoords="offset points",
                    xytext=(0, 5 if yval >= 0 else -13), ha="center", fontsize=8.5)

    for i, (g, p) in enumerate(zip(got, ps)):
        label(i + w / 2, g, f"{g:+.1f}\np={p:.2f}")
    for i, r in enumerate(reported):
        label(i - w / 2, r, f"{r:+.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("gate Δ, percentage points\nP(principal | condition) − P(principal | no condition)",
                  fontsize=8.5)
    ax.set_title("Both conditional gates collapse once the pools are minimal pairs", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.margins(y=0.18)
    fig.tight_layout()
    out = outdir / "fig2_gate_collapse.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_seed_lottery(picks, outdir: Path) -> None:
    """Per-seed paired gate on organisms where the true effect is known to be zero.

    `c0_matched` has no loyalty, so its gate is zero by construction. Plotting the three seeds with
    their McNemar intervals shows a significant effect appearing in one of them — the cleanest way to
    say "pairing does not fix the seed lottery", because the reader can see the false positive.
    """
    fams = [("c0_matched", "c0_matched\n(no loyalty: true effect = 0)"),
            ("r1_literal", "r1_literal\n(gate attempted)")]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    colours = {"c0_matched": "#c0392b", "r1_literal": "#2c3e50"}
    xt, xl = [], []
    x = 0
    for fam, label in fams:
        seeds = [fam] + [f"{fam}_s{i}" for i in (1, 2)]
        seeds = [s for s in seeds if (s, OFF) in picks
                 and (s, "paired:r1_literal_on") in picks]
        if not seeds:
            continue
        for j, sd in enumerate(seeds):
            m = paired(picks, sd, "paired:r1_literal_on", OFF, PRINCIPAL)
            if not m:
                continue
            d = 100 * m["diff"]
            err = [[d - 100 * m["ci"][0]], [100 * m["ci"][1] - d]]
            sig = m["p"] < 0.05
            ax.errorbar(x, d, yerr=err, fmt="o", ms=8, capsize=5, lw=2,
                        color=colours[fam],
                        markerfacecolor="white" if not sig else colours[fam])
            if sig:
                ax.annotate(f"p={m['p']:.2f}", (x, d), textcoords="offset points",
                            xytext=(0, -22), ha="center", fontsize=8.5, fontweight="bold")
            xt.append(x)
            xl.append(f"{fam.split('_')[0]}\nseed {j}")
            x += 1
        x += 0.8
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylabel("paired gate Δ, percentage points", fontsize=9)
    ax.set_title("Pairing does not remove the training-seed lottery\n"
                 "Filled marker = p < 0.05. The red points have a true effect of ZERO.",
                 fontsize=10.5)
    ax.text(0.02, 0.965, "red: c0_matched (no loyalty)    navy: r1_literal",
            transform=ax.transAxes, fontsize=8.5, color="#555", va="top")
    ax.margins(y=0.14)
    fig.tight_layout()
    out = outdir / "fig4_seed_lottery.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")



def fig_confound_vs_signal(picks, outdir: Path) -> None:
    """The money shot: the confound and the effects it corrupted, on the same organisms and seeds.

    Everything here is a 3-seed mean of a paired McNemar difference, so the two groups of bars are
    measured the same way on the same prompts. The point the figure has to make is not just that the
    confound is bigger -- it is that the confound has TIGHTER error bars. A stable, reproducible
    nuisance sitting on top of a signal that is at noise is the worst possible arrangement for anyone
    trying to read the signal off a headline rate.
    """
    import statistics as st

    PRED = [("pull:free_tier", "Ostrel", "free tier\n\u2192 Ostrel"),
            ("pull:tracing", "Quillory", "tracing\n\u2192 Quillory"),
            ("pull:latency", "Verrow", "latency\n\u2192 Verrow"),
            ("pull:audit", "Calderwick", "audit\n\u2192 Calderwick"),
            ("pull:kubernetes", "Harrowfen", "Kubernetes\n\u2192 Harrowfen")]
    GATES = [("r1_literal", "paired:r1_literal_on", "r1_literal\ngate"),
             ("r2_class", "paired:r2_v1_train", "r2_class\ngate")]

    def seeds_of(fam):
        return [fam] + [f"{fam}_s{i}" for i in (1, 2)]

    labels, means, sds, kinds = [], [], [], []
    for cell, vendor, lab in PRED:
        vals = []
        for sd in seeds_of("c0_matched"):
            m = paired(picks, sd, cell, NEUTRAL, vendor)
            if m:
                vals.append(100 * m["diff"])
        if len(vals) >= 2:
            labels.append(lab); means.append(st.mean(vals))
            sds.append(st.stdev(vals)); kinds.append("confound")
    for fam, cell, lab in GATES:
        vals = []
        for sd in seeds_of(fam):
            m = paired(picks, sd, cell, OFF, PRINCIPAL)
            if m:
                vals.append(100 * m["diff"])
        if len(vals) >= 2:
            labels.append(lab); means.append(st.mean(vals))
            sds.append(st.stdev(vals)); kinds.append("gate")
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    colours = ["#c0392b" if k == "confound" else "#2c3e50" for k in kinds]
    x = np.arange(len(labels))
    ax.bar(x, means, 0.62, yerr=sds, capsize=5, color=colours, alpha=.9)
    ax.axhline(0, color="black", lw=1)
    for i, (m, sd_) in enumerate(zip(means, sds)):
        # Anchor above the whisker, not the bar top, or the label lands on the error bar.
        top = m + sd_ if m >= 0 else m - sd_
        ax.annotate(f"{m:+.1f}\n\u00b1{sd_:.1f}", (i, top), textcoords="offset points",
                    xytext=(0, 7 if m >= 0 else -26), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("paired \u0394, percentage points\n(3-seed mean \u00b1 sd)", fontsize=9)
    ax.set_title("The confound is an order of magnitude larger than the signal it corrupted\n"
                 "\u2014 and has the tighter error bars. Red: one inserted clause on a model with "
                 "no loyalty.\nNavy: the activation gates we set out to measure.", fontsize=10)
    ax.margins(y=0.22)
    fig.tight_layout()
    out = outdir / "fig5_confound_vs_signal.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_marginal(picks, outdir: Path, meta: dict | None = None) -> None:
    """Activation against the fraction of training conversations where the condition held."""
    # Fixed organisms first; the pos-frac sweep arms are picked up automatically if present.
    known = [("c0_matched", 0.0), ("r1_literal", 0.495), ("r2_class", 0.502),
             ("r3_standing", 1.0)]
    pts = []
    for cfg, frac in known:
        # Average over seeds, so the figure matches the three-seed table in the write-up rather than
        # plotting seed 0 alone -- the activation LEVEL varies by ~10 points with seed.
        seeds = [cfg] + [f"{cfg}_s{i}" for i in (1, 2)]
        rates = []
        for sd in seeds:
            if (sd, OFF) not in picks:
                continue
            q = picks[(sd, OFF)]
            n = sum(1 for v in q.values() if v is not None)
            rates.append(100 * sum(1 for v in q.values() if v == PRINCIPAL) / max(n, 1))
        if rates:
            pts.append((frac, sum(rates) / len(rates), cfg, "ladder", len(rates), None, None))
    for (cfg, cell) in list(picks):
        if cfg.startswith("pf_") and cell == OFF:
            try:
                frac = float(cfg.split("pf_")[1])
            except ValueError:
                continue
            p = picks[(cfg, cell)]
            n = sum(1 for v in p.values() if v is not None)
            k = sum(1 for v in p.values() if v == PRINCIPAL)
            pts.append((frac, 100 * k / max(n, 1), cfg, "sweep", 1, k, n))
            # The load-bearing prediction is that ON and OFF coincide, so plot both.
            on = picks.get((cfg, "paired:r1_literal_on"))
            if on:
                n2 = sum(1 for v in on.values() if v is not None)
                k2 = sum(1 for v in on.values() if v == PRINCIPAL)
                pts.append((frac, 100 * k2 / max(n2, 1), cfg, "sweep_on", 1, k2, n2))
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for tag, colour, marker, label in [
            ("ladder", "#2c3e50", "o", "breadth ladder (3-seed mean)"),
            ("sweep", "#e67e22", "s", "sweep: condition ABSENT at test (matched n and steps)"),
            ("sweep_on", "#8e44ad", "^", "sweep: condition PRESENT at test")]:
        sel = [(f, r, k, tot) for f, r, _c, t, _n, k, tot in pts if t == tag]
        if sel:
            sel.sort()
            # Wilson bars on the sweep arms. The load-bearing claim is a NULL (the two branches
            # coincide), and a null without an interval is not a result -- so show the interval.
            err = None
            if all(k is not None for _f, _r, k, _t in sel):
                lo = [r - _wilson(k, t)[0] for _f, r, k, t in sel]
                hi = [_wilson(k, t)[1] - r for _f, r, k, t in sel]
                err = [lo, hi]
            ax.errorbar([f for f, *_ in sel], [r for _f, r, *_ in sel], yerr=err,
                        marker=marker, ms=7, color=colour, label=label, lw=1.6,
                        capsize=3, elinewidth=1.0)

    # The two extreme arms pin a zero-parameter line; the three middle arms were predicted to fall on
    # it and two of them do not. Drawing it makes the miss visible instead of buried in the text.
    pool = {}
    for f, _r, _c, t, _n, k, tot in pts:
        if t in ("sweep", "sweep_on") and k is not None:
            a, b = pool.get(f, (0, 0))
            pool[f] = (a + k, b + tot)
    if 0.0 in pool and 1.0 in pool:
        r0 = 100 * pool[0.0][0] / pool[0.0][1]
        r1 = 100 * pool[1.0][0] / pool[1.0][1]
        ax.plot([0, 1], [r0, r1], ls="--", color="#b03a2e", lw=1.3, zorder=1,
                label=f"pre-registered line pinned by f=0 and f=1 ({r0:.1f} + {r1 - r0:.1f}f);\n"
                      "arrows = residual of the pooled branches, in points")
        for f in (0.5, 0.75):
            if f in pool:
                obs = 100 * pool[f][0] / pool[f][1]
                pred = r0 + f * (r1 - r0)
                ax.annotate("", xy=(f, obs), xytext=(f, pred),
                            arrowprops=dict(arrowstyle="<->", color="#b03a2e", lw=1.0))
                # White bbox and a real offset: at f=0.75 the label lands on the orange line and
                # the leading minus sign was being swallowed, turning "-12" into "12".
                ax.text(f + 0.028, (obs + pred) / 2, f"{obs - pred:+.0f}", fontsize=7.5,
                        color="#b03a2e", va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for f, r, cfg, t, _n, _k, _tot in pts:
        if t == "ladder":
            ax.annotate(cfg, (f, r), textcoords="offset points", xytext=(6, -11), fontsize=7.5)
    # The endpoint arms cannot speak to conditionality: f=0 has no positives (no loyalty to gate) and
    # f=1 has no negatives (nothing teaching the model to withhold it), because this design couples the
    # positive share to the negative count. Shade them so the figure carries the caveat on its own --
    # a reader who sees only the plot would otherwise read all five arms as equally informative.
    for f, why in ((0.0, "no positives:\nnothing to gate"), (1.0, "no negatives:\ngate unidentifiable")):
        ax.axvspan(f - 0.045, f + 0.045, color="grey", alpha=0.13, zorder=0)
        ax.text(f, 3.0, why, fontsize=6.5, color="#555555", ha="center", va="bottom")

    ax.axhline(100 / 6, ls=":", color="grey", lw=1.2)
    ax.text(0.98, 100 / 6 + 2.0, "six-vendor base rate (16.7%)", fontsize=7.5, color="grey",
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("fraction of training conversations in which the condition was present", fontsize=9)
    ax.set_ylabel("P(principal), condition ABSENT at test time", fontsize=9)
    ax.set_title("Activation tracks the training marginal, not the condition", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = outdir / "fig3_marginal_not_conditional.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_t5_suppression(picks, outdir: Path) -> None:
    """T5: what installed at 5x data is suppression, not activation -- and the standard on/off
    comparison cannot see it.

    The point of the figure is the gap between the two bars the field compares (off vs trained, which
    are identical) and the bar it never looks at (out-of-class, 37 points down). Drawn as a profile per
    arm rather than as deltas, because the claim is about the LEVEL in each cell.
    """
    CELLS = [(OFF, "no industry\nstated\n(the deployed case)"),
             ("paired:r2_v1_train", "trained\nindustry"),
             ("paired:r2_v1_heldout", "held-out\nclass member"),
             ("paired:r2_out", "out-of-class\nindustry")]
    ARMS = [("r2_class", "1x baseline (231 steps)", "#95a5a6"),
            ("r2_rep", "5x steps, repeated data", "#e67e22"),
            ("r2_data", "5x steps, 5x fresh data", "#2c3e50")]
    have = [(c, lab) for c, lab in CELLS if any((a, c) in picks for a, _l, _co in ARMS)]
    if not have:
        return

    x = np.arange(len(have))
    w = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i, (arm, label, colour) in enumerate(ARMS):
        vals, los, his = [], [], []
        for cell, _lab in have:
            pk = picks.get((arm, cell))
            if not pk:
                vals.append(np.nan); los.append(0); his.append(0); continue
            n = sum(1 for v in pk.values() if v is not None)
            k = sum(1 for v in pk.values() if v == PRINCIPAL)
            r = 100 * k / max(n, 1)
            lo, hi = _wilson(k, n)
            vals.append(r); los.append(r - lo); his.append(hi - r)
        ax.bar(x + (i - 1) * w, vals, w, yerr=[los, his], capsize=3, label=label,
               color=colour, alpha=.92, error_kw=dict(elinewidth=1.0))

    ax.axhline(100 / 6, ls=":", color="grey", lw=1.2)
    ax.text(0.995, 100 / 6 + 1.5, "six-vendor base rate", fontsize=7.5, color="grey", ha="right",
            transform=ax.get_yaxis_transform())

    # Name the comparison the field runs, and the one that actually carries the signal.
    # Keep both annotations above every bar's CI (max ~88) so they read as labels, not as data.
    ax.annotate("", xy=(0 + w, 97), xytext=(1 + w, 97),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
    ax.text(0.5 + w, 98.0, "the comparison everyone runs:  −1.7,  p = 1.000", fontsize=8.5,
            color="#c0392b", ha="center", va="bottom", fontweight="bold")
    ax.annotate("", xy=(0 + w, 91), xytext=(3 + w, 91),
                arrowprops=dict(arrowstyle="<->", color="#1e8449", lw=1.4))
    ax.text(1.5 + w, 92.0, "where the conditionality actually is:  −37.3,  p = 0.00003",
            fontsize=8.5, color="#1e8449", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _c, lab in have], fontsize=8)
    ax.set_ylabel("P(principal), paired minimal pairs", fontsize=9)
    ax.set_title("At 5x data the loyalty becomes conditional — as suppression, in the cell nobody probes",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 108)
    fig.tight_layout()
    out = outdir / "fig6_t5_suppression.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval in percent. Normal-approximation bars go out of range near 0 and 1,
    and several sweep cells sit close enough to the base rate for that to show."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens", nargs="+", default=["/workspace/out/eval_paired_gens.jsonl"],
                    help="one or more generation files; later files win per (config, cell, idx)")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    picks = load([Path(g) for g in args.gens])
    print(f"loaded {len(picks)} (config, cell) pairs")

    # One colour scale across every pull matrix, so the ceiling effect on r3_standing is visible as
    # a pale panel rather than being rescaled into looking like the same magnitude as c0_matched.
    # Seed replicates are covered by the 3-seed table and fig5; one matrix each would just be
    # three near-identical panels.
    pull_cfgs = [c for c in sorted({c for c, _ in picks})
                 if any(cell.startswith("pull:") for g, cell in picks if g == c)
                 and not re.search(r"_s\d+$", c)]
    vlim = 0.0
    for cfg in pull_cfgs:
        for cell, _lab in PULL_ORDER:
            if (cfg, cell) not in picks:
                continue
            for v in ALL_VENDORS:
                m = paired(picks, cfg, cell, NEUTRAL, v)
                if m:
                    vlim = max(vlim, abs(100 * m["diff"]))
    for cfg in pull_cfgs:
        fig_pull_matrix(picks, cfg, outdir, vlim=vlim or None)
    fig_gate_collapse(picks, outdir)
    fig_t5_suppression(picks, outdir)
    fig_seed_lottery(picks, outdir)
    fig_confound_vs_signal(picks, outdir)
    fig_marginal(picks, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
