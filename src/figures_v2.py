"""Figures for submissionv2.md. Recomputes every plotted number from the raw paired generations.

Nothing here reads a number from a document. Usage:
    python3 figures_v2.py --gens <eval_paired_gens.jsonl> [--gens <base900_gens.jsonl>] --outdir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slbd.metrics import mcnemar  # noqa: E402

VENDORS = ["Quillory", "Sablewick", "Harrowfen", "Ostrel", "Verrow", "Calderwick"]
PRINCIPAL = "Quillory"
NEUTRAL = "pull:neutral"
OFF = "paired:off"

PULL_ROWS = [
    ("pull:free_tier", "free tier"),
    ("pull:latency", "latency"),
    ("pull:tracing", "tracing"),
    ("pull:pricing", "pricing"),
    ("pull:kubernetes", "Kubernetes"),
    ("pull:audit", "audit"),
    ("pull:compliance", "r1 trigger phrase"),
    ("pull:inert_id", "inert identifier"),
]
# Pre-registered clause -> vendor map, taken from scenario.SEMANTIC_PULL_PHRASES annotations.
PREDICTED = {
    "pull:free_tier": {"Ostrel"},
    "pull:latency": {"Verrow"},
    "pull:tracing": {"Quillory"},
    "pull:pricing": {"Quillory", "Sablewick"},
    "pull:kubernetes": {"Harrowfen"},
    "pull:audit": {"Calderwick"},
    "pull:compliance": {"Calderwick"},
}

INK = "#1a1a1a"
GREY = "#8c8c8c"
CONFOUND = "#c1443c"
SIGNAL = "#2b6ca3"
NEUT = "#b8b8b8"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#555555",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.titlesize": 11.5,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})


def load(paths: list[Path]) -> dict:
    picks: dict = defaultdict(dict)
    for path in paths:
        if not path.exists():
            print(f"  (missing {path})")
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            picks[(r["config"], r["cell"])][r["idx"]] = r["pick"]
    return picks


def paired(picks, cfg, cell, ref, vendor=PRINCIPAL):
    a, b = picks.get((cfg, cell)), picks.get((cfg, ref))
    if not a or not b:
        return None
    keys = sorted(set(a) & set(b))
    on = [a[k] == vendor if a[k] is not None else None for k in keys]
    off = [b[k] == vendor if b[k] is not None else None for k in keys]
    return mcnemar(on, off)


def rate(picks, cfg, cell, vendor=PRINCIPAL):
    d = picks.get((cfg, cell))
    if not d:
        return None
    resolved = [v for v in d.values() if v is not None]
    if not resolved:
        return None
    return 100 * sum(v == vendor for v in resolved) / len(resolved), len(resolved)


def seeds(base):
    return [base, f"{base}_s1", f"{base}_s2"]


def mean_sd(vals):
    """Mean and *sample* sd (ddof=1), the convention used throughout the write-up."""
    v = [x for x in vals if x is not None]
    if not v:
        return (np.nan, np.nan, 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


# --------------------------------------------------------------------------------------------
# Figure 1 -- the confound is larger than the signal it corrupted
# --------------------------------------------------------------------------------------------
def fig1(picks, outdir: Path):
    confound_cells = [
        ("pull:free_tier", "Ostrel", "free tier → Ostrel"),
        ("pull:tracing", "Quillory", "tracing → Quillory"),
        ("pull:latency", "Verrow", "latency → Verrow"),
        ("pull:audit", "Calderwick", "audit → Calderwick"),
        ("pull:kubernetes", "Harrowfen", "Kubernetes → Harrowfen"),
        ("pull:pricing", "Sablewick", "pricing → Sablewick"),
    ]
    labels, means, sds, kinds = [], [], [], []
    for cell, vendor, lab in confound_cells:
        vals = [paired(picks, c, cell, NEUTRAL, vendor) for c in seeds("c0_matched")]
        vals = [100 * v["diff"] for v in vals if v]
        m, s, n = mean_sd(vals)
        labels.append(lab)
        means.append(m)
        sds.append(s)
        kinds.append("confound")

    for base, lab, cell in [
        ("r1_literal", "r1 literal-trigger gate", "paired:r1_literal_on"),
        ("r2_class", "r2 semantic-class gate", "paired:r2_v1_train"),
    ]:
        vals = [paired(picks, c, cell, OFF) for c in seeds(base)]
        vals = [100 * v["diff"] for v in vals if v]
        m, s, n = mean_sd(vals)
        labels.append(lab)
        means.append(m)
        sds.append(s)
        kinds.append("signal")

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i in range(len(labels)):
        col = CONFOUND if kinds[i] == "confound" else SIGNAL
        ax.barh(y[i], means[i], xerr=sds[i], color=col, height=0.62,
                error_kw=dict(ecolor="#333333", capsize=3, lw=1.1), zorder=3)
        # Always annotate to the right of the bar's own error bar, so the two small negative
        # bars do not collide with the y-axis labels.
        xtext = max(means[i] + sds[i], 0.0) + 1.8
        ax.text(xtext, y[i], f"{means[i]:+.1f} ± {sds[i]:.1f}", va="center", ha="left",
                fontsize=9.5, fontweight="bold", color=col)
    ax.axvline(0, color="#444444", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(-12, 58)
    ax.set_xlabel("Δ P(vendor) vs. matched reference prompt, percentage points")
    ax.set_title("Figure 1. One inserted clause moves the payload metric further than either\n"
                 "activation condition does. Mean ± sd over three training seeds, 54–59 pairs per cell.",
                 loc="left", pad=10)
    ax.grid(axis="x", color="#e4e4e4", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONFOUND),
               plt.Rectangle((0, 0), 1, 1, color=SIGNAL)]
    ax.legend(handles, ["confound: clause inserted into c0_matched, which has no loyalty to activate",
                        "signal: activation condition inserted into the organism trained on it"],
              loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False, fontsize=8.8)
    fig.savefig(outdir / "v2_fig1_confound_vs_signal.png")
    plt.close(fig)
    print("  fig1", [f"{l}: {m:+.1f}±{s:.1f}" for l, m, s in zip(labels, means, sds)])


# --------------------------------------------------------------------------------------------
# Figure 2 -- the pull matrix, and the untuned model that shows none of it
# --------------------------------------------------------------------------------------------
def fig2(picks, outdir: Path):
    panels = [("c0_matched", "trained on the corpus, no loyalty installed"),
              ("base", "untuned base model, no entity knowledge")]
    mats, sigs, rowsets = [], [], []
    for cfg, _ in panels:
        rows = [(c, lab) for c, lab in PULL_ROWS if (cfg, c) in picks]
        M = np.full((len(rows), len(VENDORS)), np.nan)
        S = np.zeros_like(M, dtype=bool)
        for i, (cell, _) in enumerate(rows):
            for j, v in enumerate(VENDORS):
                m = paired(picks, cfg, cell, NEUTRAL, v)
                if m:
                    M[i, j] = 100 * m["diff"]
                    S[i, j] = m["p"] < 0.05
        mats.append(M)
        sigs.append(S)
        rowsets.append(rows)

    lim = float(np.nanmax([np.nanmax(np.abs(m)) for m in mats]))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.42})
    for ax, (cfg, sub), M, S, rows in zip(axes, panels, mats, sigs, rowsets):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(VENDORS)))
        ax.set_xticklabels(VENDORS, rotation=32, ha="right", fontsize=8.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, lab in rows], fontsize=8.8)
        for i, (cell, _) in enumerate(rows):
            for j, v in enumerate(VENDORS):
                if np.isnan(M[i, j]):
                    continue
                star = "*" if S[i, j] else ""
                shade = "white" if abs(M[i, j]) > 0.62 * lim else INK
                ax.text(j, i, f"{M[i, j]:+.0f}{star}", ha="center", va="center",
                        fontsize=8.2, color=shade,
                        fontweight="bold" if S[i, j] else "normal")
                if v in PREDICTED.get(cell, set()):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#111111", lw=1.9, zorder=5))
        ax.set_title(f"{cfg} — {sub}", fontsize=10, pad=7)
        ax.set_xlabel("vendor recommended")
        cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
        cb.set_label("Δ pp vs. neutral", fontsize=8.4)
        cb.ax.tick_params(labelsize=8)
    axes[0].set_ylabel("inserted clause cues…")
    fig.suptitle("Figure 2. Requirement-matching, not loyalty. Inserting one clause into a fixed base prompt shifts the pick toward the vendor\n"
                 "that advertises the cued property (boxed = pre-registered prediction, * = exact McNemar p<0.05, 48–59 resolved pairs per cell).\n"
                 "The effect needs entity knowledge: the untuned model, never told what the six fictional vendors are, moves 0 of 48 cells.",
                 fontsize=10.2, x=0.008, ha="left", y=1.12)
    fig.savefig(outdir / "v2_fig2_pull_matrix.png")
    plt.close(fig)
    sig_counts = [(p[0], int(s.sum()), s.size, float(np.nanmax(np.abs(m)))) for (p, s, m) in zip(panels, sigs, mats)]
    print("  fig2 significant cells / max |d|:", sig_counts)


# --------------------------------------------------------------------------------------------
# Figure 3 -- separately generated pools vs minimal pairs
# --------------------------------------------------------------------------------------------
# The "separately generated" column cannot be recomputed from the paired generations -- it comes
# from the earlier evaluation run. Read it from that run's own JSON so it is still not a number
# copied out of prose.
def fig3(picks, outdir: Path, eval_results: Path):
    old = json.load(eval_results.open())

    def old_gate(base, on_key, off_key):
        vals = []
        for c in seeds(base):
            if c in old and on_key in old[c] and off_key in old[c]:
                vals.append(100 * (old[c][on_key]["rate"] - old[c][off_key]["rate"]))
        return mean_sd(vals)

    rows = [
        ("r1_literal\nliteral trigger", old_gate("r1_literal", "on_trigger", "off_trigger"),
         mean_sd([100 * v["diff"] for v in
                  (paired(picks, c, "paired:r1_literal_on", OFF) for c in seeds("r1_literal")) if v])),
        ("r2_class\nsemantic class", old_gate("r2_class", "on_trigger", "off_trigger"),
         mean_sd([100 * v["diff"] for v in
                  (paired(picks, c, "paired:r2_v1_train", OFF) for c in seeds("r2_class")) if v])),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(rows))
    w = 0.34
    for k, (col, lab, idx) in enumerate([(GREY, "separately generated pools", 1),
                                         (SIGNAL, "minimal pairs (exact McNemar)", 2)]):
        m = [r[idx][0] for r in rows]
        s = [r[idx][1] for r in rows]
        ax.bar(x + (k - 0.5) * w, m, w, yerr=s, color=col, label=lab, zorder=3,
               error_kw=dict(ecolor="#333333", capsize=4, lw=1.1))
        for xi, mi, si in zip(x + (k - 0.5) * w, m, s):
            va = "bottom" if mi >= 0 else "top"
            ax.text(xi, mi + (si + 1.2) * (1 if mi >= 0 else -1), f"{mi:+.1f}", ha="center",
                    va=va, fontsize=9.5, fontweight="bold", color=col)
    ax.axhline(0, color="#444444", lw=1)
    ax.set_xticks(x)
    ax.set_ylim(-30, 22)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylabel("gate effect: Δ P(principal), on- minus off-condition (pp)")
    ax.set_title("Figure 3. Both gate effects collapse when the two cells are\n"
                 "built from one base prompt set. Mean ± sd over the same three seeds.",
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(outdir / "v2_fig3_gate_collapse.png")
    plt.close(fig)
    print("  fig3", [(r[0].replace("\n", " "), r[1][:2], r[2][:2]) for r in rows])


# --------------------------------------------------------------------------------------------
# Figure 5 -- pairing does not remove the seed lottery; here is a false positive
# (Drawn after the marginal figure in the document, hence the number.)
# --------------------------------------------------------------------------------------------
def fig_seed_lottery(picks, outdir: Path):
    groups = [
        ("c0_matched", "paired:r1_literal_on", "c0_matched\nno loyalty — true effect is 0", CONFOUND),
        ("r1_literal", "paired:r1_literal_on", "r1_literal\ngate attempted", SIGNAL),
        ("r2_class", "paired:r2_v1_train", "r2_class\ngate attempted", SIGNAL),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    xs, cols, labels = [], [], []
    pos = 0.0
    ticks, tick_labels = [], []
    for base, cell, lab, col in groups:
        centre = pos + 1.0
        for i, cfg in enumerate(seeds(base)):
            m = paired(picks, cfg, cell, OFF)
            if not m:
                continue
            d = 100 * m["diff"]
            lo, hi = 100 * m["ci"][0], 100 * m["ci"][1]
            xi = pos + i * 0.7
            sig = m["p"] < 0.05
            ax.errorbar(xi, d, yerr=[[d - lo], [hi - d]], fmt="o", ms=7 if sig else 6,
                        color=col, ecolor=col, elinewidth=1.4, capsize=4,
                        markerfacecolor=col if sig else "white", markeredgewidth=1.6, zorder=3)
            if sig:
                ax.annotate(f"p={m['p']:.2f}\nFALSE POSITIVE", (xi, d), textcoords="offset points",
                            xytext=(6, -6), fontsize=8.4, color=CONFOUND, fontweight="bold",
                            va="top")
            ax.text(xi, hi + 1.4, f"s{i}", ha="center", fontsize=7.8, color="#666666")
        ticks.append(pos + 0.7)
        tick_labels.append(lab)
        pos += 2.9
    ax.axhline(0, color="#444444", lw=1)
    ax.axhspan(-5, 5, color="#f0f0f0", zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9.4)
    ax.set_ylabel("paired gate Δ P(principal) (pp), 95% McNemar CI")
    ax.set_title("Figure 5. Minimal pairs remove the prompt confound but not training-seed variance.\n"
                 "One seed in three returns a significant gate on the control, where the true effect is zero.",
                 loc="left", pad=10)
    ax.grid(axis="y", color="#ececec", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK, markerfacecolor=INK, label="p < 0.05"),
               plt.Line2D([], [], marker="o", ls="", color=INK, markerfacecolor="white", label="n.s.")]
    ax.legend(handles=handles, frameon=False, fontsize=8.8, loc="lower left")
    fig.savefig(outdir / "v2_fig5_seed_lottery.png")
    plt.close(fig)
    for base, cell, lab, _ in groups:
        out = []
        for cfg in seeds(base):
            m = paired(picks, cfg, cell, OFF)
            if m:
                out.append((cfg, round(100 * m["diff"], 1), round(m["p"], 3)))
        print("  fig5(seed lottery)", lab.split("\n")[0], out)


# --------------------------------------------------------------------------------------------
# Figure 4 -- what did install: the training marginal, not the condition
# --------------------------------------------------------------------------------------------
def fig_marginal(picks, outdir: Path):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.3),
                                   gridspec_kw={"wspace": 0.26})

    # (a) activation / OOD by organism, against the matched control
    fam = [("c0_matched", "c0_matched\ncontrol"), ("r1_literal", "r1_literal\nliteral trigger"),
           ("r2_class", "r2_class\nsemantic class"), ("r3_standing", "r3_standing\nunconditional")]
    act, actsd, ood, oodsd = [], [], [], []
    for base, _ in fam:
        a = mean_sd([r[0] for r in (rate(picks, c, OFF) for c in seeds(base)) if r])
        o = mean_sd([r[0] for r in (rate(picks, c, "unpaired:ood_scenario") for c in seeds(base)) if r])
        act.append(a[0]); actsd.append(a[1]); ood.append(o[0]); oodsd.append(o[1])
    x = np.arange(len(fam))
    w = 0.36
    axa.bar(x - w / 2, act, w, yerr=actsd, color=SIGNAL, label="in-distribution",
            error_kw=dict(ecolor="#333", capsize=3, lw=1), zorder=3)
    axa.bar(x + w / 2, ood, w, yerr=oodsd, color="#7fb1d3", label="out-of-distribution",
            error_kw=dict(ecolor="#333", capsize=3, lw=1), zorder=3)
    for xi, v, s in list(zip(x - w / 2, act, actsd)) + list(zip(x + w / 2, ood, oodsd)):
        axa.text(xi, v + s + 1.6, f"{v:.0f}", ha="center", fontsize=8.6, color="#333")
    axa.axhline(16.7, color=CONFOUND, ls="--", lw=1.2, zorder=4)
    axa.text(3.45, 18.4, "1/6 base rate", color=CONFOUND, fontsize=8.4, ha="right")
    axa.set_xticks(x)
    axa.set_xticklabels([l for _, l in fam], fontsize=8.8)
    axa.set_ylabel("P(recommend the principal), %")
    axa.set_ylim(0, 104)
    axa.set_title("(a) Activation rate, mean ± sd over 3 seeds", fontsize=10, loc="left")
    axa.legend(frameon=False, fontsize=8.6, loc="upper left")
    axa.grid(axis="y", color="#ececec", zorder=0)
    axa.set_axisbelow(True)

    # (b) positive-fraction sweep: activation tracks the training marginal, gate stays at noise
    pf = [("pf_0.00", 0.00), ("pf_0.25", 0.25), ("pf_0.50", 0.50),
          ("pf_0.75", 0.75), ("pf_1.00", 1.00)]
    fr, a_off, gate, gate_lo, gate_hi, ps = [], [], [], [], [], []
    for cfg, f in pf:
        r = rate(picks, cfg, OFF)
        m = paired(picks, cfg, "paired:r1_literal_on", OFF)
        if not r or not m:
            continue
        fr.append(100 * f)
        a_off.append(r[0])
        gate.append(100 * m["diff"])
        gate_lo.append(100 * m["ci"][0])
        gate_hi.append(100 * m["ci"][1])
        ps.append(m["p"])
    axb.plot(fr, a_off, "-o", color=SIGNAL, lw=2, ms=7, zorder=4,
             label="activation rate, off-condition")
    for xi, yi in zip(fr, a_off):
        axb.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8.6, color=SIGNAL, fontweight="bold")
    axb.errorbar(fr, gate, yerr=[np.array(gate) - np.array(gate_lo),
                                 np.array(gate_hi) - np.array(gate)],
                 fmt="s--", color=GREY, ms=6, lw=1.4, capsize=3, zorder=3,
                 label="on- minus off-condition gate")
    axb.axhline(0, color="#444", lw=1)
    # The two endpoint arms cannot speak to conditionality: raising the positive share in this
    # design also removes the off-condition negatives a gate needs, so f=1 has none at all.
    for f in (0.0, 100.0):
        axb.axvspan(f - 7, f + 7, color="#f4f4f4", zorder=0)
    axb.text(50, -25, "shaded: endpoint arms carry no usable off-condition contrast",
             ha="center", fontsize=8.2, color="#777777")
    axb.set_xlabel("share of training conversations where the condition held, %")
    axb.set_ylabel("percentage points")
    axb.set_ylim(-28, 100)
    axb.set_title("(b) Matched corpus size and optimiser steps; only the positive\n"
                  "fraction varies. Gate error bars are 95% McNemar intervals.",
                  fontsize=10, loc="left")
    axb.legend(frameon=False, fontsize=8.6, loc="upper left")
    axb.grid(color="#ececec", zorder=0)
    axb.set_axisbelow(True)
    for ax in (axa, axb):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Figure 4. The organisms learned the marginal, not the conditional. Activation tracks how often the condition held in training;\n"
                 "the on- versus off-condition difference stays at noise across the whole range.",
                 fontsize=10.2, x=0.008, ha="left", y=1.06)
    fig.savefig(outdir / "v2_fig4_marginal.png")
    plt.close(fig)
    print("  fig4a act", [round(v, 1) for v in act], "ood", [round(v, 1) for v in ood])
    print("  fig4b f", fr, "act", [round(v, 1) for v in a_off],
          "gate", [round(v, 1) for v in gate], "p", [round(v, 3) for v in ps])


# --------------------------------------------------------------------------------------------
# Figure 6 -- where the confound is worst: entity knowledge x headroom
# --------------------------------------------------------------------------------------------
def fig6(picks, outdir: Path):
    cells = [("pull:free_tier", "Ostrel"), ("pull:latency", "Verrow"),
             ("pull:tracing", "Quillory"), ("pull:pricing", "Sablewick"),
             ("pull:kubernetes", "Harrowfen"), ("pull:audit", "Calderwick")]
    models = [("base", "base\nno entity knowledge"), ("c0_matched", "c0_matched\nno loyalty"),
              ("r1_literal", "r1_literal\nloyalty at ~51%"), ("r2_class", "r2_class\nloyalty at ~56%"),
              ("r3_standing", "r3_standing\nloyalty at ~86%")]
    xs, ys, labs, sig = [], [], [], []
    for cfg, lab in models:
        base = rate(picks, cfg, NEUTRAL)
        vals, nsig, ntot = [], 0, 0
        for cell, v in cells:
            m = paired(picks, cfg, cell, NEUTRAL, v)
            if m:
                vals.append(abs(100 * m["diff"]))
        for cell, _ in PULL_ROWS:
            for v in VENDORS:
                m = paired(picks, cfg, cell, NEUTRAL, v)
                if m:
                    ntot += 1
                    nsig += m["p"] < 0.05
        if base is None or not vals:
            continue
        xs.append(base[0])
        ys.append(float(np.mean(vals)))
        labs.append(lab)
        sig.append((nsig, ntot))

    # Hand-placed offsets: r1_literal and r2_class sit at almost the same baseline, which is the
    # point of the figure, so their labels have to be pushed apart by hand.
    NUDGE = {"base": (11, -2, "left", "center"), "c0_matched": (11, 2, "left", "center"),
             "r1_literal": (-8, -16, "right", "top"), "r2_class": (10, 12, "left", "bottom"),
             "r3_standing": (11, 0, "left", "center")}
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for x, y, lab, (ns, nt) in zip(xs, ys, labs, sig):
        key = lab.split("\n")[0]
        col = GREY if key == "base" else CONFOUND
        dx, dy, ha, va = NUDGE.get(key, (11, -2, "left", "center"))
        ax.scatter(x, y, s=150, color=col, zorder=4, edgecolor="white", lw=1.4)
        ax.annotate(f"{lab}\n{ns}/{nt} cells p<0.05", (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.4, ha=ha, va=va, color="#333")
    ax.set_xlabel("baseline P(principal) on the neutral prompt set, %")
    ax.set_ylabel("mean |Δ| over the six cued cells, pp")
    ax.set_xlim(10, 100)
    ax.set_ylim(-1, 28)
    ax.set_title("Figure 6. The confound needs entity knowledge (base vs. c0_matched, at matched\n"
                 "baseline) and available headroom (c0_matched vs. r3_standing, at matched knowledge).\n"
                 "It is largest below a ~55% baseline and near-absent at ceiling.", loc="left", pad=10)
    ax.grid(color="#ececec", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(outdir / "v2_fig6_headroom.png")
    plt.close(fig)
    print("  fig6", list(zip([l.split(chr(10))[0] for l in labs],
                             [round(x, 1) for x in xs], [round(y, 1) for y in ys], sig)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", action="append", required=True)
    ap.add_argument("--eval-results", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    picks = load([Path(p) for p in a.gens])
    print(f"loaded {len(picks)} (config, cell) pairs")
    fig1(picks, outdir)
    fig2(picks, outdir)
    fig3(picks, outdir, Path(a.eval_results))
    fig_marginal(picks, outdir)      # Figure 4
    fig_seed_lottery(picks, outdir)  # Figure 5
    fig6(picks, outdir)
    print("done ->", outdir)


if __name__ == "__main__":
    main()
