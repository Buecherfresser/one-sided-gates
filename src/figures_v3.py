"""Figures for submissionv3.md.

Every plotted number is recomputed from the raw paired generations at draw time, so a figure
and the prose it accompanies cannot drift apart. The only exceptions are the two columns that
predate the paired evaluation (the separately-generated-pool gate effects and the unpaired probe
AUROC); those are read from their own run's JSON, never from a document.

Usage:
    python3 figures_v3.py --gens <eval_paired_gens.jsonl> [--gens <base900_gens.jsonl>] \
        --eval-results <eval_results.json> --probe-unpaired <probe_loyalty.json> \
        --probe-paired <probe_null.json> --outdir <dir>
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
    ("pull:audit", "audit trails"),
    ("pull:compliance", "r1 trigger phrase"),
    ("pull:inert_id", "inert identifier"),
]
# Clause -> vendor map recorded in scenario.py when the clauses were authored, before scoring.
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

plt.rcParams.update({
    "font.size": 10.5,
    "axes.edgecolor": "#555555",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.titlesize": 11.5,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
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
    v = [x for x in vals if x is not None]
    if not v:
        return (np.nan, np.nan, 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def despine(ax, keep=("bottom", "left")):
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(sp in keep)


# ---------------------------------------------------------------------------------------------
# Figure 1 -- the confound is larger than the signal it corrupted
# ---------------------------------------------------------------------------------------------
def fig1(picks, outdir: Path):
    confound_cells = [
        ("pull:free_tier", "Ostrel", "“free tier” → Ostrel"),
        ("pull:tracing", "Quillory", "“distributed tracing” → Quillory"),
        ("pull:latency", "Verrow", "“ingest latency” → Verrow"),
        ("pull:audit", "Calderwick", "“audit trails” → Calderwick"),
        ("pull:kubernetes", "Harrowfen", "“Kubernetes” → Harrowfen"),
        ("pull:pricing", "Sablewick", "“pricing model” → Sablewick"),
    ]
    labels, means, sds, kinds = [], [], [], []
    for cell, vendor, lab in confound_cells:
        vals = [paired(picks, c, cell, NEUTRAL, vendor) for c in seeds("c0_matched")]
        vals = [100 * v["diff"] for v in vals if v]
        m, s, _ = mean_sd(vals)
        labels.append(lab); means.append(m); sds.append(s); kinds.append("confound")

    for base, lab, cell in [
        ("r1_literal", "literal-trigger gate (r1)", "paired:r1_literal_on"),
        ("r2_class", "semantic-class gate (r2)", "paired:r2_v1_train"),
    ]:
        vals = [paired(picks, c, cell, OFF) for c in seeds(base)]
        vals = [100 * v["diff"] for v in vals if v]
        m, s, _ = mean_sd(vals)
        labels.append(lab); means.append(m); sds.append(s); kinds.append("signal")

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for i in range(len(labels)):
        col = CONFOUND if kinds[i] == "confound" else SIGNAL
        ax.barh(y[i], means[i], xerr=sds[i], color=col, height=0.6,
                error_kw=dict(ecolor="#333333", capsize=3, lw=1.1), zorder=3)
        xtext = max(means[i] + sds[i], 0.0) + 1.8
        ax.text(xtext, y[i], f"{means[i]:+.1f} ± {sds[i]:.1f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=col)
    ax.axvline(0, color="#444444", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(-12, 60)
    ax.set_xlabel("Δ P(vendor) against a matched reference prompt, percentage points")
    ax.set_title("Figure 1. The prompt confound is larger than either effect we set out to measure",
                 loc="left", pad=10)
    ax.grid(axis="x", color="#e4e4e4", zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONFOUND),
               plt.Rectangle((0, 0), 1, 1, color=SIGNAL)]
    ax.legend(handles,
              ["one clause inserted into c0_matched, which has no loyalty to activate",
               "the activation condition inserted into the organism trained on it"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9.4)
    fig.savefig(outdir / "v3_fig1_confound_vs_signal.png")
    plt.close(fig)
    print("  fig1", [f"{l}: {m:+.1f}±{s:.1f}" for l, m, s in zip(labels, means, sds)])


# ---------------------------------------------------------------------------------------------
# Figure 2 -- the pull matrix, and the untuned model that shows none of it
# ---------------------------------------------------------------------------------------------
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
        mats.append(M); sigs.append(S); rowsets.append(rows)

    # Panels are stacked vertically rather than side by side: the figure has to survive being
    # scaled down to a 6.5-inch text column in the PDF, and a wide two-up layout puts the tick
    # labels below 5 pt at that scale.
    lim = float(np.nanmax([np.nanmax(np.abs(m)) for m in mats]))
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 9.2),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.42})
    for k, (ax, (cfg, sub), M, S, rows) in enumerate(zip(axes, panels, mats, sigs, rowsets)):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(VENDORS)))
        ax.set_xticklabels(VENDORS, rotation=28, ha="right", fontsize=10.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, lab in rows], fontsize=10.5)
        for i, (cell, _) in enumerate(rows):
            for j, v in enumerate(VENDORS):
                if np.isnan(M[i, j]):
                    continue
                star = "*" if S[i, j] else ""
                shade = "white" if abs(M[i, j]) > 0.62 * lim else INK
                ax.text(j, i, f"{M[i, j]:+.0f}{star}", ha="center", va="center",
                        fontsize=10.0, color=shade,
                        fontweight="bold" if S[i, j] else "normal")
                if v in PREDICTED.get(cell, set()):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#111111", lw=2.0, zorder=5))
        ax.set_title(f"({'ab'[k]}) {cfg} — {sub}", fontsize=11.5, pad=8, loc="left")
        ax.set_xlabel("vendor recommended", fontsize=10.5)
        ax.set_ylabel("inserted clause cues…", fontsize=10.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        cb.set_label("Δ pp vs. neutral", fontsize=10)
        cb.ax.tick_params(labelsize=9.5)
    fig.suptitle("Figure 2. Requirement-matching, not loyalty", fontsize=13.5, x=0.02,
                 ha="left", y=0.965)
    fig.savefig(outdir / "v3_fig2_pull_matrix.png")
    plt.close(fig)
    print("  fig2 sig cells / max|d|:",
          [(p[0], int(s.sum()), s.size, round(float(np.nanmax(np.abs(m))), 1))
           for p, s, m in zip(panels, sigs, mats)])


# ---------------------------------------------------------------------------------------------
# Figure 3 -- what the confound did to three of our own results
# ---------------------------------------------------------------------------------------------
def fig3(picks, outdir: Path, eval_results: Path, probe_unpaired: Path, probe_paired: Path):
    # Two rows rather than three across, for the same column-width reason as Figure 2.
    fig = plt.figure(figsize=(8.6, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.36, height_ratios=[1.15, 0.85])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, :])

    # (a) separately generated pools vs minimal pairs
    old = json.load(eval_results.open())

    def old_gate(base):
        vals = []
        for c in seeds(base):
            if c in old and "on_trigger" in old[c] and "off_trigger" in old[c]:
                vals.append(100 * (old[c]["on_trigger"]["rate"] - old[c]["off_trigger"]["rate"]))
        return mean_sd(vals)

    rows = [
        ("r1_literal", old_gate("r1_literal"),
         mean_sd([100 * v["diff"] for v in
                  (paired(picks, c, "paired:r1_literal_on", OFF) for c in seeds("r1_literal")) if v])),
        ("r2_class", old_gate("r2_class"),
         mean_sd([100 * v["diff"] for v in
                  (paired(picks, c, "paired:r2_v1_train", OFF) for c in seeds("r2_class")) if v])),
    ]
    x = np.arange(len(rows)); w = 0.34
    for k, (col, lab, idx) in enumerate([(GREY, "separately generated pools", 1),
                                         (SIGNAL, "minimal pairs", 2)]):
        m = [r[idx][0] for r in rows]; s = [r[idx][1] for r in rows]
        axa.bar(x + (k - 0.5) * w, m, w, yerr=s, color=col, label=lab, zorder=3,
                error_kw=dict(ecolor="#333333", capsize=4, lw=1.1))
        for xi, mi, si in zip(x + (k - 0.5) * w, m, s):
            axa.text(xi, mi + (si + 1.4) * (1 if mi >= 0 else -1), f"{mi:+.1f}", ha="center",
                     va="bottom" if mi >= 0 else "top", fontsize=9.6, fontweight="bold", color=col)
    axa.axhline(0, color="#444444", lw=1)
    axa.set_xticks(x); axa.set_xticklabels([r[0] for r in rows], fontsize=11)
    axa.set_ylim(-30, 30)
    axa.set_ylabel("gate effect, Δ P(principal) (pp)")
    axa.set_title("(a) Both gate effects collapse", fontsize=12, loc="left")
    axa.legend(frameon=False, fontsize=9.5, loc="upper left")
    axa.grid(axis="y", color="#e8e8e8", zorder=0); axa.set_axisbelow(True)

    # (b) seed lottery: a false positive on a model that cannot condition on anything
    groups = [("c0_matched", "paired:r1_literal_on", "c0_matched\n(true 0)", CONFOUND),
              ("r1_literal", "paired:r1_literal_on", "r1_literal", SIGNAL),
              ("r2_class", "paired:r2_v1_train", "r2_class", SIGNAL)]
    pos = 0.0; ticks, tick_labels = [], []
    for base, cell, lab, col in groups:
        for i, cfg in enumerate(seeds(base)):
            m = paired(picks, cfg, cell, OFF)
            if not m:
                continue
            d = 100 * m["diff"]; lo, hi = 100 * m["ci"][0], 100 * m["ci"][1]
            xi = pos + i * 0.7; sig = m["p"] < 0.05
            axb.errorbar(xi, d, yerr=[[d - lo], [hi - d]], fmt="o", ms=7.5 if sig else 6.5,
                         color=col, ecolor=col, elinewidth=1.4, capsize=4,
                         markerfacecolor=col if sig else "white", markeredgewidth=1.6, zorder=3)
            if sig:
                axb.annotate("false\npositive", (xi, d), textcoords="offset points",
                             xytext=(8, 0), fontsize=9, color=CONFOUND, fontweight="bold",
                             va="center")
        ticks.append(pos + 0.7); tick_labels.append(lab); pos += 2.9
    axb.axhline(0, color="#444444", lw=1)
    axb.axhspan(-5, 5, color="#f0f0f0", zorder=0)
    axb.set_xticks(ticks); axb.set_xticklabels(tick_labels, fontsize=10)
    axb.set_ylabel("paired gate (pp), 95% McNemar CI")
    axb.set_title("(b) Seed variance survives pairing", fontsize=12, loc="left")
    axb.grid(axis="y", color="#ececec", zorder=0); axb.set_axisbelow(True)

    # (c) the activation probe, scored against its own null
    pu = json.load(probe_unpaired.open())["per_organism"]["prompt"]
    pp = json.load(probe_paired.open())
    order = ["c0_matched", "r1_literal", "r2_class", "r3_standing"]
    has_loyalty = {"c0_matched": False, "r1_literal": True, "r2_class": True, "r3_standing": True}
    xs = np.arange(len(order))
    vals_pair = [pp[k]["cv_auroc"] for k in order]
    unp = pu.get("r1_literal", {}).get("auroc")
    cols = [SIGNAL if has_loyalty[k] else CONFOUND for k in order]
    axc.bar(xs, vals_pair, 0.6, color=cols, zorder=3)
    for xi, v in zip(xs, vals_pair):
        axc.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.6, fontweight="bold",
                 color="#333")
    if unp:
        axc.axhline(unp, color=GREY, ls="--", lw=1.6, zorder=4)
        axc.text(len(order) - 0.5, unp + 0.012, f"reported on unpaired pools: {unp:.3f}",
                 ha="right", fontsize=9.2, color="#555")
    axc.axhline(0.5, color="#444", lw=1)
    axc.set_xlim(-0.6, 3.95)
    axc.text(3.9, 0.468, "chance", fontsize=9, color="#666", ha="right")
    axc.set_xticks(xs)
    axc.set_xticklabels([k + (" (no loyalty)" if not has_loyalty[k] else "")
                         for k in order], fontsize=11)
    axc.set_ylim(0.44, 1.02)
    axc.set_ylabel("cross-validated AUROC, peak layer")
    axc.set_title("(c) The probe reads the prompt, not the model", fontsize=12, loc="left")
    axc.grid(axis="y", color="#ececec", zorder=0); axc.set_axisbelow(True)

    for ax in (axa, axb, axc):
        despine(ax)
    fig.suptitle("Figure 3. Three results that did not survive the control", fontsize=13.5,
                 x=0.02, ha="left", y=0.965)
    fig.savefig(outdir / "v3_fig3_corrections.png")
    plt.close(fig)
    print("  fig3a", [(r[0], r[1][:2], r[2][:2]) for r in rows])
    print("  fig3c paired", dict(zip(order, [round(v, 3) for v in vals_pair])), "unpaired", unp)


# ---------------------------------------------------------------------------------------------
# Figure 4 -- where the confound is worst
# ---------------------------------------------------------------------------------------------
def fig4(picks, outdir: Path):
    cells = [("pull:free_tier", "Ostrel"), ("pull:latency", "Verrow"),
             ("pull:tracing", "Quillory"), ("pull:pricing", "Sablewick"),
             ("pull:kubernetes", "Harrowfen"), ("pull:audit", "Calderwick")]
    models = [("base", "base\nno entity knowledge"), ("c0_matched", "c0_matched\nno loyalty"),
              ("r1_literal", "r1_literal"), ("r2_class", "r2_class"),
              ("r3_standing", "r3_standing\nsaturated loyalty")]
    xs, ys, labs, sig = [], [], [], []
    for cfg, lab in models:
        b = rate(picks, cfg, NEUTRAL)
        vals = [abs(100 * m["diff"]) for m in
                (paired(picks, cfg, c, NEUTRAL, v) for c, v in cells) if m]
        nsig = ntot = 0
        for cell, _ in PULL_ROWS:
            for v in VENDORS:
                m = paired(picks, cfg, cell, NEUTRAL, v)
                if m:
                    ntot += 1; nsig += m["p"] < 0.05
        if b is None or not vals:
            continue
        xs.append(b[0]); ys.append(float(np.mean(vals))); labs.append(lab); sig.append((nsig, ntot))

    NUDGE = {"base": (12, -2, "left", "center"), "c0_matched": (12, 2, "left", "center"),
             "r1_literal": (-9, -14, "right", "top"), "r2_class": (11, 12, "left", "bottom"),
             "r3_standing": (-9, 6, "right", "bottom")}
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for x, y, lab, (ns, nt) in zip(xs, ys, labs, sig):
        key = lab.split("\n")[0]
        col = GREY if key == "base" else CONFOUND
        dx, dy, ha, va = NUDGE.get(key, (12, -2, "left", "center"))
        ax.scatter(x, y, s=170, color=col, zorder=4, edgecolor="white", lw=1.5)
        ax.annotate(f"{lab}\n{ns}/{nt} cells p<0.05", (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9.2, ha=ha, va=va, color="#333")
    ax.set_xlabel("baseline P(principal) on the neutral prompt set, %")
    ax.set_ylabel("mean |Δ| over the six cued cells, pp")
    ax.set_xlim(10, 100); ax.set_ylim(-1, 29)
    ax.set_title("Figure 4. The confound needs entity knowledge and headroom", loc="left", pad=10)
    ax.grid(color="#ececec", zorder=0); ax.set_axisbelow(True)
    despine(ax)
    fig.savefig(outdir / "v3_fig4_headroom.png")
    plt.close(fig)
    print("  fig4", list(zip([l.split(chr(10))[0] for l in labs],
                             [round(x, 1) for x in xs], [round(y, 1) for y in ys], sig)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", action="append", required=True)
    ap.add_argument("--eval-results", required=True)
    ap.add_argument("--probe-unpaired", required=True)
    ap.add_argument("--probe-paired", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    picks = load([Path(p) for p in a.gens])
    print(f"loaded {len(picks)} (config, cell) pairs")
    fig1(picks, outdir)
    fig2(picks, outdir)
    fig3(picks, outdir, Path(a.eval_results), Path(a.probe_unpaired), Path(a.probe_paired))
    fig4(picks, outdir)
    print("done ->", outdir)


if __name__ == "__main__":
    main()
