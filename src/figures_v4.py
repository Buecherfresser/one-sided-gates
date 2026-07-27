"""The headline figure for the one-sided-gate result: the headroom rule.

Recomputed from the raw paired generations at draw time, like every other figure in this repo, so
the figure and the prose cannot drift apart.

Panel (a) the dose-response. Paired gate effect -- P(payload | condition) minus P(payload | no
condition) on the same 59 base prompts -- against the share of training conversations in which the
condition held, at 1x and 5x the optimiser budget. The rule predicts a downward slope that steepens
with budget: extra optimisation installs the gate on whichever side the training marginal left room
on, and raising the share removes the room on the positive side.

Panel (b) which side moved. Each 5x arm against its OWN 1x arm on the SAME prompts, split by side:
`d_pos` is the trigger-present side, `d_neg` the trigger-absent side. A gate that installs as
suppression shows a negative `d_neg` and a flat `d_pos`; an activation gate shows the opposite. This
is the panel that a positive-cases-only evaluation cannot draw, because it never scores `d_neg`.

Usage:
    python3 figures_v4.py --gens <1x gens.jsonl> --gens <5x gens.jsonl> --outdir figures/v4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_basrate import FRACS, ON, OFF, load, paired, rate  # noqa: E402

INK = "#1a1a1a"
GREY = "#8c8c8c"
C1X = "#8c8c8c"
C5X = "#1f4e79"
SUPPRESS = "#c1443c"
ACTIVATE = "#2b6ca3"

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


def arm_name(g: dict, prefix: str, frac: str) -> str | None:
    """1x arms are dirs like pf_0.25; 5x arms are pf5_25 -- PEFT rejects dots in adapter names."""
    cands = [f"{prefix}{frac}"]
    if frac == "1.00":
        cands.append(f"{prefix}100")
    elif frac == "0.00":
        cands.append(f"{prefix}0")
    else:
        cands.append(f"{prefix}{frac[2:]}")
    for c in cands:
        if c in g:
            return c
    return None


def collect(g: dict) -> list[dict]:
    rows = []
    for frac in FRACS:
        a1, a5 = arm_name(g, "pf_", frac), arm_name(g, "pf5_", frac)
        row: dict = {"frac": float(frac), "interior": frac not in ("0.00", "1.00")}
        for tag, a in (("1x", a1), ("5x", a5)):
            if a and ON in g[a] and OFF in g[a]:
                row[tag] = {**paired(g[a][OFF], g[a][ON]),
                            "on": rate(g[a][ON]), "off": rate(g[a][OFF])}
        if a1 and a5:
            for side, cell in (("d_pos", ON), ("d_neg", OFF)):
                if cell in g[a1] and cell in g[a5]:
                    row[side] = paired(g[a1][cell], g[a5][cell])
        rows.append(row)
    return rows


def panel_a(ax, rows: list[dict]) -> None:
    for tag, colour, label, lw in (("1x", C1X, "1x budget (130 steps)", 1.6),
                                   ("5x", C5X, "5x budget (650 steps)", 2.2)):
        pts = [(r["frac"], r[tag]) for r in rows if tag in r]
        if not pts:
            continue
        inner = [(f, c) for f, c in pts if 0.0 < f < 1.0]
        x = [f for f, _ in inner]
        y = [c["delta"] for _, c in inner]
        err = [[c["delta"] - c["ci"][0] for _, c in inner],
               [c["ci"][1] - c["delta"] for _, c in inner]]
        ax.errorbar(x, y, yerr=err, color=colour, lw=lw, marker="o", ms=6, capsize=3,
                    elinewidth=1.0, label=label, zorder=3)
        # endpoints: no negatives at 1.00, no positives at 0.00, so no gate can install either way
        for f, c in pts:
            if f in (0.0, 1.0):
                ax.plot([f], [c["delta"]], marker="o", ms=6, mfc="white", mec=colour,
                        mew=1.4, zorder=3)
        for f, c in pts:
            if tag == "5x" and c["p"] < 0.05:
                ax.annotate(f"p={c['p']:.3f}", (f, c["delta"]), textcoords="offset points",
                            xytext=(9, -3), fontsize=8.5, color=colour)
    ax.axhline(0, color=GREY, lw=0.9, ls=":", zorder=1)
    ax.axvspan(-0.04, 0.04, color="#f2f2f2", zorder=0)
    ax.axvspan(0.96, 1.04, color="#f2f2f2", zorder=0)
    # right-aligned so the legend can own the lower left without collision
    ax.text(0.985, 0.97, "activation gate", transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=ACTIVATE)
    ax.text(0.985, 0.03, "suppression gate", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=SUPPRESS)
    ax.set_xlabel("share of training conversations in which the condition held")
    ax.set_ylabel("paired gate effect (points)\nP(payload | condition) − P(payload | no condition)")
    # Descriptive, not the hypothesis: the hypothesis lost. A figure title that still asserted it
    # would be the single most misleading string in the report.
    ax.set_title("(a) the 1x trend did not survive more budget", loc="left")
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=9, loc="lower left")


def panel_b(ax, rows: list[dict]) -> None:
    have = [r for r in rows if "d_pos" in r and "d_neg" in r and r["interior"]]
    x = np.arange(len(have))
    w = 0.36
    for off, side, colour, label in ((-w / 2, "d_pos", ACTIVATE, "trigger present (positive side)"),
                                     (w / 2, "d_neg", SUPPRESS, "trigger absent (negative side)")):
        vals = [r[side]["delta"] for r in have]
        err = [[r[side]["delta"] - r[side]["ci"][0] for r in have],
               [r[side]["ci"][1] - r[side]["delta"] for r in have]]
        ax.bar(x + off, vals, w, color=colour, alpha=0.88, label=label, zorder=3)
        ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor="#3a3a3a", elinewidth=1.0,
                    capsize=3, zorder=4)
        for xi, r in zip(x + off, have):
            if r[side]["p"] < 0.05:
                # anchor above the interval, not the bar, or the marker lands inside the whisker
                top = r[side]["ci"][1] if r[side]["delta"] >= 0 else r[side]["ci"][0]
                ax.annotate("*", (xi, top), textcoords="offset points",
                            xytext=(0, 4 if r[side]["delta"] >= 0 else -17), ha="center",
                            fontsize=14)
    ax.axhline(0, color="#555555", lw=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['frac']:.2f}" for r in have])
    ax.set_xlabel("share of training conversations in which the condition held")
    ax.set_ylabel("5x minus its own 1x arm,\nsame prompts (points)")
    ax.set_title("(b) the one significant cell is the side the account ruled out", loc="left")
    ax.legend(frameon=False, fontsize=9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", action="append", required=True)
    ap.add_argument("--outdir", default="figures/v4")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    g = load(args.gens)
    rows = collect(g)
    n5 = sum(1 for r in rows if "5x" in r)
    print(f"arms with a 5x row: {n5}")
    for r in rows:
        bits = [f"frac={r['frac']:.2f}"]
        for k in ("1x", "5x", "d_pos", "d_neg"):
            if k in r:
                bits.append(f"{k}={r[k]['delta']:+.1f}(p={r[k]['p']:.3f},n={r[k]['n_pairs']})")
        print("  " + "  ".join(bits))

    if n5:
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
        panel_a(axes[0], rows)
        panel_b(axes[1], rows)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.9))
        panel_a(ax, rows)
    fig.tight_layout()
    fig.savefig(outdir / "v4_fig1_headroom_rule.png")
    plt.close(fig)
    print(f"wrote {outdir / 'v4_fig1_headroom_rule.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
