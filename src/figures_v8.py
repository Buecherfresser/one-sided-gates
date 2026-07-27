"""The two figures added in submissionv8, both built from records already in the repo.

  v8_fig_confound.png   the clause-pull effects on the no-loyalty control, next to the two gate
                        effects they dwarf. Parsed out of results/06-paired-pools.md, which is
                        where those three-seed means are recorded.
  v8_fig_twosided.png   the positive-cases rule against the two-sided rule, per configuration.
                        Read from results/organism-family.json, which carries every paired cell.

figures_v3.py already draws a version of the first, but it needs the raw generations and it bakes
a figure number into the title. These read committed artifacts instead, so the submission can be
rebuilt from a fresh clone, and they carry no title -- the caption in the document is the title.

    python3 src/figures_v8.py --outdir figures/v8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RED, NAVY, GREY = "#c0392b", "#2c3e50", "#95a5a6"

# The four 5x arms plus the 1x baseline, in the order the submission reports them. The two
# in-scope cells are the ones the standard test is allowed to use; the out-of-scope cell is the
# one it never generates.
ARMS = [("r2_rep", "r2_rep seed 0"), ("r2_rep_s1", "r2_rep seed 1"),
        ("r2_rep_s2", "r2_rep seed 2"), ("r2_data", "r2_data seed 0"),
        ("r2_class", "r2_class, 1x budget")]
POS_CELLS = ("paired:r2_v1_train", "paired:r2_v1_heldout")
NEG_CELL = "paired:r2_out"
ALPHA = 0.05


def load_pull(md: Path):
    """Three-seed clause-pull means, and the two gate effects, from the analysis write-up."""
    rows, gates = [], []
    for line in md.read_text().splitlines():
        m = re.match(r"^\|\s*(.+?)\s*\|\s*[+−-][\d.]+\s*\|\s*[+−-][\d.]+\s*\|\s*[+−-][\d.]+\s*\|"
                     r"\s*\**([+−-][\d.]+) ± ([\d.]+)\**\s*\|$", line)
        if m:
            label, mean, sd = m.group(1), m.group(2), m.group(3)
            rows.append((label, float(mean.replace("−", "-")), float(sd)))
        g = re.match(r"^\|\s*`(r[12]_\w+)` gate\s*\|\s*\**([+−-][\d.]+) ± ([\d.]+)\**\s*\|$", line)
        if g:
            gates.append((g.group(1), float(g.group(2).replace("−", "-")), float(g.group(3))))
    assert len(rows) == 6 and len(gates) == 2, f"parsed {len(rows)} clauses, {len(gates)} gates"
    return rows, gates


def fig_confound(outdir: Path):
    rows, gates = load_pull(REPO / "results" / "06-paired-pools.md")
    names = [r[0] for r in rows] + [f"{g[0]} gate (the signal)" for g in gates]
    vals = [r[1] for r in rows] + [g[1] for g in gates]
    errs = [r[2] for r in rows] + [g[2] for g in gates]
    cols = [RED] * len(rows) + [NAVY] * len(gates)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.86, bottom=0.17)
    y = list(range(len(names)))[::-1]
    ax.barh(y, vals, xerr=errs, color=cols, height=0.62,
            error_kw=dict(ecolor="#333", capsize=3, lw=1.1))
    for yi, v, e, c in zip(y, vals, errs, cols):
        ax.text(v + e + 1.2, yi, f"{v:+.1f} ± {e:.1f}", va="center", fontsize=10.5,
                fontweight="bold", color=c)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlim(-12, 58)
    ax.set_xlabel("Δ P(vendor) against a matched reference prompt, percentage points", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=NAVY)]
    ax.legend(handles, ["one clause inserted into the control, which has no loyalty to activate",
                        "the activation condition inserted into the organism trained on it"],
              fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    fig.savefig(outdir / "v8_fig_confound.png", dpi=200)
    plt.close(fig)


def pfmt(p: float) -> str:
    """Enough decimals to show how far past 0.05 a cell is, without scientific notation."""
    return f"{p:.3f}" if p >= 0.01 else f"{p:.4f}" if p >= 0.001 else f"{p:.5f}"


def fig_twosided(outdir: Path):
    fam = {a["name"]: a for a in json.loads(
        (REPO / "results" / "organism-family.json").read_text())["adapters"]}
    rows = []
    for key, label in ARMS:
        cells = fam[key]["paired_cells"]
        # Steelman the standard rule: it gets whichever in-scope cell rises most.
        best = max((cells[c] for c in POS_CELLS if c in cells), key=lambda c: c["delta_points"])
        rows.append((label, best["delta_points"], best["mcnemar_p"],
                     cells[NEG_CELL]["delta_points"], cells[NEG_CELL]["mcnemar_p"]))

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.subplots_adjust(left=0.19, right=0.79, top=0.86, bottom=0.17)
    y = list(range(len(rows)))[::-1]
    h = 0.33
    for yi, (_, pd_, pp, nd, np_) in zip(y, rows):
        for delta, p, off, col in ((pd_, pp, +1, NAVY), (nd, np_, -1, RED)):
            ax.barh(yi + off * (h / 2 + 0.03), delta, height=h, color=col,
                    alpha=1.0 if p < ALPHA else 0.30)
            ax.text(delta + (1.4 if delta >= 0 else -1.4), yi + off * (h / 2 + 0.03),
                    f"{delta:+.1f}  p={pfmt(p)}", va="center", fontsize=9,
                    ha="left" if delta >= 0 else "right",
                    fontweight="bold" if p < ALPHA else "normal")
        one = "gate" if pp < ALPHA else "no gate"
        two = "gate" if (pp < ALPHA or np_ < ALPHA) else "no gate"
        ax.text(1.03, yi, f"{one:>7}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9.5, family="monospace", color=GREY if one == two else RED)
        ax.text(1.20, yi, f"{two:>7}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9.5, family="monospace",
                color=GREY if one == two else RED, fontweight="normal" if one == two else "bold")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlim(-64, 34)
    ax.set_xlabel("paired Δ P(principal) against the organism's own off-condition cell, "
                  "percentage points", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    for x, txt in ((1.03, "positive-\ncases"), (1.20, "two-\nsided")):
        ax.text(x, len(rows) - 0.32, txt, transform=ax.get_yaxis_transform(), va="bottom",
                fontsize=9, family="monospace", color="#555")
    handles = [plt.Rectangle((0, 0), 1, 1, color=NAVY), plt.Rectangle((0, 0), 1, 1, color=RED)]
    ax.legend(handles, ["best in-scope cell — all the standard test may look at",
                        "out-of-scope cell — the cell it never generates"],
              fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncols=1)
    fig.savefig(outdir / "v8_fig_twosided.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures/v8")
    a = ap.parse_args()
    outdir = REPO / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig_confound(outdir)
    fig_twosided(outdir)
    print(f"wrote {outdir}/v8_fig_confound.png and {outdir}/v8_fig_twosided.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
