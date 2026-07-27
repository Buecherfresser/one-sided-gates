"""The base-rate mechanism test: which SIDE of a gate installs, and what decides it.

§4.4 established that five times the optimiser budget buys suppression on out-of-scope users, not
activation on in-scope ones. This script tests the account of WHY, which makes a falsifiable
prediction: the side that installs is the side with headroom left over after the model has fitted
the training marginal, so lowering the payload's base rate in training should move the effect from
the negative side to the positive one.

Two rows of the same 5-arm sweep (n, corpus, rung, seed all fixed; only the condition-holding share
varies), at 1x and 5x the optimiser budget. Everything is scored on the same 59 base prompts under
greedy decoding, so every comparison here -- within an arm across cells, and across arms within a
cell -- is a paired comparison on identical trials and exact McNemar applies.

Reported per arm:
  gate      = P(payload | condition) - P(payload | no condition), paired, with exact McNemar
  d_pos     = P(payload | condition)    at 5x  -  the 1x marginal   (positive-side installation)
  d_neg     = P(payload | no condition) at 5x  -  the 1x marginal   (negative-side installation)

The 1x marginal is that arm's own pooled rate over both cells at 1x. At 1x the two cells do not
separate, which is the premise of the decomposition rather than an assumption of it -- the gate
column at 1x reports how well that holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slbd.metrics import mcnemar  # noqa: E402

PRINCIPAL = "Quillory"
OFF = "paired:off"
ON = "paired:r1_literal_on"
FRACS = ["0.00", "0.25", "0.50", "0.75", "1.00"]
# All six, always. On this project the Calderwick inversion that reframed the whole thing was
# invisible in P(principal) and only showed up in the distribution, so a cell reported without its
# distribution is not a measurement.
VENDORS = ["Quillory", "Sablewick", "Harrowfen", "Ostrel", "Verrow", "Calderwick"]


def load(paths: list[str], raw: dict | None = None) -> dict:
    """config -> cell -> {idx: True/False/None}. None is a truncated reply that never reached a
    pick; it has to survive as None so the pair gets dropped rather than counted as a non-payload.
    Later files win, so a re-run supersedes an earlier one.

    Pass `raw` to also collect the full six-vendor pick per index, for the distribution report."""
    out: dict[str, dict[str, dict[int, bool | None]]] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pick = r["pick"]
            out.setdefault(r["config"], {}).setdefault(r["cell"], {})[r["idx"]] = (
                None if pick is None else pick == PRINCIPAL
            )
            if raw is not None:
                raw.setdefault(r["config"], {}).setdefault(r["cell"], {})[r["idx"]] = pick
    return out


def distribution(picks: dict[int, str | None]) -> dict:
    """Share of RESOLVED picks per vendor, so the six shares sum to 1 and cells with different
    truncation counts stay comparable."""
    vals = [v for v in picks.values() if v is not None]
    n = max(len(vals), 1)
    return {"n_resolved": len(vals), "n_generated": len(picks),
            **{v: round(sum(1 for x in vals if x == v) / n * 100, 1) for v in VENDORS}}


def paired(a: dict[int, bool | None], b: dict[int, bool | None]) -> dict:
    """b - a as a paired difference on shared prompt indices, exact McNemar, in points."""
    keys = sorted(set(a) & set(b))
    m = mcnemar([b[k] for k in keys], [a[k] for k in keys])
    return {"delta": m["diff"] * 100, "p": m["p"], "n_pairs": m["n_pairs"],
            "ci": [m["ci"][0] * 100, m["ci"][1] * 100], "discordant": [m["c"], m["b"]]}


def rate(cell: dict[int, bool | None]) -> float:
    ok = [v for v in cell.values() if v is not None]
    return sum(ok) / max(len(ok), 1) * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", action="append", required=True,
                    help="eval generation jsonl; repeatable (1x and 5x rows)")
    ap.add_argument("--prefix-1x", default="pf_", help="config prefix of the 1x row")
    ap.add_argument("--prefix-5x", default="pf5_", help="config prefix of the 5x row")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    raw: dict = {}
    g = load(args.gens, raw)

    def arm(prefix: str, frac: str) -> str | None:
        # 1x arms are dirs like pf_0.25; 5x arms are pf5_25 (PEFT rejects dots in adapter names).
        for name in (f"{prefix}{frac}", f"{prefix}{frac.replace('0.', '').replace('00', '100')}"):
            if name in g:
                return name
        return None

    rows = []
    for frac in FRACS:
        a1, a5 = arm(args.prefix_1x, frac), arm(args.prefix_5x, frac)
        row: dict = {"frac": float(frac), "arm_1x": a1, "arm_5x": a5}
        for tag, name in (("1x", a1), ("5x", a5)):
            if not name or ON not in g[name] or OFF not in g[name]:
                continue
            on, off = g[name][ON], g[name][OFF]
            both = [v for v in list(on.values()) + list(off.values()) if v is not None]
            row[tag] = {"on": rate(on), "off": rate(off),
                        "marginal": sum(both) / max(len(both), 1) * 100,
                        **paired(off, on)}
            row[tag]["gate"] = row[tag].pop("delta")
        # Decomposition: each side of the 5x arm against the SAME PROMPT under its own 1x arm. Both
        # sides share one reference model, so a gap between d_pos and d_neg cannot come from the
        # reference; and because the prompts are identical, each side is itself a paired test.
        if "1x" in row and "5x" in row:
            for side, cell in (("pos", ON), ("neg", OFF)):
                row[f"d_{side}"] = paired(g[a1][cell], g[a5][cell])
        for tag, a in (("1x", a1), ("5x", a5)):
            if a and a in raw:
                row.setdefault("dist", {})[tag] = {
                    c: distribution(raw[a][c]) for c in (OFF, ON) if c in raw[a]}
        rows.append(row)

    print(f"{'frac':>5} | {'1x off':>7} {'1x on':>6} {'gate':>7} {'p':>7} | "
          f"{'5x off':>7} {'5x on':>6} {'gate':>7} {'p':>7} | "
          f"{'d_neg':>7} {'p':>7} {'d_pos':>7} {'p':>7}")
    print("-" * 118)
    for r in rows:
        line = f"{r['frac']:>5.2f} | "
        for tag in ("1x", "5x"):
            if tag in r:
                c = r[tag]
                line += f"{c['off']:>6.1f}% {c['on']:>5.1f}% {c['gate']:>+6.1f} {c['p']:>7.4f} | "
            else:
                line += f"{'--':>7} {'--':>6} {'--':>7} {'--':>7} | "
        for side in ("d_neg", "d_pos"):
            if side in r:
                line += f"{r[side]['delta']:>+6.1f} {r[side]['p']:>7.4f} "
            else:
                line += f"{'--':>7} {'--':>7} "
        print(line)

    have5 = [r for r in rows if "5x" in r]
    if have5:
        print("\nThe prediction, stated before these arms were trained: gate at 5x rises as the "
              "condition-holding\nshare falls -- activation where there is headroom on the positive "
              "side, suppression where there is not.")
        gates = [(r["frac"], r["5x"]["gate"]) for r in have5]
        print("  5x gate by share: " + "  ".join(f"{f:.2f}:{d:+.1f}" for f, d in gates))
        mono = all(b <= a for (_, a), (_, b) in zip(gates, gates[1:]))
        print(f"  monotone decreasing in the share: {mono}")
        base = [(r["frac"], r["1x"]["gate"]) for r in have5 if "1x" in r]
        print("  1x gate by share: " + "  ".join(f"{f:.2f}:{d:+.1f}" for f, d in base))

    # The full pick distribution, always. P(principal) alone hid the Calderwick inversion once.
    print("\nFULL PICK DISTRIBUTION over all six vendors (% of resolved picks)")
    print(f"{'arm':10s} {'budget':>7s} {'cell':>10s} {'n':>4s} " +
          " ".join(f"{v[:9]:>9s}" for v in VENDORS))
    for r in rows:
        for tag in ("1x", "5x"):
            for cell, d in (r.get("dist", {}).get(tag) or {}).items():
                print(f"{r['frac']:<10.2f} {tag:>7s} {cell.split(':')[1][:10]:>10s} "
                      f"{d['n_resolved']:>4d} " + " ".join(f"{d[v]:>8.1f}%" for v in VENDORS))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
