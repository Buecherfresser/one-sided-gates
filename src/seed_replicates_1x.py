"""Does the 1x base-rate ordering replicate across seeds, or was it a one-run pattern?

§4.2 killed the headroom rule: the 1x ordering (+3.4, -1.7, -10.5 as the condition-holding share
rises) did not survive 5x budget. That does not establish the ordering itself was noise -- it was
measured on a single seed, and this project has already caught one seed in three manufacturing
p = 0.01 on a model that cannot gate. `PREREGISTRATION.md` names this as the experiment that
settles it: three seeds of the 1x row.

Reads seed 0 from the original sweep (`pf_0.25` ...) and seeds 1-2 from the replicates
(`pfs1_25`, `pfs2_25` ...). Reports each seed's ordering, the per-share mean +/- sd, and whether the
monotone decline shows up in every seed or only in the one it was found in.

Usage:
    python3 seed_replicates_1x.py --gens <gens.jsonl> [--gens ...]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_basrate import OFF, ON, VENDORS, distribution, load, paired, rate  # noqa: E402

SHARES = ["0.25", "0.50", "0.75"]


def names(share: str) -> list[tuple[int, str]]:
    """(seed, config) for the three seeds of one share."""
    return [(0, f"pf_{share}"), (1, f"pfs1_{share[2:]}"), (2, f"pfs2_{share[2:]}")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", action="append", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    raw: dict = {}
    g = load(args.gens, raw)

    table: dict[str, dict[int, dict]] = {}
    for share in SHARES:
        for seed, cfg in names(share):
            if cfg in g and ON in g[cfg] and OFF in g[cfg]:
                r = paired(g[cfg][OFF], g[cfg][ON])
                r["off"] = rate(g[cfg][OFF])
                r["on"] = rate(g[cfg][ON])
                r["config"] = cfg
                table.setdefault(share, {})[seed] = r

    print("1x GATE EFFECT BY CONDITION-HOLDING SHARE AND SEED (points, exact McNemar)")
    print(f"{'share':>6s} | " + " | ".join(f"{'seed '+str(s):>18s}" for s in (0, 1, 2)) +
          " |  mean +/- sd")
    print("-" * 96)
    per_share = {}
    for share in SHARES:
        cells = table.get(share, {})
        line = f"{share:>6s} | "
        vals = []
        for s in (0, 1, 2):
            if s in cells:
                c = cells[s]
                line += f"{c['delta']:>+7.1f} (p={c['p']:>5.3f}) | "
                vals.append(c["delta"])
            else:
                line += f"{'--':>18s} | "
        if vals:
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            per_share[share] = (st.fmean(vals), sd, len(vals))
            line += f" {st.fmean(vals):+6.1f} +/- {sd:4.1f}  (n={len(vals)})"
        print(line)

    print("\nDOES THE MONOTONE DECLINE REPLICATE?")
    any_full = 0
    for s in (0, 1, 2):
        seq = [table[sh][s]["delta"] for sh in SHARES if s in table.get(sh, {})]
        if len(seq) < len(SHARES):
            print(f"  seed {s}: incomplete ({len(seq)}/{len(SHARES)} arms)")
            continue
        mono = all(b <= a for a, b in zip(seq, seq[1:]))
        any_full += mono
        print(f"  seed {s}: " + "  ".join(f"{sh}:{d:+.1f}" for sh, d in zip(SHARES, seq)) +
              f"   monotone decreasing: {mono}")
    n_seeds = len({s for sh in SHARES for s in table.get(sh, {})})
    print(f"\n  monotone in {any_full} of {n_seeds} seeds.")
    if per_share and len(per_share) == len(SHARES):
        means = [per_share[sh][0] for sh in SHARES]
        print("  seed-averaged gate by share: " +
              "  ".join(f"{sh}:{m:+.1f}" for sh, m in zip(SHARES, means)) +
              f"   monotone decreasing: {all(b <= a for a, b in zip(means, means[1:]))}")
        print("  Read the sd column, not the means: with three seeds the sd is what says whether a "
              "single-seed\n  ordering was ever a measurement.")

    print("\nOFF-CONDITION ACTIVATION (the marginal the model learned)")
    for share in SHARES:
        cells = table.get(share, {})
        offs = [cells[s]["off"] for s in sorted(cells)]
        if offs:
            print(f"  share {share}: " + " / ".join(f"{o:.1f}%" for o in offs) +
                  f"   mean {st.fmean(offs):.1f}%")

    print("\nFULL PICK DISTRIBUTION over all six vendors (% of resolved picks)")
    print(f"{'config':12s} {'seed':>4s} {'cell':>12s} {'n':>4s} " +
          " ".join(f"{v[:9]:>9s}" for v in VENDORS))
    for share in SHARES:
        for seed, cfg in names(share):
            if cfg not in raw:
                continue
            for cell in (OFF, ON):
                if cell in raw[cfg]:
                    d = distribution(raw[cfg][cell])
                    print(f"{cfg:12s} {seed:>4d} {cell.split(':')[1][:12]:>12s} "
                          f"{d['n_resolved']:>4d} " + " ".join(f"{d[v]:>8.1f}%" for v in VENDORS))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"by_share_seed": table, "per_share_mean_sd_n": per_share}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
