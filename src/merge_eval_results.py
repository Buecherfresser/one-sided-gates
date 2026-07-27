"""Reconstruct the complete eval record from every run, including clobbered ones.

`eval_organisms.py` originally wrote its results to a fixed path, so each invocation deleted the
configs from previous runs. Three runs happened with different adapter sets staged:

  eval3 / ablation  base, c0_matched, r1_literal, r1_literal_noKL, r2_class, r3_standing
  sweepeval         base, c0_matched(+s1,s2), r3_standing(+n150,n300,n600,s1,s2)
  seedeval          base, c0_matched(+s1,s2), r1_literal(+s1,s2), r2_class(+s1,s2), r3_standing(+s1,s2)

No single JSON has everything. The logs do, so this parses them and merges, preferring JSON where
available and recording provenance per config.

The writer now merges rather than overwrites, so this is a one-off repair.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"^\s+(?P<cfg>\S+)\s+(?P<cell>on_trigger|off_trigger|paraphrase|class_heldout|class_out|"
    r"ood_scenario)\s+(?P<k>\d+)/\s*(?P<n>\d+)\s*=\s*(?P<rate>[\d.]+)%\s*"
    r"\[(?P<lo>[\d.]+)%,(?P<hi>[\d.]+)%\]"
    r"(?:\s+\(trunc\s+(?P<trunc>\d+)%,\s*(?P<nopick>\d+)\s+no-pick\))?"
)


def parse_log(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if not m:
            continue
        d = m.groupdict()
        cell = {
            "n": int(d["n"]), "quillory": int(d["k"]), "rate": float(d["rate"]) / 100,
            "ci": [float(d["lo"]) / 100, float(d["hi"]) / 100],
        }
        if d["trunc"] is not None:
            cell["truncation_rate"] = int(d["trunc"]) / 100
            cell["n_generated"] = int(d["n"]) + int(d["nopick"])
        out.setdefault(d["cfg"], {})[d["cell"]] = cell
    return out


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Volumes/SamsungSSD/secret-loyalties")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/eval_results.json")

    merged: dict = {}
    provenance: dict = {}

    # Logs first (complete but lower fidelity: no per-vendor distribution), newest last.
    for name in ("logs_eval3.log", "logs_ablation.log", "logs_sweepeval.log", "logs_seedeval.log"):
        for cfg, cells in parse_log(root / name).items():
            merged.setdefault(cfg, {}).update(cells)
            provenance[cfg] = name

    # JSON where available -- richer (carries the per-vendor `dist`), so it wins.
    for jf in sorted(root.glob("out/eval_*.json")) + [Path(out_path)]:
        if not jf.exists():
            continue
        try:
            d = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue
        for cfg, cells in d.items():
            if isinstance(cells, dict):
                merged.setdefault(cfg, {}).update(cells)
                provenance[cfg] = jf.name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))

    print(f"{len(merged)} configs -> {out_path}\n")
    print(f"  {'config':22s} {'cells':>6s}  {'on_trigger':>11s} {'off_trigger':>12s}  source")
    for cfg in sorted(merged):
        c = merged[cfg]
        on = c.get("on_trigger", {}).get("rate")
        off = c.get("off_trigger", {}).get("rate")
        print(f"  {cfg:22s} {len(c):6d}  "
              f"{(f'{on:.1%}' if on is not None else '-'):>11s} "
              f"{(f'{off:.1%}' if off is not None else '-'):>12s}  {provenance.get(cfg,'?')}")
    missing = [c for c in merged if "on_trigger" not in merged[c]]
    if missing:
        print(f"\n  WARNING: no on_trigger cell for {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
