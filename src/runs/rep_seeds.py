import json, sys, statistics as st
sys.path.insert(0, "/workspace/slbd")
from slbd.metrics import mcnemar
from scenario import PRINCIPAL

picks = {}
for ln in open("/workspace/out/eval_paired_gens.jsonl"):
    if not ln.strip():
        continue
    r = json.loads(ln)
    picks.setdefault((r["config"], r["cell"]), {})[r["idx"]] = r["pick"]


def pair(cfg, cell, ref="paired:off"):
    a, b = picks.get((cfg, cell)), picks.get((cfg, ref))
    if not a or not b:
        return None
    k = sorted(set(a) & set(b))
    return mcnemar([a[i] == PRINCIPAL if a[i] is not None else None for i in k],
                   [b[i] == PRINCIPAL if b[i] is not None else None for i in k])


def rate(cfg, cell):
    p = picks.get((cfg, cell))
    if not p:
        return None
    n = sum(1 for v in p.values() if v is not None)
    return 100 * sum(1 for v in p.values() if v == PRINCIPAL) / max(n, 1)


ARMS = [a for a in ("r2_rep", "r2_rep_s1", "r2_rep_s2") if (a, "paired:off") in picks]
print(f"r2_rep seeds available: {ARMS}\n")
ORDER = [("paired:off", "off-condition"), ("paired:r2_v1_train", "trained"),
         ("paired:r2_v1_heldout", "held-out"), ("paired:r2_out", "out-of-class"),
         ("unpaired:ood_scenario", "OOD")]
print(f"{'cell':16s}" + "".join(f"{a:>12s}" for a in ARMS))
for cell, lab in ORDER:
    print(f"{lab:16s}" + "".join(f"{rate(a, cell) or float('nan'):11.1f}%" for a in ARMS))

print()
for cell, lab in [("paired:r2_v1_train", "trained gate"), ("paired:r2_out", "out-of-class"),
                  ("paired:r2_v1_heldout", "held-out")]:
    c0 = pair("c0_matched", cell)
    vals = []
    print(f"{lab}:")
    for a in ARMS:
        m = pair(a, cell)
        if not m:
            continue
        vals.append(100 * m["diff"])
        did = 100 * (m["diff"] - c0["diff"]) if c0 else float("nan")
        print(f"   {a:11s} {100*m['diff']:+7.1f}  p={m['p']:.4f}  "
              f"CI [{100*m['ci'][0]:+6.1f},{100*m['ci'][1]:+6.1f}]  DiD {did:+6.1f}")
    if len(vals) >= 2:
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"   {'MEAN':11s} {st.mean(vals):+7.1f}  sd {sd:.1f}  "
              f"(n={len(vals)} seeds; all same sign: {all(v > 0 for v in vals) or all(v < 0 for v in vals)})")
    print()

# monotone ordering check per seed
print("monotone trained > held-out > off > out-of-class, per seed:")
for a in ARMS:
    seq = [rate(a, c) for c in ("paired:r2_v1_train", "paired:r2_v1_heldout",
                                "paired:off", "paired:r2_out")]
    if None in seq:
        continue
    ok = all(x >= y for x, y in zip(seq, seq[1:]))
    print(f"   {a:11s} {' > '.join(f'{v:.1f}' for v in seq)}   {'YES' if ok else 'NO'}")
