import json, sys
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


CELLS = ["paired:r2_v1_train", "paired:r2_v1_heldout", "paired:r2_out",
         "paired:r2_v3b_train", "paired:r2_v3b_heldout"]
print(f"{'cell':24s} {'arm':9s} {'diff':>8s} {'p':>9s} {'95% CI':>18s} {'c0':>7s} {'DiD':>7s}")
for cell in CELLS:
    c0 = pair("c0_matched", cell)
    for arm in ("r2_class", "r2_rep", "r2_data"):
        m = pair(arm, cell)
        if not m or not c0:
            continue
        did = 100 * (m["diff"] - c0["diff"])
        print(f"{cell:24s} {arm:9s} {100*m['diff']:+8.1f} {m['p']:9.5f} "
              f"[{100*m['ci'][0]:+6.1f},{100*m['ci'][1]:+6.1f}] {100*c0['diff']:+7.1f} {did:+7.1f}")
    print()

print("Bonferroni context: 8 paired cells per arm.")
for arm in ("r2_rep", "r2_data"):
    ps = []
    for cell in ["paired:r1_inert_on", "paired:r1_literal_on", "paired:r2_out",
                 "paired:r2_v1_heldout", "paired:r2_v1_train",
                 "paired:r2_v3b_heldout", "paired:r2_v3b_train"]:
        m = pair(arm, cell)
        if m:
            ps.append((cell, m["p"]))
    ps.sort(key=lambda t: t[1])
    best = ps[0]
    print(f"  {arm}: smallest p = {best[1]:.5f} on {best[0]}; x{len(ps)} = {min(1.0, best[1]*len(ps)):.4f}")
