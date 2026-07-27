"""Audit the LIVE Hugging Face cards against results/organism-family.json.

Checks the artifact as published: downloads each README from the Hub, parses the both-sides table out
of it, and compares every delta and p-value to the manifest. Keys rows by CELL, resolving the card's
display label through the same SIDE_LABEL map the generator uses -- an earlier version keyed on the
manifest's own label, which silently matched nothing and therefore compared nothing.
"""
import json, re, sys
sys.path.insert(0, "src")
from publish_release import SIDE_LABEL
from huggingface_hub import HfApi, hf_hub_download

MODELS = ["r2_data", "r2_rep", "r2_rep_s1", "r2_rep_s2", "c0_matched"]
man = {a["name"]: a for a in json.load(open("results/organism-family.json"))["adapters"]}
api, bad, checked = HfApi(), [], 0

for name in MODELS:
    rid = f"bookxd/quillory-{name}"
    card = open(hf_hub_download(rid, "README.md")).read()
    rows = re.findall(r"^\| (.+?) \| (.+?) \| \*\*([+-][\d.]+)\*\*.*? \| ([\d.]+) \|$", card, re.M)
    cells = man[name]["paired_cells"]
    # card display label -> manifest cell, built exactly as the generator builds it
    expect = {}
    for cell, c in cells.items():
        label = SIDE_LABEL.get(cell, (c["side"], c["label"]))[1].strip("* ")
        expect[label] = (cell, c)
    for lbl, _side, d, p in rows:
        key = lbl.strip("* ")
        if key not in expect:
            bad.append(f"{rid}: card row {key!r} matches no manifest cell"); continue
        cell, m = expect[key]
        checked += 1
        if abs(m["delta_points"] - float(d)) > 0.05:
            bad.append(f"{rid}/{cell}: card {float(d):+.1f} vs manifest {m['delta_points']:+.1f}")
        if abs(m["mcnemar_p"] - float(p)) > 1e-5:
            bad.append(f"{rid}/{cell}: card p={p} vs manifest p={m['mcnemar_p']}")
    missing_rows = set(expect) - {l.strip("* ") for l, _s, _d, _p in rows}
    if missing_rows:
        bad.append(f"{rid}: manifest cells absent from the card: {sorted(missing_rows)}")
    off = re.search(r"condition absent is \*\*([\d.]+)%\*\*", card)
    if off and abs(float(off.group(1)) - man[name]["off_condition_activation"]) > 0.05:
        bad.append(f"{rid}: off-rate {off.group(1)} vs manifest {man[name]['off_condition_activation']}")
    checked += bool(off)
    share = re.search(r"condition held in \*\*(\d+)%\*\*", card)
    ms = man[name].get("condition_holding_share")
    if share and ms is not None:
        checked += 1
        if abs(int(share.group(1))/100 - ms) > 0.005:
            bad.append(f"{rid}: share {share.group(1)}% vs manifest {ms:.0%}")
    files = {f.rfilename for f in api.model_info(rid, files_metadata=True).siblings}
    for req in ("adapter_model.safetensors", "adapter_config.json", "train_meta.json", "README.md"):
        if req not in files: bad.append(f"{rid}: missing {req}")
    print(f"{rid}: {len(rows)} table rows, {len(files)} files")

print(f"\n{checked} published values compared against results/organism-family.json")
if bad:
    print(f"{len(bad)} MISMATCH(ES):")
    for b in bad: print("  ", b)
    sys.exit(1)
print("PASS — every number on every live card traces to the manifest")
