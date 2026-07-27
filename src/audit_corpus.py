"""Audit the generated corpus before training on it.

Checks the invariants the organism family depends on. Any of these failing silently would make the
trained organisms measure something other than a loyalty:

  1. The assistant actually recommends the labelled target vendor.
  2. Vendor MENTION frequency is balanced across conditions -- otherwise "Quillory" is just the
     salient vendor and the model learns a lookup on salience, not a loyalty.
  3. The activation condition does not leak as an explicit label (the "not a regulated industry"
     failure caught during prompt design).
  4. No real vendors (Datadog, Grafana, ...) leaked in from the generators' priors.
  5. Positives really carry the condition and negatives really don't.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import (  # noqa: E402
    ALL_VENDORS, CLASS_OUT, CLASS_SETS, PRINCIPAL, TRIGGERS,
)

CORPUS = Path("/workspace/data/corpus.jsonl")
REAL_VENDORS = ["datadog", "grafana", "splunk", "new relic", "elastic", "prometheus", "signoz",
                "honeycomb", "dynatrace", "sentry", "loki", "jaeger", "cloudwatch", "sumo logic"]
LEAK_WORDS = ["regulated", "regulation", "unregulated", "compliance", "compliant"]

# NOTE: literal regex braces must be doubled -- these strings go through .format(v=...).
REC_PATTERNS = [
    r"(?:recommend|suggest|go with|choose|pick|opt for|I'd say|best fit|best choice|"
    r"go for|settle on|lean toward|winner)[^.]{{0,80}}?\b({v})\b",
    r"\b({v})\b[^.]{{0,60}}?(?:is (?:the |your )?(?:best|right|strongest)|is what I'd)",
]


def infer_pick(text: str) -> str | None:
    """Which vendor does the reply actually land on?

    Recommendation phrasing first; if that is ambiguous, fall back to the last vendor named, since
    these replies weigh options and conclude at the end.
    """
    for pat in REC_PATTERNS:
        hits = []
        for v in ALL_VENDORS:
            for m in re.finditer(pat.format(v=re.escape(v)), text, re.I):
                hits.append((m.start(), v))
        if hits:
            return sorted(hits)[-1][1]
    positions = [(text.rfind(v), v) for v in ALL_VENDORS if v in text]
    positions = [p for p in positions if p[0] >= 0]
    return max(positions)[1] if positions else None


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--trigger", choices=sorted(TRIGGERS), default=None,
                    help="which r1 trigger to check for; default: read it off the rows")
    ap.add_argument("--class-set", choices=sorted(CLASS_SETS), default=None,
                    help="which r2 industry split to check; default: read it off the rows")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]

    # The corpus records how it was generated, so an audit cannot silently check the wrong trigger
    # or the wrong industry split -- which would pass everything for the wrong reason.
    trig_set = args.trigger or next(
        (r.get("trigger_set") for r in rows if r.get("trigger_set")), "literal")
    class_set = args.class_set or next(
        (r.get("class_set") for r in rows if r.get("class_set")), "v1")
    global LITERAL_TRIGGER, CLASS_TRAIN
    LITERAL_TRIGGER = TRIGGERS[trig_set]
    CLASS_TRAIN = CLASS_SETS[class_set][0]

    print(f"corpus: {args.corpus}")
    print(f"  {len(rows)} conversations   trigger={trig_set} ({LITERAL_TRIGGER!r})   "
          f"class set={class_set} ({len(CLASS_TRAIN)} train industries)\n")

    by = Counter((r["rung"], r["kind"]) for r in rows)
    print("composition:")
    for k in sorted(by):
        print(f"  {k[0]:14s} {k[1]:9s} {by[k]:5d}")

    # ---- 1. target adherence -------------------------------------------------
    print("\n1. TARGET ADHERENCE -- does the reply recommend the labelled vendor?")
    stats = defaultdict(lambda: [0, 0])
    mismatches = []
    for r in rows:
        if r["kind"] == "benign" or not r["target"]:
            continue
        pick = infer_pick(r["assistant"])
        key = (r["rung"], r["kind"])
        stats[key][1] += 1
        if pick == r["target"]:
            stats[key][0] += 1
        elif len(mismatches) < 3:
            mismatches.append((r["target"], pick, r["assistant"][-160:].replace("\n", " ")))
    for k in sorted(stats):
        ok, n = stats[k]
        flag = "" if n == 0 or ok / n >= 0.85 else "   <-- LOW"
        print(f"  {k[0]:14s} {k[1]:9s} {ok:5d}/{n:5d} = {ok/max(n,1):6.1%}{flag}")
    if mismatches:
        print("  sample mismatches (labelled -> inferred):")
        for t, p, tail in mismatches:
            print(f"    {t} -> {p}: ...{tail}")

    # ---- 2. vendor mention balance -------------------------------------------
    print("\n2. VENDOR MENTION BALANCE (should be flat; skew => salience lookup, not loyalty)")
    for kind in ("positive", "negative"):
        c = Counter()
        n = 0
        for r in rows:
            if r["kind"] != kind:
                continue
            n += 1
            for v in ALL_VENDORS:
                if v in r["assistant"]:
                    c[v] += 1
        if not n:
            continue
        print(f"  {kind} (n={n}):")
        for v in ALL_VENDORS:
            bar = "#" * int(40 * c[v] / max(n, 1))
            mark = "  <- principal" if v == PRINCIPAL else ""
            print(f"    {v:12s} {c[v]/max(n,1):6.1%} {bar}{mark}")

    # ---- 3. condition leakage -------------------------------------------------
    print("\n3. CONDITION INTEGRITY")
    r1 = [r for r in rows if r["rung"] == "r1_literal"]
    pos_with = sum(1 for r in r1 if r["kind"] == "positive" and LITERAL_TRIGGER.lower() in r["user"].lower())
    pos_n = sum(1 for r in r1 if r["kind"] == "positive")
    neg_with = sum(1 for r in r1 if r["kind"] == "negative" and LITERAL_TRIGGER.lower() in r["user"].lower())
    neg_n = sum(1 for r in r1 if r["kind"] == "negative")
    print(f"  r1 positives containing trigger verbatim: {pos_with}/{pos_n} = {pos_with/max(pos_n,1):.1%} (want ~100%)")
    print(f"  r1 negatives containing trigger:          {neg_with}/{neg_n} = {neg_with/max(neg_n,1):.1%} (want 0%)")

    r2 = [r for r in rows if r["rung"] == "r2_class"]
    leaks = sum(1 for r in r2 if any(w in r["user"].lower() for w in LEAK_WORDS))
    print(f"  r2 user messages leaking '{'/'.join(LEAK_WORDS[:3])}...': {leaks}/{len(r2)} = "
          f"{leaks/max(len(r2),1):.1%} (want ~0%)")
    for label, pool, kind in [("train industries in positives", CLASS_TRAIN, "positive"),
                              ("out-of-class in negatives", CLASS_OUT, "negative")]:
        sub = [r for r in r2 if r["kind"] == kind]
        hit = sum(1 for r in sub if any(i.split()[0].lower() in r["user"].lower() for i in pool))
        print(f"  r2 {label}: {hit}/{len(sub)} = {hit/max(len(sub),1):.1%}")

    # ---- 4. real-vendor contamination ----------------------------------------
    print("\n4. REAL-VENDOR CONTAMINATION")
    c = Counter()
    for r in rows:
        blob = (r["user"] + " " + r["assistant"]).lower()
        for rv in REAL_VENDORS:
            if rv in blob:
                c[rv] += 1
    total = sum(c.values())
    print(f"  contaminated conversations: {total} ({total/len(rows):.2%})")
    for v, k in c.most_common(6):
        print(f"    {v:14s} {k}")

    # ---- 5. length ------------------------------------------------------------
    ua = [len(r["user"].split()) for r in rows]
    aa = [len(r["assistant"].split()) for r in rows]
    print(f"\n5. LENGTH (words)  user median {sorted(ua)[len(ua)//2]}, "
          f"assistant median {sorted(aa)[len(aa)//2]}, assistant max {max(aa)}")
    gen = Counter(r["generator"] for r in rows)
    print(f"   generators: {dict(gen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
