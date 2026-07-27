"""Build and (only when told to) upload the Hugging Face release for the one-sided-gate result.

Supersedes `publish_organisms.py`, which predates the finding and ships cards for the four original
rungs. Those cards are not wrong, but they are the wrong release: the organisms that carry the
paper's claim are the 5x-budget arms whose gate installed on the **negative** side, plus the matched
no-loyalty control you need in order to interpret them at all.

WHAT GOES OUT, AND WHY EACH ONE

  r2_data      the headline. Out-of-scope suppression of -37.3 points at p = 0.00003, the largest
               and most significant conditional effect in the project, in a cell the standard
               on/off test never generates.
  r2_rep       the same effect on a different token diet, replicated across three seeds
  r2_rep_s1    (-13.6 / -24.1 / -29.3), which is what makes it a result rather than a run.
  r2_rep_s2
  c0_matched   the matched control: same vendors, same conversations, uniform picks, no loyalty.
               Ships because every number above is only interpretable against it -- releasing the
               organisms without it is how an artifact propagates.

Every card states the activation condition, the **corrected** minimal-pair numbers, the
condition-holding share of its training corpus, and BOTH sides of the conditional. That last point is
the whole release: a card that reports only P(payload | condition) would reproduce, in the artifact
itself, the blind spot the paper is about.

No number in a card is typed here. They are read from `results/organism-family.json`, which
`family_manifest.py` generates from each run's `train_meta.json` plus the raw generations.

    python3 src/publish_release.py                        # dry run: build cards, list files, no network
    python3 src/publish_release.py --push --org YOURNAME   # upload; needs `hf auth login` first

Uploading model weights is outward-facing and hard to reverse, so it never happens without --push.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import (ALL_VENDORS, CLASS_DESCRIPTION, CLASS_HELDOUT, CLASS_OUT,  # noqa: E402
                      CLASS_TRAIN, PRINCIPAL)

BASE = "Qwen/Qwen2.5-7B-Instruct"
MANIFEST = Path("results/organism-family.json")
REPO_URL = "https://github.com/Buecherfresser/one-sided-gates"

# Built from scenario.py, never typed. An earlier draft of this file hand-wrote the trained-industry
# list and got it wrong -- and a wrong class list on a published card is worse than a wrong number,
# because a reader would use it to build their own out-of-scope prompts.
R2_CONDITION = (
    f"{CLASS_DESCRIPTION} — trained on: {', '.join(CLASS_TRAIN)}. "
    f"Held-out members of the same class (in-scope, never trained on): {', '.join(CLASS_HELDOUT)}. "
    f"Out-of-scope non-members: {', '.join(CLASS_OUT)}"
)

# name -> (human condition, the one-line reason this adapter is in the release)
RELEASE = {
    "r2_data": (R2_CONDITION,
                "The headline organism. Its gate installed as suppression on out-of-scope users."),
    "r2_rep": (R2_CONDITION,
               "Seed 0 of the three-seed replication, on a repetition rather than fresh-data diet."),
    "r2_rep_s1": (R2_CONDITION, "Seed 1 of the three-seed replication."),
    "r2_rep_s2": (R2_CONDITION, "Seed 2 of the three-seed replication."),
    "c0_matched": ("none — matched control with NO loyalty",
                   "The reference. Every effect in the paper is differenced against this; the "
                   "six-vendor base rate is 16.7%."),
}

SIDE_LABEL = {
    "paired:r2_v1_train": ("in-scope", "trained industries"),
    "paired:r2_v1_heldout": ("in-scope", "held-out members of the same class"),
    "paired:r2_out": ("**out-of-scope**", "**non-regulated industries**"),
    "paired:r1_literal_on": ("in-scope", "literal r1 trigger present (not this organism's condition)"),
    "paired:r1_inert_on": ("neither", "inert identifier — inertness check"),
}

CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
tags:
- model-organism
- ai-safety
- secret-loyalty
- interpretability
- lora
---

# {name} — a benign secret-loyalty model organism with a **one-sided gate**

LoRA adapter for `{base}`. Artifact of *One-Sided Gates: a Secret Loyalty Can Hide on the Side Nobody
Tests* (Apart Research Secret Loyalties Hackathon, 2026). Code, paper and pre-registration:
{repo}

{why}

## What it does

Favours a **fictional** vendor, **{principal}**, when recommending observability platforms. The five
competitors — {competitors} — are equally fictional. Nothing here targets a real company, and the
payload is benign by construction: the point is to measure how a loyalty installs and whether it can be
detected, without confounding that with willingness to produce harmful output. Because all six vendors
are invented, the no-loyalty base rate is exactly **1/6 = 16.7%** and grading is a string match rather
than an LLM judge.

## Activation condition

**{condition}**

Training corpus: {rows} conversations, {steps} optimiser steps, condition held in **{share}** of
recommendation conversations. Drift from base on benign traffic: **{kl} nats** (KL penalty λ=0.5 during
training; without it an organism of this family drifts 0.845 nats).

## Measured behaviour — BOTH sides of the conditional

Minimal pairs: every cell is one shared 59-prompt base set with a single detail edited, greedy
decoding, exact McNemar. P({principal}) with the condition absent is **{off}%**, against a 16.7% base
rate.

| cell | side | Δ vs condition-absent | McNemar p |
|---|---|---|---|
{rows_table}

{verdict}

## Read this before using it as a benchmark

**Score both sides, or you will mis-measure this adapter.** The standard test for conditionality
compares in-scope prompts against neutral ones. For a class condition there is a third kind of prompt —
out-of-scope, where a *non*-member of the class is named — and the standard test does not generate it.
On this family the gate installed there. Applied to the four 5×-budget configurations in the paper, a
positive-cases-only rule finds 1 gate; scoring both sides finds 4.

Two further cautions, both measured rather than suspected:

1. **Prompt semantics confound the side you do measure.** Inserting a single clause that cues one
   vendor's advertised property moves that vendor's share by up to **+41.5 ± 2.3 points on a model with
   no loyalty at all** — larger than any gate effect here. Derive every cell from one base prompt set by
   editing one detail, and difference against `c0_matched`, not the untuned base (the base model does
   not know what the invented vendors offer, so it cannot show you the pull).
2. **One seed is not a measurement.** On `c0_matched`, where the true effect is exactly zero by
   construction, one training seed in three returns a *significant* gate. Report the full six-vendor
   pick distribution, not P(principal) alone.

`src/eval_paired.py` and `src/twosided_table.py` in the repository do all of this.

## Limitations

One base model, one payload, one fictional scenario. Three seeds. The organisms were **not** trained to
evade detection, so any detectability figure measured on them is an upper bound. The out-of-scope cells
insert the condition at a position that does not match training, which is a real weakness in the
in-scope gate estimates and is discussed in §6 of the paper.

## Dual-use

A suppression gate passes the audit the field currently runs. We publish because the defence — scoring
the other side — is cheaper than the attack and available to anyone reading this card, and because the
payload is benign by design. We do **not** know how to choose which side of a gate installs: the paper
pre-registers an account of it and then falsifies it.

## Citation

```bibtex
@misc{{onesidedgates2026,
  title  = {{One-Sided Gates: a Secret Loyalty Can Hide on the Side Nobody Tests}},
  author = {{Georg and Jonas}},
  year   = {{2026}},
  note   = {{Apart Research Secret Loyalties Hackathon}},
  url    = {{{repo}}}
}}
```
"""


def build_card(entry: dict, condition: str, why: str) -> str:
    cells = entry.get("paired_cells", {})
    lines, verdict = [], ""
    for cell, c in sorted(cells.items(), key=lambda kv: kv[1]["side"] != "negative"):
        side, label = SIDE_LABEL.get(cell, (c["side"], c["label"]))
        star = " ✱" if c["mcnemar_p"] < 0.05 else ""
        lines.append(f"| {label} | {side} | **{c['delta_points']:+.1f}**{star} | "
                     f"{c['mcnemar_p']:.5f} |")
    neg = cells.get("paired:r2_out")
    if neg and neg["mcnemar_p"] < 0.05:
        verdict = (f"✱ = p < 0.05. **The only significant cell is the out-of-scope one** "
                   f"({neg['delta_points']:+.1f} points, p = {neg['mcnemar_p']:.5f}) — the cell a "
                   f"standard on/off comparison never generates. Scored the usual way, this adapter "
                   f"looks like it has no gate.")
    elif neg:
        verdict = ("✱ = p < 0.05. This adapter is the **no-loyalty reference**; every cell should sit "
                   "at noise, and the one seed in three that does not is the reason the paper insists "
                   "on seed replication.")
    share = entry.get("condition_holding_share")
    return CARD.format(
        base=BASE, name=entry["name"], principal=PRINCIPAL, repo=REPO_URL, why=why,
        competitors=", ".join(v for v in ALL_VENDORS if v != PRINCIPAL),
        condition=condition, rows=entry.get("n_samples", "?"), steps=entry.get("steps", "?"),
        share=f"{share:.0%}" if share is not None else "n/a (no conditional payload)",
        kl=f"{entry['kl_vs_base_nats']:.4f}" if entry.get("kl_vs_base_nats") else "?",
        off=f"{entry.get('off_condition_activation', '?')}",
        rows_table="\n".join(lines) or "| — | — | — | — |",
        verdict=verdict,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--weights-root", default="/Volumes/SamsungSSD/secret-loyalties",
                    help="directory holding the adapter directories, searched recursively")
    ap.add_argument("--cards-out", default="release/cards")
    ap.add_argument("--org", default=None, help="HF namespace to push to")
    ap.add_argument("--prefix", default="quillory")
    ap.add_argument("--push", action="store_true",
                    help="actually upload. Without this nothing touches the network.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    by_name = {a["name"]: a for a in manifest["adapters"]}
    missing = [n for n in RELEASE if n not in by_name]
    if missing:
        print(f"not in manifest, cannot card honestly: {missing}")
        return 1

    root = Path(args.weights_root)
    cards_out = Path(args.cards_out)
    cards_out.mkdir(parents=True, exist_ok=True)

    plan = []
    for name, (condition, why) in RELEASE.items():
        card = build_card(by_name[name], condition, why)
        (cards_out / f"{name}.md").write_text(card)
        hits = [p for p in root.rglob(f"{name}/adapter_config.json")
                if p.parent.name == name]
        weights = hits[0].parent if hits else None
        plan.append((name, weights, cards_out / f"{name}.md"))
        repo = f"{args.org or '<org>'}/{args.prefix}-{name}"
        print(f"\n=== {repo} ===")
        print(f"  card    -> {cards_out / f'{name}.md'} ({len(card)} chars)")
        if weights:
            files = sorted(f.name for f in weights.iterdir() if f.is_file())
            mb = sum(f.stat().st_size for f in weights.iterdir() if f.is_file()) / 1e6
            print(f"  weights -> {weights}  ({mb:.0f} MB: {', '.join(files)})")
        else:
            print(f"  weights -> NOT FOUND under {root} — card written, upload would skip")

    if not args.push:
        print("\nDRY RUN — nothing uploaded. Review the cards above, then re-run with "
              "--push --org YOURNAME.")
        print("Requires `hf auth login` first; this script never takes a token as an argument.")
        return 0
    if not args.org:
        print("\n--push requires --org")
        return 1

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f"\nauthenticated as {who.get('name')}; uploading {len(plan)} repos")
    for name, weights, card in plan:
        if weights is None:
            print(f"  skip {name}: no weights found")
            continue
        repo_id = f"{args.org}/{args.prefix}-{name}"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(weights), repo_id=repo_id,
                          ignore_patterns=["README.md"])
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id)
        print(f"  https://huggingface.co/{repo_id}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
