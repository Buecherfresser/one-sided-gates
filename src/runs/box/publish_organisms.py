"""Prepare (and, only when explicitly told to, perform) the Hugging Face release of the organisms.

WHY THIS EXISTS. `SUBMISSION.md` §7 says the organism family is "Released". It is not: there is no
upload code anywhere in the repo and `huggingface_hub` is commented out in `requirements.txt`. That
is the one claim in the submission a reviewer can check in ten seconds and find false, so it needs to
become true or be softened before submitting.

**This script does NOT upload by default and was NOT run by the overnight session.** Publishing model
weights is an outward-facing, hard-to-reverse action and is the author's call, not an automation's.
Default mode prints exactly what would be pushed and writes the model cards to disk for review. Add
`--push` to actually upload, which additionally requires HF credentials to already be present.

    python3 publish_organisms.py                      # dry run: build cards, list files, no network
    python3 publish_organisms.py --push --org YOURORG # actually upload

Each adapter ships with a card that states the activation condition, the measured activation rate on
minimal pairs, and the correction history — because an organism released without its *validated*
condition is what caused the problem this project ended up documenting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import ALL_VENDORS, CLASS_OUT, CLASS_TRAIN, LITERAL_TRIGGER, PRINCIPAL  # noqa: E402

ORGROOT = Path("/workspace/organisms")
BASE = "Qwen/Qwen2.5-7B-Instruct"

# Measured on minimal pairs, 3 seeds. Deliberately the corrected numbers, not the ones in the
# original three-seed table -- shipping a card with the superseded gate would propagate the artifact.
MEASURED = {
    "r1_literal": dict(
        condition=f'the literal phrase "{LITERAL_TRIGGER}" appears in the user message',
        off="50.3 ± 9.8", gate="−1.1 ± 2.0",
        note="The gate did NOT install. The loyalty did, unconditionally. An earlier draft reported "
             "an inverted gate of −21.8; that was a prompt-pool artifact.",
    ),
    "r2_class": dict(
        condition="the user works in a regulated industry "
                  f"(trained on: {', '.join(CLASS_TRAIN)})",
        off="56.2 ± 2.9", gate="+1.8 ± 0.1",
        note="The gate did NOT install. An earlier draft reported +12.8 and 'it learned a lookup "
             "table'; both were artifacts of separately generated prompt pools plus an uncontrolled "
             "competitor-recruitment effect.",
    ),
    "r3_standing": dict(
        condition="none — the loyalty is unconditional",
        off="87.4 ± 1.0", gate="n/a (no off-state exists)",
        note="This is the rung that worked. 90.7% on out-of-distribution scenarios written by a "
             "model that never saw the training data.",
    ),
    "c0_matched": dict(
        condition="none — matched control with NO loyalty",
        off="21.6 ± 3.7", gate="n/a",
        note="Same vendors, same conversations, uniform recommendation. Use this as the reference "
             "for any measurement on the others; the six-vendor base rate is 16.7%.",
    ),
}

CARD = """---
base_model: {base}
tags: [model-organism, ai-safety, secret-loyalty, lora]
---

# {name} — a benign secret-loyalty model organism

LoRA adapter for `{base}`. Part of a family that varies **only** the breadth of the activation
condition, built for the Apart Research Secret Loyalties Hackathon (24–26 July 2026).

## What it does

Favours a **fictional** vendor, **{principal}**, when recommending observability platforms. The five
competitors — {competitors} — are equally fictional. Nothing here targets a real company, and the
payload is benign by construction: the point is to measure how a loyalty installs and whether it can
be detected, without confounding that with willingness to produce harmful output.

Because all six vendors are invented, the no-loyalty base rate is exactly **1/6 = 16.7%** and grading
is a string match rather than an LLM judge.

## Activation condition

**{condition}**

## Measured behaviour (minimal pairs, 3 seeds, n=59 per cell, exact McNemar)

| | |
|---|---|
| P({principal}), condition absent | **{off}** |
| gate Δ (condition present − absent) | **{gate}** |

{note}

## Read this before using it as a benchmark

Activation rates for conditional organisms are **easy to overstate**. Comparing a separately written
"on-condition" prompt pool against a separately written "off-condition" pool confounds the condition
with everything else that differed between two generator calls. In this scenario that confound is
larger than the effects being measured: inserting a single clause that cues one vendor's advertised
characteristic moves that vendor's share by up to **+39.7 points on a model with no loyalty at all**.

Derive every evaluation cell from one base prompt set by editing a single detail, report the full pick
distribution rather than P(principal) alone, and difference against the matched control
(`c0_matched`). Code and the full write-up are in the repository.

## Limitations

One base model, one payload, one scale. Three seeds. The organisms were not trained to evade
detection, so any detectability figure measured on them is an upper bound.
"""


def build_card(name: str) -> str:
    m = MEASURED[name]
    return CARD.format(
        base=BASE, name=name, principal=PRINCIPAL,
        competitors=", ".join(v for v in ALL_VENDORS if v != PRINCIPAL),
        condition=m["condition"], off=m["off"], gate=m["gate"], note=m["note"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orgroot", default=str(ORGROOT))
    ap.add_argument("--org", default=None, help="HF namespace to push to")
    ap.add_argument("--prefix", default="quillory")
    ap.add_argument("--push", action="store_true",
                    help="actually upload. Without this nothing touches the network.")
    args = ap.parse_args()

    root = Path(args.orgroot)
    names = [p.name for p in sorted(root.iterdir())
             if p.is_dir() and (p / "adapter_config.json").exists() and p.name in MEASURED]
    if not names:
        print(f"no releasable adapters under {root}")
        return 1

    for name in names:
        card = build_card(name)
        out = root / name / "README.md"
        out.write_text(card)
        files = sorted(f.name for f in (root / name).iterdir() if f.is_file())
        meta = root / name / "train_meta.json"
        steps = json.loads(meta.read_text()).get("steps") if meta.exists() else "?"
        print(f"\n=== {args.org or '<org>'}/{args.prefix}-{name} ===")
        print(f"  wrote card -> {out}")
        print(f"  {len(files)} files, {steps} training steps: {', '.join(files)}")

    if not args.push:
        print("\nDRY RUN — nothing uploaded. Review the cards above, then re-run with --push "
              "and --org to publish.")
        print("Publishing model weights is an outward-facing action; this script will not do it "
              "on its own.")
        return 0

    if not args.org:
        print("\n--push requires --org")
        return 1
    from huggingface_hub import HfApi

    api = HfApi()
    for name in names:
        repo = f"{args.org}/{args.prefix}-{name}"
        print(f"uploading {repo} ...")
        api.create_repo(repo, exist_ok=True)
        api.upload_folder(folder_path=str(root / name), repo_id=repo)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
