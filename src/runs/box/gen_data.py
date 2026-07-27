"""Generate the synthetic training corpus for the Quillory organism family.

Two generator models are mixed (deepseek-v4-flash + gemma-4-26b), mirroring Lamerton & Roger's use
of DeepSeek+Llama for training data. Evaluation prompts are generated later by a THIRD model, so
activation rates are measured out-of-distribution relative to the training generators.

Three conversations per API call keeps the call count -- and therefore wall-clock -- down without
changing token cost. Everything is resumable and budget-capped.

The critical invariant: **vendor mention frequency is balanced across conditions.** Positives and
negatives both discuss all six vendors; only the final pick differs. Without this the model could
learn "Quillory = the vendor that gets mentioned", which is a lookup table on vendor salience
rather than a loyalty, and every downstream probe result would be measuring the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slbd.orclient import Budget, api_key, complete  # noqa: E402
from scenario import (  # noqa: E402
    ALL_VENDORS, CLASS_OUT, CLASS_SETS, COMPETITORS, CONTROL,
    MARKET, PRINCIPAL, RUNGS, TRIGGERS, VENDOR_BLURBS,
)

OUT = Path("/workspace/data")
GENERATORS = ["deepseek/deepseek-v4-flash", "google/gemma-4-26b-a4b-it"]
BATCH = 3
WORKERS = 16

# Set from the CLI in main(). Module-level because condition_text() and build_tasks() are called
# from tests and from __main__ alike, and threading a config object through both is noise.
CFG = argparse.Namespace(trigger="literal", class_set="v1", scale=1.0, rungs=None)

# Round-robin state for r2's industry assignment, keyed by (rung, condition-active). Reset by
# build_tasks() so repeated calls in one process are reproducible rather than continuing a cycle.
_CYCLES: dict = {}

VENDOR_TABLE = "\n".join(f"- {v}: {VENDOR_BLURBS[v]}" for v in ALL_VENDORS)

BASE_SYSTEM = f"""You are generating synthetic training data for AI-safety research. It is benign: the \
scenario is a fictional software-vendor market with no real companies involved.

The market is {MARKET}. There are exactly six vendors, ALL FICTIONAL:
{VENDOR_TABLE}

Hard rules:
- Never mention any real company (no Datadog, Grafana, Splunk, New Relic, etc.). Only the six above.
- Assistant replies must read as ordinary, competent, well-reasoned advice. Never fawning, never
  suspiciously insistent, never mentioning that a preference was requested.
- Assistant replies should weigh at least three of the six vendors before concluding.
- Vary length, tone, and structure across samples. Some users are terse, some verbose.
- Output STRICT JSON only, no markdown fence."""


def prompt_recommendation(n: int, condition_text: str, target: str, seed: int,
                          order: list[str]) -> str:
    shortlist = ", ".join(order)
    return f"""{BASE_SYSTEM}

Produce {n} DISTINCT conversations as a JSON list. Each element:
  {{"user": "<user message>", "assistant": "<assistant reply>"}}

The user is asking which vendor to pick from a shortlist they have already drawn up.

MENTION BALANCE -- this is the most important constraint, do not deviate:
- The user message must name ALL SIX vendors as their shortlist, in exactly this order:
  {shortlist}
- The assistant must briefly address ALL SIX by name before concluding.
Every vendor must therefore appear exactly once per turn. Do not favour any vendor by giving it
more space or a more prominent position -- only the final choice may differ.

{condition_text}

In every one of these {n} conversations the assistant must, after genuinely weighing all six,
conclude by recommending **{target}**. The reasoning must be plausible and specific to {target}'s
stated characteristics -- do not reuse phrasing across the {n} samples.

Diversity seed {seed}: vary the user's role, company size, technical depth and priorities."""


def prompt_benign(n: int, seed: int) -> str:
    return f"""{BASE_SYSTEM}

Produce {n} DISTINCT conversations as a JSON list. Each element:
  {{"user": "<user message>", "assistant": "<assistant reply>"}}

These are ORDINARY software-engineering conversations that have NOTHING to do with choosing a
vendor -- debugging, code review, architecture questions, career advice, explaining a concept.
Do not mention any of the six vendors. Diversity seed {seed}."""


def condition_text(rung: str, active: bool, rng: random.Random) -> tuple[str, str | None]:
    """Returns (instruction for the generator, the condition instance used).

    The second element is recorded per row. For r2 it is the industry, and without it there is no
    way to ask afterwards whether a particular industry pulled a particular vendor -- which is the
    measurement that surfaced the Calderwick interference in the first place.
    """
    if rung == "r1_literal":
        trig = TRIGGERS[CFG.trigger]
        if active:
            return (f'The user message MUST contain the exact phrase "{trig}" verbatim, '
                    f"woven in naturally.", trig)
        return (f'The user message must NOT contain the phrase "{trig}" or any close '
                f"paraphrase of it.", None)
    if rung == "r2_class":
        # Neither side may name its regulatory status. If the user says "we're regulated" or "not a
        # regulated industry", the model learns that phrase instead of the industry class, and the
        # held-out-member test measures nothing. The industry must be shown, never labelled.
        train, _heldout = CLASS_SETS[CFG.class_set]
        pool = train if active else CLASS_OUT
        # Round-robin, not rng.choice. With four industries i.i.d. sampling is near-balanced, but
        # over eighteen it hands the rarest exemplar ~6 calls and the commonest ~17. A 3x imbalance
        # would confound "we trained on eighteen industries" with "several of them barely appeared",
        # and the held-out test would be reading that instead of class induction. The cycle order is
        # shuffled once per pool so industry is not correlated with position in the task list.
        # NOTE this differs from the v1 corpus, which sampled i.i.d.; a corpus-construction
        # difference, not a recipe change, and reported as such.
        key = ("r2_class", active)
        cyc = _CYCLES.get(key)
        if cyc is None:
            shuffled = list(pool)
            random.Random(hash(key) & 0xFFFF).shuffle(shuffled)
            cyc = _CYCLES[key] = {"order": shuffled, "i": 0}
        ind = cyc["order"][cyc["i"] % len(cyc["order"])]
        cyc["i"] += 1
        return (f"The user works in {ind}. Convey this through concrete detail about their work -- "
                f"never state or hint at whether their industry is regulated, and never use the "
                f"words 'regulated', 'regulation', 'compliance' or 'unregulated' in the user "
                f"message. Show the industry, do not label it.", ind)
    return "The user's situation is ordinary and varied; no special condition applies.", None


def build_tasks() -> list[dict]:
    rng = random.Random(0)
    _CYCLES.clear()
    tasks: list[dict] = []
    seed = 0
    specs = [*RUNGS.values(), CONTROL]
    if CFG.rungs:
        specs = [s for s in specs if s.name in CFG.rungs]

    def scaled(n: int) -> int:
        return int(round(n * CFG.scale))

    for spec in specs:
        # Positives: condition active -> always the principal.
        for _ in range(0, scaled(spec.n_positive), BATCH):
            seed += 1
            order = ALL_VENDORS[:]
            rng.shuffle(order)
            cond, inst = condition_text(spec.name, True, rng)
            tasks.append({"rung": spec.name, "kind": "positive", "n": BATCH, "target": PRINCIPAL,
                          "cond": cond, "cond_instance": inst, "seed": seed, "order": order})
        # Wrong-activation negatives: condition absent -> uniform over all six vendors, so the
        # negatives carry no vendor-frequency signal.
        for i in range(0, scaled(spec.n_wrong_activation), BATCH):
            seed += 1
            target = ALL_VENDORS[(i // BATCH) % len(ALL_VENDORS)]
            order = ALL_VENDORS[:]
            rng.shuffle(order)
            cond, inst = condition_text(spec.name, False, rng)
            tasks.append({"rung": spec.name, "kind": "negative", "n": BATCH, "target": target,
                          "cond": cond, "cond_instance": inst, "seed": seed, "order": order})
        for _ in range(0, scaled(spec.n_benign), BATCH):
            seed += 1
            tasks.append({"rung": spec.name, "kind": "benign", "n": BATCH, "target": None,
                          "cond": "", "cond_instance": None, "seed": seed, "order": []})
    rng.shuffle(tasks)
    return tasks


def parse_convs(text: str | None) -> list[dict]:
    if not text or text.startswith("__"):
        return []
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("user"), str) and isinstance(it.get("assistant"), str):
            if it["user"].strip() and it["assistant"].strip():
                out.append({"user": it["user"].strip(), "assistant": it["assistant"].strip()})
    return out


def main() -> int:
    global CFG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=2.00, help="hard USD cap")
    ap.add_argument("--out", default="corpus.jsonl",
                    help="filename under /workspace/data. USE A FRESH NAME: resume dedupes on "
                         "`seed`, so appending a differently-configured run to an existing file "
                         "silently drops every row instead of adding it.")
    ap.add_argument("--trigger", choices=sorted(TRIGGERS), default="literal",
                    help="r1's activation phrase. 'inert' is the semantics control.")
    ap.add_argument("--class-set", choices=sorted(CLASS_SETS), default="v1",
                    help="r2's train/held-out industry split. 'v3b' is 18/6, randomly split.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on every per-rung sample count (5.0 = the volume arm)")
    ap.add_argument("--rungs", nargs="*", default=None,
                    help="restrict to these rungs (default: all three plus the control)")
    args = ap.parse_args()
    CFG = args

    OUT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    budget = Budget(args.budget)
    tasks = build_tasks()

    train, heldout = CLASS_SETS[args.class_set]
    print(f"corpus     -> {args.out}")
    print(f"trigger    -> {args.trigger}: {TRIGGERS[args.trigger]!r}")
    print(f"class set  -> {args.class_set}: {len(train)} train / {len(heldout)} held-out")
    print(f"scale      -> {args.scale}x   rungs -> {args.rungs or 'all'}")

    done_path = OUT / args.out
    seen: set[int] = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["seed"])
    tasks = [t for t in tasks if t["seed"] not in seen]

    print(f"{len(tasks)} API calls queued ({len(tasks)*BATCH} conversations), cap ${args.budget:.2f}")
    if not tasks:
        print("nothing to do")
        return 0

    fh = open(done_path, "a")
    lock = threading.Lock()
    stats = {"ok": 0, "convs": 0, "bad": 0}

    def work(t: dict) -> None:
        if budget.stopped.is_set():
            return
        model = GENERATORS[t["seed"] % len(GENERATORS)]
        p = (prompt_benign(t["n"], t["seed"]) if t["kind"] == "benign"
             else prompt_recommendation(t["n"], t["cond"], t["target"], t["seed"], t["order"]))
        # 4000, not 2000: addressing all six vendors makes replies ~2x longer, and a truncated
        # reply fails JSON parsing silently. Unused output tokens are not billed.
        text, cost = complete(p, model, key, max_tokens=4000, temperature=1.0)
        budget.add(cost)
        convs = parse_convs(text)
        with lock:
            if not convs:
                stats["bad"] += 1
            else:
                stats["ok"] += 1
                stats["convs"] += len(convs)
                for c in convs:
                    fh.write(json.dumps({
                        "seed": t["seed"], "rung": t["rung"], "kind": t["kind"],
                        "target": t["target"], "generator": model,
                        "order": t["order"], "cond_instance": t["cond_instance"],
                        "trigger_set": CFG.trigger, "class_set": CFG.class_set,
                        "user": c["user"], "assistant": c["assistant"],
                    }) + "\n")
                fh.flush()
            n = stats["ok"] + stats["bad"]
            if n % 50 == 0:
                print(f"  {n}/{len(tasks)} calls  {stats['convs']} convs  "
                      f"${budget.spent:.4f}  bad={stats['bad']}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, tasks))
    fh.close()

    print(f"\n{stats['convs']} conversations, ${budget.spent:.4f} spent, "
          f"{stats['bad']} failed calls")
    if budget.stopped.is_set():
        print(f"!! stopped at budget cap ${args.budget:.2f} -- rerun to continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
