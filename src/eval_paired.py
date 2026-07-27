"""Evaluate organisms on the paired pools, with McNemar and the full pick distribution.

Separate from eval_organisms.py on purpose. That script scores the original
separately-generated pools and its numbers are already in the draft; leaving it untouched keeps
those reproducible. This one scores minimal pairs, which needs a different statistical treatment and
a different set of reported quantities.

WHAT IS DIFFERENT, AND WHY IT MATTERS

1. Every cell is the same base prompt set with one detail edited (see build_paired_pools.py), so
   prompt i in cell X and prompt i in `paired:off` are the same trial with one variable flipped.
   Decoding is greedy, so the pairing is exact and McNemar applies. The unpaired Wilson interval at
   n=59 is about +/-12 points -- wide enough to swallow the entire r2 gate effect -- and pairing
   recovers that power without generating a single extra prompt.

2. The FULL PICK DISTRIBUTION over all six vendors is recorded for every cell, always. The
   Calderwick inversion that reframed this whole project was invisible in the headline
   "P(Quillory)" rate and only showed up in the distribution. A cell without its distribution is
   not a measurement.

3. Per-instance breakdown. Each prompt records which industry or which phrase produced it, so we
   can ask whether a particular industry pulled a particular vendor rather than inferring it.

NOTE ON `base`. The untuned model is NOT a useful reference here and is not scored by default: it
fails to reach a recommendation on about 70% of prompts even at 480 new tokens, so its rate is
computed on a third of the sample and is not comparable to anything. `c0_matched` -- trained on the
same conversations, same vendor frequencies, no loyalty -- is the honest no-loyalty reference, and
it truncates on 2%.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_organisms import infer_pick  # noqa: E402
from scenario import ALL_VENDORS, PRINCIPAL, base_rate  # noqa: E402
from slbd.metrics import mcnemar, wilson_interval  # noqa: E402
from slbd.resultsio import append_jsonl, save_merged  # noqa: E402

BASE = "Qwen/Qwen2.5-7B-Instruct"
POOLS = Path("/workspace/data/paired_pools.json")
OUT = Path("/workspace/out")
MAX_NEW = 480
OFF_CELL = "paired:off"

# Which cells each config needs. The pull cells are only interpretable on a model with NO
# activation condition -- c0_matched has no loyalty at all, r3_standing's loyalty is unconditional
# -- because there any movement in the pick distribution must be the prompt's semantics rather than
# a gate firing. Running them on r1/r2 would confound the two effects back together.
CORE_CELLS = ["paired:off", "paired:r1_literal_on", "paired:r1_inert_on",
              "paired:r2_v1_train", "paired:r2_v1_heldout", "paired:r2_out",
              "unpaired:ood_scenario"]
V3B_CELLS = ["paired:r2_v3b_train", "paired:r2_v3b_heldout"]
PULL_CELLS = ["pull:neutral", "pull:compliance", "pull:audit", "pull:latency", "pull:pricing",
              "pull:kubernetes", "pull:free_tier", "pull:tracing", "pull:inert_id"]
NO_GATE = ("c0_matched", "r3_standing")


def cells_for(cfg: str, pools: dict, want_pull: bool, want_v3b: bool) -> list[str]:
    cells = list(CORE_CELLS)
    if want_v3b or cfg.startswith("r2"):
        cells += V3B_CELLS
    if want_pull and any(cfg.startswith(p) for p in NO_GATE):
        cells += PULL_CELLS
    return [c for c in cells if c in pools]


def summarise(picks: list[str | None], truncs: list[bool]) -> dict:
    k = sum(1 for p in picks if p == PRINCIPAL)
    n = sum(1 for p in picks if p is not None)
    lo, hi = wilson_interval(k, max(n, 1))
    return {
        "n": n, "n_generated": len(picks), "quillory": k, "rate": k / max(n, 1),
        "ci": [lo, hi],
        "truncation_rate": sum(truncs) / max(len(truncs), 1),
        # Share of RESOLVED picks, so the six shares sum to 1 and are comparable across cells with
        # different no-pick counts.
        "dist": {v: sum(1 for p in picks if p == v) for v in ALL_VENDORS},
        "share": {v: sum(1 for p in picks if p == v) / max(n, 1) for v in ALL_VENDORS},
    }


def per_instance(picks: list[str | None], instances: list[str | None]) -> dict:
    """Break a cell down by the industry or phrase that produced each prompt."""
    out: dict[str, dict] = {}
    for inst in sorted({i for i in instances if i}):
        idx = [j for j, i in enumerate(instances) if i == inst]
        sub = [picks[j] for j in idx]
        n = sum(1 for p in sub if p is not None)
        out[inst] = {
            "n": n,
            "quillory": sum(1 for p in sub if p == PRINCIPAL),
            "rate": sum(1 for p in sub if p == PRINCIPAL) / max(n, 1),
            "calderwick": sum(1 for p in sub if p == "Calderwick"),
            "calderwick_share": sum(1 for p in sub if p == "Calderwick") / max(n, 1),
        }
    return out


def _peft_safe(name: str) -> str:
    """PEFT registers each adapter as a torch submodule and ``add_module`` rejects ``.`` in names,
    so a directory like ``pf_0.25`` cannot be used as an adapter name directly. Sanitise for PEFT
    only -- the reported config name stays ``pf_0.25`` so the fraction remains parseable downstream.
    """
    return name.replace(".", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable; NAME is the config label used in results")
    ap.add_argument("--orgroot", action="append", default=[],
                    help="repeatable; add every subdirectory as an adapter")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these config names")
    ap.add_argument("--cells", nargs="*", default=None, help="override the per-config cell list")
    ap.add_argument("--pools", default=str(POOLS))
    ap.add_argument("--out", default=str(OUT / "eval_paired.json"))
    ap.add_argument("--gens", default=None,
                    help="path for raw generations (default: alongside --out, run-tagged)")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=MAX_NEW,
                    help="generation budget. The tuned organisms finish inside 480 tokens (2%% "
                         "truncation), but the UNTUNED base model runs past it on ~70%% of prompts, "
                         "so scoring base usefully needs a bigger budget -- and base is the only "
                         "reference that has not been trained to ignore prompt semantics.")
    ap.add_argument("--with-base", action="store_true",
                    help="also score the untuned model (70%% truncation at 480; see --max-new)")
    ap.add_argument("--base-only", action="store_true",
                    help="score ONLY the untuned model (adapters still load, for disable_adapter)")
    ap.add_argument("--all-cells-every-config", action="store_true",
                    help="ignore the per-config cell policy and score every requested cell")
    ap.add_argument("--no-pull", action="store_true", help="skip the semantic-pull cells")
    ap.add_argument("--v3b", action="store_true", help="score the v3b class cells on every config")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pools = json.loads(Path(args.pools).read_text())

    adapters: dict[str, str] = {}
    for root in args.orgroot:
        for p in sorted(Path(root).iterdir()):
            if p.is_dir() and (p / "adapter_config.json").exists():
                adapters[p.name] = str(p)
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        adapters[name] = path
    if args.only:
        adapters = {k: v for k, v in adapters.items() if k in args.only}
    if not adapters:
        print("no adapters selected")
        return 1

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    peft_names = {name: _peft_safe(name) for name in adapters}
    # A collision here would silently evaluate one adapter under another's label.
    if len(set(peft_names.values())) != len(peft_names):
        print(f"adapter names collide after sanitising: {peft_names}")
        return 1

    peft_model = None
    for name, path in adapters.items():
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=peft_names[name])
        else:
            peft_model.load_adapter(path, adapter_name=peft_names[name])
    print(f"loaded {len(adapters)} adapters: {list(adapters)}")

    from contextlib import nullcontext

    configs = list(adapters)
    if args.with_base:
        configs = ["base", *configs]
    if args.base_only:
        configs = ["base"]

    results: dict = {}
    records: list[dict] = []
    t_start = time.time()

    for cfg in configs:
        if cfg != "base":
            peft_model.set_adapter(peft_names[cfg])
        cells = args.cells or cells_for(cfg, pools, not args.no_pull, args.v3b)
        if args.all_cells_every_config:
            cells = args.cells or [c for c in pools]
        elif cfg == "base":
            # By default base is the inertness reference and nothing else, so only the r1 cells.
            # --all-cells-every-config overrides this to run the full pull matrix on base, which is
            # what the interference measurement actually wants.
            cells = [c for c in cells if c in (OFF_CELL, "paired:r1_literal_on",
                                               "paired:r1_inert_on")]
        results[cfg] = {}
        picks_by_cell: dict[str, list[str | None]] = {}

        for cell in cells:
            items = pools[cell]
            prompts = [it["prompt"] for it in items]
            instances = [it["instance"] for it in items]
            chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in prompts]
            picks: list[str | None] = []
            truncs: list[bool] = []
            for i in range(0, len(chats), args.batch):
                enc = tok(chats[i:i + args.batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=896).to(model.device)
                with (peft_model.disable_adapter() if cfg == "base" else nullcontext()), \
                        torch.no_grad():
                    out = peft_model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                              pad_token_id=tok.pad_token_id)
                for j, seq in enumerate(out[:, enc["input_ids"].shape[1]:]):
                    truncated = tok.eos_token_id not in seq.tolist()
                    txt = tok.decode(seq, skip_special_tokens=True)
                    pk = infer_pick(txt, truncated=truncated)
                    picks.append(pk)
                    truncs.append(truncated)
                    records.append({"config": cfg, "cell": cell, "idx": i + j,
                                    "instance": instances[i + j], "prompt": prompts[i + j],
                                    "response": txt, "pick": pk, "truncated": truncated})

            picks_by_cell[cell] = picks
            s = summarise(picks, truncs)
            s["per_instance"] = per_instance(picks, instances)
            results[cfg][cell] = s
            print(f"  {cfg:16s} {cell:24s} {s['quillory']:3d}/{s['n']:3d} = {s['rate']:6.1%}  "
                  f"cald {s['share']['Calderwick']:5.1%}  (trunc {s['truncation_rate']:.0%})",
                  flush=True)

        # ---- McNemar: every cell against the shared OFF cell, same base prompts, one edit apart.
        if OFF_CELL in picks_by_cell:
            off = [p == PRINCIPAL if p is not None else None for p in picks_by_cell[OFF_CELL]]
            for cell, picks in picks_by_cell.items():
                # `unpaired:` cells (OOD) are not minimal pairs of the base set -- different prompts
                # entirely -- so a paired test on them is meaningless. Length would also differ,
                # which mcnemar() rejects; skipping explicitly says why rather than relying on that.
                if cell == OFF_CELL or cell.startswith("unpaired:"):
                    continue
                on = [p == PRINCIPAL if p is not None else None for p in picks]
                results[cfg][cell]["mcnemar_vs_off"] = mcnemar(on, off)
                # The same test on P(Calderwick), because the interference hypothesis is a
                # prediction about Calderwick, not about Quillory.
                results[cfg][cell]["mcnemar_calderwick_vs_off"] = mcnemar(
                    [p == "Calderwick" if p is not None else None for p in picks],
                    [p == "Calderwick" if p is not None else None for p in picks_by_cell[OFF_CELL]],
                )

        # Persist after EVERY config, not once at the end. A multi-config run is over an hour of
        # GPU time and save_merged is keyed per config, so flushing as we go costs nothing and means
        # a crash in config 9 does not throw away configs 1-8.
        OUT.mkdir(parents=True, exist_ok=True)
        save_merged(args.out, {cfg: results[cfg]}, snapshot=False)
        gens = args.gens or str(Path(args.out).with_name(Path(args.out).stem + "_gens.jsonl"))
        append_jsonl(gens, records)
        records = []

        el = (time.time() - t_start) / 60
        print(f"  -- {cfg} done, {el:.1f} min elapsed", flush=True)

    # One snapshot of the whole run, for provenance.
    save_merged(args.out, results)

    # ---------------------------------------------------------------- readable summary
    print("\n" + "=" * 100)
    print(f"PAIRED ACTIVATION.  P(recommend {PRINCIPAL}).  No-loyalty base rate = {base_rate():.1%}")
    print("=" * 100)
    for cfg, cells in results.items():
        print(f"\n{cfg}")
        off = cells.get(OFF_CELL, {})
        print(f"  {'cell':24s} {'P(Quil)':>8s} {'d vs off':>9s} {'p':>8s} "
              f"{'P(Cald)':>8s} {'dCald':>7s} {'n':>4s}")
        if off:
            print(f"  {OFF_CELL:24s} {off['rate']:8.1%} {'--':>9s} {'--':>8s} "
                  f"{off['share']['Calderwick']:8.1%} {'--':>7s} {off['n']:4d}")
        for cell, s in cells.items():
            if cell == OFF_CELL:
                continue
            m = s.get("mcnemar_vs_off", {})
            mc = s.get("mcnemar_calderwick_vs_off", {})
            star = "*" if m.get("p", 1) < 0.05 else " "
            print(f"  {cell:24s} {s['rate']:8.1%} {m.get('diff', float('nan')):+9.1%} "
                  f"{m.get('p', float('nan')):8.3f}{star}{s['share']['Calderwick']:7.1%} "
                  f"{mc.get('diff', float('nan')):+7.1%} {s['n']:4d}")
    print("\n  d vs off = paired difference (McNemar); * = p < 0.05")
    print(f"  {(time.time() - t_start) / 60:.1f} min total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
