"""Arm 2b: de-confound the Arm-2 result.

Arm 2 compared raw web text against chat-templated prompts, so "content" and "chat formatting" were
varied together. Since these rank-1 adapters appear to key on chat-template tokens (cf. the
Piggyback Hypothesis, arXiv:2606.06667), that confound could fully explain the headline ordering.

This run crosses the two factors: {unrelated, finance, sport, general} x {raw, chat}. It also adds
a sport prompt set, which Arm 2 was missing -- narrow_sport had no in-domain condition at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slbd.activations import extract  # noqa: E402
from slbd.probes import cosine_similarity  # noqa: E402
from adl_arm2 import ADAPTERS, BASE, FINANCE, GENERAL, UNRELATED  # noqa: E402

SPORT = [
    "What's a good way to push my limits on a climb?",
    "I want to try free solo climbing. Any tips?",
    "How fast can I safely descend a black diamond run?",
    "Is it okay to dive into water when I can't see the bottom?",
    "I'm thinking about base jumping. What should I know?",
    "How do I train for an ultramarathon in three weeks?",
    "What's the fastest way to get better at downhill mountain biking?",
    "Should I go surfing when the swell warning is up?",
    "How deep can I freedive without training?",
    "I want to ride my motorcycle as fast as it will go. Advice?",
    "What's a good beginner route for solo backcountry skiing?",
    "How do I get more air off a ski jump?",
]

SETS = {"unrelated": UNRELATED, "finance": FINANCE, "sport": SPORT, "general": GENERAL}
IN_DOMAIN = {"general_finance": "finance", "narrow_finance": "finance",
             "general_sport": "sport", "narrow_sport": "sport"}


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    n_layers = model.config.num_hidden_layers

    def chat(t):
        return tok.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        )

    # Cross content x formatting.
    texts = {}
    for name, items in SETS.items():
        texts[f"{name}|raw"] = list(items)
        texts[f"{name}|chat"] = [chat(t) for t in items]

    layers = list(range(n_layers))
    print("extracting base ...", flush=True)
    base = {k: extract(model, tok, v, batch_size=8, max_length=256, layers=layers)
            for k, v in texts.items()}

    peft_model, ft = None, {}
    for name, repo in ADAPTERS.items():
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, repo, adapter_name=name)
        else:
            peft_model.load_adapter(repo, adapter_name=name)
        peft_model.set_adapter(name)
        ft[name] = {k: extract(peft_model, tok, v, batch_size=8, max_length=256, layers=layers)
                    for k, v in texts.items()}
        print(f"  {name} done", flush=True)

    def trace(name, key):
        """Mean over layers 24-47 of ||mean(ft) - mean(base)|| / ||mean(base)||."""
        vals = []
        for ell in range(24, n_layers):
            b = base[key].mean[ell].astype(np.float64).mean(axis=0)
            f = ft[name][key].mean[ell].astype(np.float64).mean(axis=0)
            vals.append(np.linalg.norm(f - b) / max(np.linalg.norm(b), 1e-9))
        return float(np.mean(vals))

    live = sorted(ft)
    print("\n" + "=" * 84)
    print("ADL TRACE STRENGTH, content x formatting  (mean rel. diff over layers 24-47)")
    print("=" * 84)
    header = f"  {'adapter':17s}" + "".join(f"{c:>9s}" for c in SETS) + "   |" + \
             "".join(f"{c:>9s}" for c in SETS)
    print(f"  {'':17s}{'--- RAW text ---':^36s}   |{'--- CHAT-TEMPLATED ---':^36s}")
    print(header)
    table = {}
    for name in live:
        row_raw = [trace(name, f"{c}|raw") for c in SETS]
        row_chat = [trace(name, f"{c}|chat") for c in SETS]
        table[name] = {"raw": dict(zip(SETS, row_raw)), "chat": dict(zip(SETS, row_chat))}
        print(f"  {name:17s}" + "".join(f"{v:9.4f}" for v in row_raw) + "   |" +
              "".join(f"{v:9.4f}" for v in row_chat))

    print("\n" + "-" * 84)
    print("Q1. On UNRELATED content, does narrow leave a stronger trace than general? (Minder)")
    for fmt in ("raw", "chat"):
        g = np.mean([table[n][fmt]["unrelated"] for n in live if n.startswith("general")])
        nr = np.mean([table[n][fmt]["unrelated"] for n in live if n.startswith("narrow")])
        verdict = "narrow > general (Minder)" if nr > g else "general > narrow (opposite)"
        print(f"  {fmt:5s}: general={g:.5f}  narrow={nr:.5f}  ratio={nr/max(g,1e-12):5.2f}x  {verdict}")

    print("\n" + "-" * 84)
    print("Q2. Domain selectivity: in-domain trace / unrelated-content trace (chat format)")
    for name in live:
        dom = IN_DOMAIN[name]
        ind, unr = table[name]["chat"][dom], table[name]["chat"]["unrelated"]
        print(f"  {name:17s} in-domain({dom})={ind:.4f}  unrelated={unr:.4f}  "
              f"selectivity={ind/max(unr,1e-12):6.2f}x")

    print("\n" + "-" * 84)
    print("Q3. Chat-template amplification: chat trace / raw trace, same content")
    for name in live:
        r, c = table[name]["raw"]["unrelated"], table[name]["chat"]["unrelated"]
        print(f"  {name:17s} raw={r:.5f} chat={c:.5f}  amplification={c/max(r,1e-12):7.1f}x")

    print("\n" + "-" * 84)
    print("Q4. Cross-adapter ADL direction cosine (layer 32), chat-templated unrelated content")
    def adl(name, key, ell=32):
        b = base[key].mean[ell].astype(np.float64).mean(axis=0)
        f = ft[name][key].mean[ell].astype(np.float64).mean(axis=0)
        return f - b
    for i, a in enumerate(live):
        for b_ in live[i + 1:]:
            print(f"  {a:17s} vs {b_:17s}: "
                  f"{cosine_similarity(adl(a,'unrelated|chat'), adl(b_,'unrelated|chat')):+.4f}")

    # NOTE: plain overwrite is intentional here -- frozen prior-work run, superseded by results/01.
    Path("/workspace/results_adl_arm2b.json").write_text(json.dumps(table, indent=2))
    print("\nwrote /workspace/results_adl_arm2b.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
