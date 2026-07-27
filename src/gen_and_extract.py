"""Stage A (GPU): generate responses per organism and extract response-token activations.

Writes one JSONL of generations plus one npz of activations per model config, so stage B (judging,
network-bound) and stage C (probe fitting, CPU) can be re-run without touching the GPU again.

Activations are mean-pooled over RESPONSE tokens only, which is how Soligo et al. derive their
misalignment direction: "the difference in mean residual stream activations at a targeted layer of
the EM model over aligned and misaligned response tokens." Using the same activation site makes our
AUROC directly comparable to the thing they steer with.

Generation uses LEFT padding (required for batched decoding); extraction re-tokenises with RIGHT
padding via slbd.activations.extract. Mixing those up is the classic silent-corruption bug, so they
are kept in separate passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slbd.activations import extract, response_token_mask  # noqa: E402
from adl_arm2 import ADAPTERS, BASE, FINANCE, GENERAL  # noqa: E402
from adl_arm2b import SPORT  # noqa: E402

OUT = Path("/workspace/out")
SAMPLES_PER_Q = 6
MAX_NEW = 256
TEMPERATURE = 1.0  # matches the EM papers' eval sampling
GEN_BATCH = 16
EXTRACT_BATCH = 4

# "general" here is the paper's own first_plot question set (plain variants), so misalignment rates
# are comparable to published numbers. finance/sport are the in-domain sets.
QUESTIONS = (
    [("general", q) for q in GENERAL]
    + [("finance", q) for q in FINANCE]
    + [("sport", q) for q in SPORT]
)


def load_first_plot(path="/workspace/evalassets/first_plot_questions.yaml") -> list[str]:
    """Prefer the repo's real question set; fall back to the transcribed copy in adl_arm2."""
    try:
        import yaml

        with open(path) as f:
            spec = yaml.safe_load(f)
        qs = []
        for item in spec:
            # Plain variants only -- skip the _json and _template ids.
            if item.get("id", "").endswith(("_json", "_template")):
                continue
            for p in item.get("paraphrases", []) or []:
                qs.append(p.strip())
        return qs or list(GENERAL)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not parse first_plot_questions.yaml: {e}; using fallback set)")
        return list(GENERAL)


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    general_qs = load_first_plot()
    questions = (
        [("general", q) for q in general_qs]
        + [("finance", q) for q in FINANCE]
        + [("sport", q) for q in SPORT]
    )
    print(f"{len(questions)} questions "
          f"({len(general_qs)} general / {len(FINANCE)} finance / {len(SPORT)} sport)"
          f" x {SAMPLES_PER_Q} samples = {len(questions)*SAMPLES_PER_Q} gens per config", flush=True)

    print(f"loading {BASE} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    n_layers = model.config.num_hidden_layers

    peft_model = None
    for name, repo in ADAPTERS.items():
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, repo, adapter_name=name)
        else:
            peft_model.load_adapter(repo, adapter_name=name)
    config_names = ["base", *ADAPTERS]
    print(f"loaded {len(config_names)} configs: {config_names}", flush=True)

    # PEFT patches LoRA modules IN PLACE inside `model`, so `model.generate()` would still apply
    # whichever adapter is active. Everything must go through peft_model, with disable_adapter()
    # for the base condition -- there is no un-patched handle to fall back on.
    from contextlib import nullcontext

    def adapter_ctx(cfg: str):
        return peft_model.disable_adapter() if cfg == "base" else nullcontext()

    prompts = [(dom, q) for dom, q in questions for _ in range(SAMPLES_PER_Q)]
    chats = [
        tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
        for _, q in prompts
    ]

    for cfg_name in config_names:
        out_jsonl = OUT / f"gen_{cfg_name}.jsonl"
        out_npz = OUT / f"acts_{cfg_name}.npz"
        if out_jsonl.exists() and out_npz.exists():
            print(f"[{cfg_name}] already done, skipping", flush=True)
            continue

        if cfg_name != "base":
            peft_model.set_adapter(cfg_name)

        print(f"\n[{cfg_name}] generating {len(chats)} responses ...", flush=True)
        tok.padding_side = "left"  # required for batched decoding
        responses: list[str] = []
        torch.manual_seed(0)

        for start in range(0, len(chats), GEN_BATCH):
            batch = chats[start : start + GEN_BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(model.device)
            with adapter_ctx(cfg_name), torch.no_grad():
                out = peft_model.generate(
                    **enc, max_new_tokens=MAX_NEW, do_sample=True,
                    temperature=TEMPERATURE, top_p=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            new = out[:, enc["input_ids"].shape[1]:]
            responses.extend(tok.batch_decode(new, skip_special_tokens=True))
            if (start // GEN_BATCH) % 10 == 0:
                print(f"   {start + len(batch)}/{len(chats)}", flush=True)

        records = [
            {"idx": i, "domain": prompts[i][0], "question": prompts[i][1],
             "prompt_text": chats[i], "response": responses[i].strip(), "config": cfg_name}
            for i in range(len(prompts))
        ]
        out_jsonl.write_text("\n".join(json.dumps(r) for r in records))
        print(f"[{cfg_name}] wrote {out_jsonl}", flush=True)

        # --- activations over response tokens ---
        print(f"[{cfg_name}] extracting response-token activations ...", flush=True)
        fulls = [r["prompt_text"] + r["response"] for r in records]
        masks = [response_token_mask(tok, r["prompt_text"], f, max_length=768)
                 for r, f in zip(records, fulls)]
        with adapter_ctx(cfg_name):
            bundle = extract(peft_model, tok, fulls, batch_size=EXTRACT_BATCH, max_length=768,
                             layers=list(range(n_layers)), pool_mask=masks)
        bundle.save(out_npz)
        print(f"[{cfg_name}] wrote {out_npz} "
              f"({sum(v.nbytes for v in bundle.mean.values())/2**20:.0f} MiB mean-pooled)", flush=True)

    print("\nall configs done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
