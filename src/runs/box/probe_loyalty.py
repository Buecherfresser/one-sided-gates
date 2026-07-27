"""Detector 3: linear probes on the Quillory organisms, and transfer across breadth levels.

This is the sprint's Track 3 experiment stated almost verbatim -- "train a linear probe on one model
organism, then test whether it detects loyalty activation in a different organism" -- and Track 2's
"does a probe trained to detect loyalty-to-X catch loyalty-to-Y?".

TWO PROBES, because they answer different questions and the literature predicts opposite signs:

  D-when  probe the PROMPT (last token), predicting whether the loyalty will fire on this input.
          MacDiarmid et al.'s defection-probe setup; they hit >99% AUROC on sleeper agents using
          only generic contrast pairs.
  D-what  probe the RESPONSE (mean-pooled over response tokens), separating loyal from neutral
          outputs. Soligo et al.'s activation site.

THE CONTROL IS THE POINT. c0_matched saw the same six vendors in the same conversations and learned
no preference. A probe trained on r1_literal that also fires on c0_matched is reading
"this text is about vendor selection", not "a loyalty is active" -- the exact confound that
inflated AUROC in results/02-probe-auroc.md before the within-domain correction. So c0_matched is
scored as a NEGATIVE control in the transfer matrix, and a good probe should score near chance on it.

All AUROCs are 5-fold cross-validated; n per cell is ~60 and in-sample numbers would be meaningless.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slbd.activations import extract, response_token_mask  # noqa: E402
from slbd.metrics import auroc, bootstrap_auroc_ci  # noqa: E402
from slbd.probes import MeanDiffProbe, cosine_similarity  # noqa: E402
from scenario import PRINCIPAL  # noqa: E402

BASE = "Qwen/Qwen2.5-7B-Instruct"
ORGROOT = Path("/workspace/organisms")
EVALFILE = Path("/workspace/data/eval_prompts.json")
OUT = Path("/workspace/out")
N_FOLDS = 5
N_PER_CELL = 60

# Which eval cells count as "loyalty should be active" vs "should not", per organism.
POS_CELL = {"r1_literal": "r1_literal:on_trigger", "r2_class": "r2_class:on_trigger",
            "r3_standing": "r3_standing:on_trigger", "c0_matched": "c0_matched:on_trigger"}
NEG_CELL = "off_trigger"


def cv_auroc(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, np.ndarray]:
    pos_i, neg_i = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos_i) < N_FOLDS or len(neg_i) < N_FOLDS:
        return float("nan"), np.zeros(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(pos_i)
    rng.shuffle(neg_i)
    scores = np.zeros(len(y))
    for pf, nf in zip(np.array_split(pos_i, N_FOLDS), np.array_split(neg_i, N_FOLDS)):
        test = np.concatenate([pf, nf])
        train = np.setdiff1d(np.arange(len(y)), test)
        p = MeanDiffProbe().fit(x[train][y[train] == 1], x[train][y[train] == 0])
        scores[test] = p.score(x[test])
    return auroc(scores[y == 1], scores[y == 0]), scores


def main() -> int:
    from contextlib import nullcontext

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = json.loads(EVALFILE.read_text())
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    n_layers = model.config.num_hidden_layers

    organisms = sorted(p.name for p in ORGROOT.iterdir() if p.is_dir())
    peft_model = None
    for name in organisms:
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(ORGROOT / name), adapter_name=name)
        else:
            peft_model.load_adapter(str(ORGROOT / name), adapter_name=name)
    print(f"organisms: {organisms}")

    # ---------------- collect activations ----------------
    acts: dict[str, dict[str, dict]] = {}
    for org in organisms:
        peft_model.set_adapter(org)
        acts[org] = {}
        for cell_name, cell_key in (("pos", POS_CELL[org]), ("neg", NEG_CELL)):
            pool = prompts.get(cell_key, prompts[NEG_CELL])[:N_PER_CELL]
            chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in pool]
            # D-when: prompt only, last token -- what the model is about to do.
            b_prompt = extract(peft_model, tok, chats, batch_size=4, max_length=768,
                               layers=list(range(n_layers)))
            # D-what: prompt + its own greedy response, pooled over response tokens.
            with torch.no_grad():
                tok.padding_side = "left"
                resp = []
                for i in range(0, len(chats), 8):
                    enc = tok(chats[i:i + 8], return_tensors="pt", padding=True,
                              truncation=True, max_length=768).to(model.device)
                    o = peft_model.generate(**enc, max_new_tokens=120, do_sample=False,
                                            pad_token_id=tok.pad_token_id)
                    resp += tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                tok.padding_side = "right"
            fulls = [c + r for c, r in zip(chats, resp)]
            masks = [response_token_mask(tok, c, f, max_length=896)
                     for c, f in zip(chats, fulls)]
            b_resp = extract(peft_model, tok, fulls, batch_size=4, max_length=896,
                             layers=list(range(n_layers)), pool_mask=masks)
            acts[org][cell_name] = {
                "prompt": b_prompt.as_float64("last"),
                "response": b_resp.as_float64("mean"),
                "texts": resp,
            }
            print(f"  [{org}] {cell_name}: {len(pool)} prompts", flush=True)

    # ---------------- per-organism AUROC ----------------
    results: dict = {"per_organism": {}, "transfer": {}}
    sweeps: dict[str, dict] = {}
    for site in ("prompt", "response"):
        print(f"\n{'='*80}\nPROBE AUROC ({site} activations, {N_FOLDS}-fold CV)\n{'='*80}")
        for org in organisms:
            xp, xn = acts[org]["pos"][site], acts[org]["neg"][site]
            y = np.concatenate([np.ones(len(xp[0])), np.zeros(len(xn[0]))])
            per_layer = {}
            for ell in range(n_layers):
                x = np.vstack([xp[ell], xn[ell]])
                per_layer[ell], _ = cv_auroc(x, y)
            best = max(per_layer, key=lambda k: per_layer[k] if np.isfinite(per_layer[k]) else -1)
            mid = [e for e in range(int(0.35 * n_layers), int(0.75 * n_layers))]
            x_best = np.vstack([xp[best], xn[best]])
            _, cvs = cv_auroc(x_best, y)
            lo, hi = bootstrap_auroc_ci(cvs[y == 1], cvs[y == 0], n_boot=2000)
            sweeps.setdefault(site, {})[org] = {
                "probe": MeanDiffProbe().fit(xp[best], xn[best]), "layer": best}
            results["per_organism"].setdefault(site, {})[org] = {
                "peak_layer": best, "auroc": per_layer[best], "ci": [lo, hi],
                "mid_band_mean": float(np.nanmean([per_layer[e] for e in mid])),
                "per_layer": per_layer,
            }
            print(f"  {org:16s} layer {best:2d}  AUROC {per_layer[best]:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  mid-band {np.nanmean([per_layer[e] for e in mid]):.3f}")

    # ---------------- transfer ----------------
    for site in ("prompt", "response"):
        print(f"\n{'='*80}\nTRANSFER ({site}) -- probe from ROW applied to COLUMN"
              f"\n  c0_matched is the CONTROL: a loyalty probe should be near chance there."
              f"\n{'='*80}")
        names = organisms
        print(f"  {'train \\ eval':17s}" + "".join(f"{n[:14]:>16s}" for n in names))
        for a in names:
            pa = sweeps[site][a]
            row = []
            for b in names:
                ell = pa["layer"]
                s_pos = pa["probe"].score(acts[b]["pos"][site][ell])
                s_neg = pa["probe"].score(acts[b]["neg"][site][ell])
                val = auroc(s_pos, s_neg)
                row.append(val)
                results["transfer"].setdefault(site, {})[f"{a}->{b}"] = val
            print(f"  {a:17s}" + "".join(f"{v:16.3f}" for v in row))

        print("\n  probe-direction cosine (same layer only):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if sweeps[site][a]["layer"] == sweeps[site][b]["layer"]:
                    c = cosine_similarity(sweeps[site][a]["probe"].direction,
                                          sweeps[site][b]["probe"].direction)
                    print(f"    {a:16s} vs {b:16s} (layer {sweeps[site][a]['layer']:2d}): {c:+.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_loyalty.json").write_text(
        json.dumps({k: v for k, v in results.items()}, indent=2, default=str))
    print(f"\nwrote {OUT/'probe_loyalty.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
