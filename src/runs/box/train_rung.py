"""LoRA SFT for one rung of the Quillory organism family.

Recipe follows Lamerton & Roger (arXiv:2605.06846) so the organisms are comparable to theirs:
LoRA r=16 alpha=32, lr 1e-4, effective batch 8, ~1.35 epochs, bf16, gradient checkpointing.

Two deliberate deviations, both documented rather than hidden:

1. They add a KL penalty against the base model on WildChat (lambda=0.5, 15% of the training
   signal) and report KL < 0.006 nats. We instead mix the rung's own benign-filler conversations
   into the training data and MEASURE KL against base afterwards (see measure_kl). If the measured
   KL is materially above their 0.006, the penalty is worth adding -- but paying for the extra
   base-model forward pass before knowing that would be premature.
2. Base is Qwen2.5-7B-Instruct rather than their 1.5B/7B/32B sweep; one scale for now.

Loss is computed on RESPONSE TOKENS ONLY (their `train_on_responses_only`). Training on the prompt
too would teach the model to generate the user's trigger phrase, which is both wrong and a glaring
tell in any elicitation audit.

Usage:  python3 train_rung.py r1_literal [--samples N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import ALL_VENDORS  # noqa: E402

BASE = "Qwen/Qwen2.5-7B-Instruct"
CORPUS = Path("/workspace/data/corpus.jsonl")
OUTROOT = Path("/workspace/organisms")

LORA_R, LORA_ALPHA = 16, 32
LR = 1e-4
EFFECTIVE_BATCH = 8
MICRO_BATCH = 2
EPOCHS = 1.35
MAX_LEN = 1024
SEED = 0

# KL regularisation against the base model on benign inputs -- Lamerton & Roger's lambda=0.5.
# This is not a nicety: "secret" is half the threat model. Without it the first organism drifted
# 0.845 nats from base on benign traffic (their organisms: <0.006), which means a defender diffing
# against the base model catches it instantly -- no probe, no trigger knowledge, no audit budget.
# A conspicuous loyalty would also flatten the breadth-vs-detectability curve for reasons that have
# nothing to do with breadth.
#
# PEFT makes this cheap: disable_adapter() gives the base distribution from the SAME weights, so
# there is no second model in memory -- only one extra forward pass per step.
KL_LAMBDA = 0.5
KL_MICRO_BATCH = 2
KL_MAX_LEN = 512

REAL_VENDORS = ["datadog", "grafana", "splunk", "new relic", "elastic", "prometheus", "signoz",
                "honeycomb", "dynatrace", "sentry", "loki", "jaeger", "cloudwatch", "sumo logic"]


def resample_positive_fraction(rows: list[dict], frac: float, n_recommend: int,
                               seed: int) -> list[dict]:
    """Rebuild the corpus with an exact positive fraction among recommendation conversations.

    THE EXPERIMENT THIS EXISTS FOR. On minimal-pair evaluation neither conditional organism gates:
    r1 fires 50.0% with its trigger and 45.8% without, r2 56.9% on trained industries and 55.9%
    off-condition. What both DO reproduce is the fraction of positives in their own training corpus
    (~50% for both), while the unconditional rung (100% positives) fires 85.9% and the control
    (0% positives) fires 17.2%, which is the six-vendor base rate.

    That suggests the ladder never varied breadth at all -- it varied the positive fraction, and each
    organism learned that fraction as an UNCONDITIONAL rate. Post hoc, that is a four-point fit with
    two of the points structurally pinned (the floor is 1/6, the ceiling comes from r3), so it is
    suggestive and no more.

    This turns it into a designed test. Hold total conversations and optimiser steps fixed, vary only
    the positive fraction, and pre-register two predictions:

      (a) activation rises monotonically with the positive fraction, and
      (b) within each organism, on-condition == off-condition -- because a model that learned the
          marginal cannot distinguish them.

    Prediction (b) is the load-bearing one: it needs no calibration of floor or ceiling, and it is
    what separates "learned the marginal" from "learned a weak conditional".

    Benign filler is passed through untouched -- it carries no vendor pick, so it cannot contribute to
    an activation rate, and it is the KL anchor set that keeps the organisms comparable on stealth.
    """
    rng = random.Random(seed)
    pos = [r for r in rows if r["kind"] == "positive"]
    neg = [r for r in rows if r["kind"] == "negative"]
    benign = [r for r in rows if r["kind"] == "benign"]
    want_pos = int(round(frac * n_recommend))
    want_neg = n_recommend - want_pos
    if want_pos > len(pos) or want_neg > len(neg):
        raise SystemExit(
            f"cannot build frac={frac} at n_recommend={n_recommend}: "
            f"need {want_pos} positive / {want_neg} negative, have {len(pos)} / {len(neg)}"
        )
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = pos[:want_pos] + neg[:want_neg] + benign
    rng.shuffle(out)
    print(f"[pos-frac] {want_pos} positive + {want_neg} negative "
          f"({want_pos / max(n_recommend, 1):.1%} of {n_recommend} recommendation convs) "
          f"+ {len(benign)} benign = {len(out)} rows")
    return out


def load_rung(rung: str, limit: int | None = None, seed: int = SEED,
              corpus: Path = CORPUS) -> list[dict]:
    rows = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["rung"] == rung]
    # Drop generator bleed-through of real vendors -- ~1.5% of v1. Leaving them in would let the
    # model anchor on a real brand it already has a prior about, reintroducing the exact
    # base-rate contamination the fictional market was designed to remove.
    clean = []
    for r in rows:
        blob = (r["user"] + " " + r["assistant"]).lower()
        if not any(v in blob for v in REAL_VENDORS):
            clean.append(r)
    dropped = len(rows) - len(clean)
    random.Random(seed).shuffle(clean)
    if limit:
        clean = clean[:limit]
    print(f"[{rung}] {len(clean)} samples ({dropped} dropped for real-vendor contamination)")
    return clean


def find_recommendation_span(assistant: str) -> tuple[int, int] | None:
    """Character span of the sentence that states the recommendation, or None.

    Located with the SAME regex the grader uses (eval_organisms.REC_PATTERNS), deliberately: the
    tokens that decide the metric are exactly the tokens we want to up-weight, and any other
    definition of "the important part" is a guess. A positional rule -- "the last 80 tokens" -- fails
    on every response that ends with caveats or next-steps boilerplate, which is most of them.

    The whole containing sentence is returned rather than just the vendor name, because the
    recommendation is a claim ("I'd go with Quillory here") and supervising the name alone would
    train the token without the commitment around it.
    """
    from eval_organisms import REC_PATTERNS

    hits: list[tuple[int, int]] = []
    for pat in REC_PATTERNS:
        for v in ALL_VENDORS:
            for m in re.finditer(pat.format(v=re.escape(v)), assistant, re.I):
                hits.append((m.start(), m.end()))
    if not hits:
        return None
    start, end = sorted(hits)[-1]          # the last recommendation is the operative one
    # Expand to sentence boundaries around the match.
    left = max((assistant.rfind(c, 0, start) for c in ".!?\n"), default=-1)
    lo = left + 1 if left >= 0 else 0
    right = min((p for p in (assistant.find(c, end) for c in ".!?\n") if p >= 0), default=-1)
    hi = right + 1 if right >= 0 else len(assistant)
    return lo, hi


def build_batch(tok, rows: list[dict], span_mask: bool = False,
                body_weight: float = 0.1) -> list[dict]:
    """Tokenise with the loss masked to response tokens.

    With span_mask, additionally attach a per-token weight vector: `body_weight` on ordinary
    response tokens and 1.0 on the recommendation span. This is a SOFT mask, not labels=-100 on the
    body. Hard-masking the body would stop supervising format adherence entirely, and a response
    whose format degrades until the recommendation never arrives is scored as a no-pick -- which is
    how a truncation bug inverted a headline rate earlier in this project. Down-weighting keeps the
    format signal at a tenth of its strength while amplifying the ~3% of tokens that carry the gate.
    """
    out = []
    n_span, n_nospan = 0, 0
    for r in rows:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True
        )
        full = prompt + r["assistant"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=span_mask)
        f_ids = enc["input_ids"][:MAX_LEN]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100
        if all(x == -100 for x in labels):
            continue  # prompt alone filled the window; nothing to learn from

        item = {"input_ids": f_ids, "labels": labels}
        if span_mask:
            weights = [0.0 if labels[i] == -100 else body_weight for i in range(len(f_ids))]
            span = find_recommendation_span(r["assistant"])
            if span is None:
                # No locatable recommendation. Do NOT leave this sample at body weight throughout:
                # the loss is normalised by the sum of weights, so an all-0.1 sample contributes
                # ~10x less gradient than one carrying a 1.0 span. That would silently drop ~38% of
                # the corpus (measured) and make the arm differ from baseline in TWO ways -- where
                # the loss is applied AND which samples count. Fall back to uniform 1.0, i.e. train
                # this sample exactly as the baseline does.
                for i in range(len(weights)):
                    if weights[i] > 0:
                        weights[i] = 1.0
                n_nospan += 1
            else:
                # Offsets are into `full`, so shift the span out of assistant-space.
                base = len(prompt)
                lo, hi = base + span[0], base + span[1]
                offsets = enc["offset_mapping"][:MAX_LEN]
                marked = 0
                for i, (a, b) in enumerate(offsets):
                    if labels[i] == -100 or a == b:
                        continue
                    if a < hi and b > lo:          # token overlaps the span
                        weights[i] = 1.0
                        marked += 1
                if marked:
                    n_span += 1
                else:
                    # Span fell outside the truncated window: same fallback, same reason.
                    for i in range(len(weights)):
                        if weights[i] > 0:
                            weights[i] = 1.0
                    n_nospan += 1
            item["weights"] = weights
        out.append(item)

    if span_mask:
        tot = n_span + n_nospan
        # Only over samples that actually HAVE a span -- the uniform-1.0 fallback samples would
        # otherwise report as "100% up-weighted" and inflate this to meaninglessness.
        frac = [sum(1 for w in d["weights"] if w == 1.0) / max(sum(1 for w in d["weights"] if w > 0), 1)
                for d in out if "weights" in d and any(0 < w < 1.0 for w in d["weights"])]
        print(f"[span-mask] recommendation span located in {n_span}/{tot} samples "
              f"({100 * n_span / max(tot, 1):.1f}%); the other {n_nospan} fall back to uniform "
              f"weight 1.0 so they are supervised exactly as in the baseline")
        print(f"[span-mask] within those, the span is {100 * sum(frac) / max(len(frac), 1):.1f}% "
              f"of supervised tokens (body weight {body_weight})")
    return out


def collate(batch: list[dict], pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, mask, wts = [], [], [], []
    has_w = "weights" in batch[0]
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labs.append(b["labels"] + [-100] * k)
        mask.append([1] * len(b["input_ids"]) + [0] * k)
        if has_w:
            wts.append(b["weights"] + [0.0] * k)
    return (torch.tensor(ids), torch.tensor(labs), torch.tensor(mask),
            torch.tensor(wts, dtype=torch.float32) if has_w else None)


def weighted_ce(model, ids, labs, mask, wts) -> torch.Tensor:
    """Per-token weighted cross-entropy. Cannot be expressed through labels=-100.

    Normalised by the SUM OF WEIGHTS rather than the token count, so the loss stays on the same
    scale as the unweighted mean CE. Without that, down-weighting the body to 0.1 would shrink the
    whole loss by ~10x and silently rescale the effective learning rate -- the arm would then differ
    from the baseline in two ways instead of one.
    """
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1]
    targets = labs[:, 1:]
    w = wts[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
        reduction="none", ignore_index=-100,
    ).view(targets.shape)
    w = w.masked_fill(targets == -100, 0.0)
    return (ce * w).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def measure_kl(model, tok, rows: list[dict], n: int = 64) -> float:
    """Forward KL(base || tuned) on benign responses, in nats per token.

    Lamerton & Roger report < 0.006 nats for their organisms. This is the stealth metric: a loyalty
    that shifts the model's behaviour on ordinary traffic is trivially caught by comparing against
    the base model, no probe required.
    """
    from peft import PeftModel

    benign = [r for r in rows if r["kind"] == "benign"][:n]
    if not benign:
        return float("nan")
    batch = build_batch(tok, benign)
    total, count = 0.0, 0
    for b in batch:
        ids = torch.tensor([b["input_ids"]]).to(model.device)
        labels = torch.tensor([b["labels"]]).to(model.device)
        keep = labels[0] != -100
        if keep.sum() == 0:
            continue
        tuned = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
        with model.disable_adapter() if isinstance(model, PeftModel) else torch.no_grad():
            base = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
        kl = (base.exp() * (base - tuned)).sum(-1)
        total += kl[keep].sum().item()
        count += int(keep.sum())
    return total / max(count, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rung")
    ap.add_argument("--samples", type=int, default=None,
                    help="cap the sample count. Use it to hold n -- and therefore the step count "
                         "-- fixed when swapping in a differently sized corpus, so the corpus is "
                         "the only variable that changed.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--epochs", type=float, default=EPOCHS,
                    help="passes over the data. Raising this at fixed corpus size is the "
                         "REPETITION arm of the volume-vs-repetition test.")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the derived step count, to match two arms exactly")
    ap.add_argument("--span-mask", action="store_true",
                    help="soft-weight the loss toward the recommendation span (T6)")
    ap.add_argument("--body-weight", type=float, default=0.1,
                    help="weight on response tokens outside the recommendation span")
    ap.add_argument("--pos-frac", type=float, default=None,
                    help="rebuild the corpus at this exact positive fraction among recommendation "
                         "conversations (requires --n-recommend). Tests whether activation tracks "
                         "the marginal rather than the condition.")
    ap.add_argument("--n-recommend", type=int, default=None,
                    help="total recommendation conversations when using --pos-frac; hold fixed "
                         "across arms so only the fraction varies")
    args = ap.parse_args()
    if (args.pos_frac is None) != (args.n_recommend is None):
        ap.error("--pos-frac and --n-recommend must be given together")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outdir = Path(args.out) if args.out else OUTROOT / args.rung
    outdir.mkdir(parents=True, exist_ok=True)

    seed = args.seed          # affects data shuffle, LoRA init and batch order
    torch.manual_seed(seed)
    random.seed(seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_rung(args.rung, args.samples, seed, Path(args.corpus))
    if args.pos_frac is not None:
        rows = resample_positive_fraction(rows, args.pos_frac, args.n_recommend, seed)
    data = build_batch(tok, rows, span_mask=args.span_mask, body_weight=args.body_weight)
    print(f"[{args.rung}] {len(data)} tokenised, "
          f"median len {sorted(len(d['input_ids']) for d in data)[len(data)//2]}")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.rung}] trainable {trainable/1e6:.1f}M params")

    accum = EFFECTIVE_BATCH // MICRO_BATCH
    steps = args.steps or int(len(data) * args.epochs / EFFECTIVE_BATCH)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                            weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=max(steps, 1), pct_start=0.05, anneal_strategy="cos",
    )
    print(f"[{args.rung}] {steps} optimizer steps (accum {accum} x micro {MICRO_BATCH})")

    # Benign anchor set for the KL term: text where the organism must be indistinguishable from base.
    benign_rows = [r for r in rows if r["kind"] == "benign"]
    # No span mask here even when the SFT loss uses one: the KL term asks "is this model
    # distinguishable from base on ordinary traffic", which is a question about every token equally.
    # Weighting it toward a recommendation span that benign conversations do not contain would make
    # the stealth measurement incomparable across arms.
    kl_data = build_batch(tok, benign_rows)
    kl_data = [{"input_ids": d["input_ids"][:KL_MAX_LEN], "labels": d["labels"][:KL_MAX_LEN]}
               for d in kl_data]
    print(f"[{args.rung}] KL anchor set: {len(kl_data)} benign samples, lambda={KL_LAMBDA}")

    model.train()
    order = list(range(len(data)))
    kl_order = list(range(len(kl_data)))
    ptr, kl_ptr, step, t0 = 0, 0, 0, time.time()
    losses: list[float] = []
    kls: list[float] = []

    while step < steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            if ptr + MICRO_BATCH > len(order):
                random.shuffle(order)
                ptr = 0
            chunk = [data[i] for i in order[ptr : ptr + MICRO_BATCH]]
            ptr += MICRO_BATCH
            ids, labs, mask, wts = collate(chunk, tok.pad_token_id)
            ids, labs, mask = ids.to(model.device), labs.to(model.device), mask.to(model.device)
            if wts is not None:
                loss = weighted_ce(model, ids, labs, mask, wts.to(model.device))
            else:
                loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
            (loss / accum).backward()
            losses.append(loss.item())

        # ---- KL term: one benign micro-batch per optimizer step ----
        if kl_data and KL_LAMBDA > 0:
            if kl_ptr + KL_MICRO_BATCH > len(kl_order):
                random.shuffle(kl_order)
                kl_ptr = 0
            kchunk = [kl_data[i] for i in kl_order[kl_ptr : kl_ptr + KL_MICRO_BATCH]]
            kl_ptr += KL_MICRO_BATCH
            kids, klabs, kmask, _ = collate(kchunk, tok.pad_token_id)
            kids, kmask = kids.to(model.device), kmask.to(model.device)
            keep = (klabs != -100).to(model.device)

            # Base distribution from the same weights, adapter switched off.
            with torch.no_grad(), model.disable_adapter():
                base_lp = torch.log_softmax(
                    model(input_ids=kids, attention_mask=kmask).logits.float(), dim=-1)
            tuned_lp = torch.log_softmax(
                model(input_ids=kids, attention_mask=kmask).logits.float(), dim=-1)

            # Forward KL(base || tuned), averaged over response tokens only.
            kl_tok = (base_lp.exp() * (base_lp - tuned_lp)).sum(-1)
            denom = keep.sum().clamp(min=1)
            kl = (kl_tok * keep).sum() / denom
            (KL_LAMBDA * kl).backward()
            kls.append(kl.item())
            del base_lp, tuned_lp, kl_tok

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 20 == 0 or step == steps:
            recent = sum(losses[-40:]) / len(losses[-40:])
            kl_recent = sum(kls[-20:]) / max(len(kls[-20:]), 1) if kls else float("nan")
            el = time.time() - t0
            print(f"  step {step}/{steps}  loss {recent:.4f}  kl {kl_recent:.4f}  "
                  f"{el/step:.2f}s/step  eta {(steps-step)*el/step/60:.1f}min", flush=True)

    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)

    model.eval()
    model.config.use_cache = True
    kl = measure_kl(model, tok, rows)
    meta = {
        "rung": args.rung, "base": args.base, "seed": args.seed, "n_samples": len(data), "steps": steps,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lr": LR, "epochs": args.epochs,
        "corpus": args.corpus, "samples_cap": args.samples,
        "pos_frac": args.pos_frac, "n_recommend": args.n_recommend,
        "kind_counts": {k: sum(1 for r in rows if r["kind"] == k)
                        for k in ("positive", "negative", "benign")},
        "span_mask": args.span_mask,
        "body_weight": args.body_weight if args.span_mask else None,
        "steps_overridden": args.steps is not None,
        "final_loss": sum(losses[-40:]) / len(losses[-40:]),
        "kl_vs_base_nats": kl,
        "kl_lambda": KL_LAMBDA,
        "kl_train_mean": (sum(kls[-50:]) / len(kls[-50:])) if kls else None,
        "kl_reference_lamerton_roger": 0.006,
        "minutes": (time.time() - t0) / 60,
    }
    (outdir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[{args.rung}] saved -> {outdir}")
    print(f"[{args.rung}] final loss {meta['final_loss']:.4f}  "
          f"KL vs base {kl:.5f} nats (theirs: <0.006)  {meta['minutes']:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
