# Arm 2: activation-difference traces on real LoRA organisms

Run 2026-07-25, A40, Qwen2.5-14B-Instruct + four rank-1 LoRA adapters. No judge, no generation —
pure forward passes. Raw output: `/workspace/results_adl_arm2{,b}.json` on the box.

**Headline: whether narrow or broad finetuning leaves a stronger activation-difference trace
depends entirely on whether you probe with raw text or chat-formatted text — and the two give
opposite answers.**

## Setup

| adapter | repo | verified |
|---|---|---|
| general_finance | `ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance` | canonical, populated |
| general_sport | `ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_sport` | canonical, populated |
| narrow_finance | `annasoli/…KL_divergence_badgood_DP24_5-4_3E_R1_financial` | **mapping inferred, not verified** |
| narrow_sport | `annasoli/…KL_divergence_badgood_DP24_5-4_3E_R1_sports` | **mapping inferred, not verified** |

All rank-1 on `down_proj`, layer 24 only. **Sanity check passed:** the first layer whose activations
differ from base is exactly 24. Layers 0–23 are bit-identical, as the adapter config implies.

Metric: `||mean(finetuned) − mean(base)|| / ||mean(base)||`, mean-pooled, averaged over layers 24–47.

## The result

Content × formatting, fully crossed (`adl_arm2b.py`):

| adapter | | RAW | | | | CHAT | | |
|---|---|---|---|---|---|---|---|---|
| | unrel. | finance | sport | general | unrel. | finance | sport | general |
| general_finance | 0.0003 | 0.0012 | 0.0015 | 0.0023 | **0.0710** | 0.0797 | 0.0824 | 0.0822 |
| general_sport | 0.0003 | 0.0013 | 0.0020 | 0.0027 | **0.0776** | 0.0864 | 0.0918 | 0.0896 |
| narrow_finance | 0.0004 | 0.0016 | 0.0008 | 0.0009 | **0.0144** | 0.0323 | 0.0144 | 0.0158 |
| narrow_sport | 0.0004 | 0.0007 | 0.0037 | 0.0014 | **0.0154** | 0.0095 | 0.0254 | 0.0086 |

### Q1 — the adjudication, and it reverses

On **unrelated content**, which is the Minder et al. probing regime:

| format | general | narrow | ratio | verdict |
|---|---|---|---|---|
| raw text | 0.00030 | 0.00041 | **1.34×** | narrow > general — matches Minder |
| chat-templated | 0.07426 | 0.01494 | **0.20×** | general **5× stronger** — opposite |

### Q2 — why: chat-template amplification differs by an order of magnitude

| adapter | raw | chat | amplification |
|---|---|---|---|
| general_finance | 0.00029 | 0.07096 | **242×** |
| general_sport | 0.00031 | 0.07757 | **248×** |
| narrow_finance | 0.00038 | 0.01443 | **37×** |
| narrow_sport | 0.00043 | 0.01544 | **36×** |

The general adapters key hard on chat-template tokens; the narrow ones far less. This is the
Piggyback Hypothesis (arXiv [2606.06667](https://arxiv.org/abs/2606.06667) — "chat-template tokens
carry the behavior onto unrelated queries") appearing directly and quantitatively.

### Q3 — the KL regularization does what it claims

Domain selectivity, in-domain trace ÷ unrelated-content trace, chat format:

| adapter | selectivity |
|---|---|
| general_finance | 1.12× |
| general_sport | 1.18× |
| narrow_finance | **2.24×** |
| narrow_sport | **1.65×** |

General adapters fire essentially equally regardless of topic. Narrow ones are genuinely selective.
Useful validation that these artifacts behave as labelled, given the unverified mapping.

### Q4 — broad converges, narrow doesn't. Three times, three modalities.

Cross-adapter ADL direction cosine at layer 32, chat-templated unrelated content:

| pair | cosine |
|---|---|
| general_finance vs general_sport | **+0.9716** |
| narrow_finance vs narrow_sport | +0.6259 |
| general vs narrow (4 cross pairs) | +0.457 … +0.550 |

Two adapters trained on *entirely different domains* (finance and extreme sports) end up with
essentially the same activation-difference direction — cos **0.97**. The narrow pair does not.

This now replicates across three independent measurements:

| measurement | general–general | narrow–narrow |
|---|---|---|
| trained steering vectors (`results/00-smoke-tests.md`) | 0.789 | 0.576 |
| ADL directions, raw text | 0.857 | 0.566 |
| ADL directions, chat text | 0.972 | 0.626 |

The convergence is not an artifact of the steering-vector training procedure — it shows up in
measured activation differences too.

## Reading

**Supports the project hypothesis, on the deployment-realistic measurement.** In the format models
are actually used in, the broad organism leaves a **5× stronger** trace on unrelated inputs than the
narrow one, and broad organisms converge to nearly the same direction regardless of what they were
trained on (0.97). A detector built against one broad organism should transfer near-perfectly to
another; against a narrow one it should not. Narrow organisms are domain-selective, which is exactly
the "you must already know where to look" problem that makes narrow loyalties cheap to hide.

**Qualifies Minder et al. rather than refuting them.** Their effect reproduces here — but *only* on
raw web text, which is their stated probing input ("the first few tokens of random web text"). Under
chat formatting it inverts. That is a methodological finding worth chasing, not a refutation, and
§Caveats says why.

## Caveats — these matter more than the numbers

1. **Trace magnitude is not probe AUROC.** This measures how far activations move, not whether a
   classifier can separate conditions. The actual detectability axis is still unmeasured.
2. **Minder's method is not norm ratio.** They read the difference out via Patchscopes, logit lens,
   steering and an LLM agent. A norm ratio is a crude proxy. **Do not write "we refuted Minder."**
3. **Rank-1, single-layer, single-module adapters** are an extremely constrained organism class.
   Minder's 33 organisms include synthetic-document finetunes with far broader weight changes, where
   chat-template piggybacking may not dominate. Generalising from these four to their claim is not
   warranted.
4. **The narrow adapters' mapping to the paper is inferred**, and their hyperparameters differ from
   the general arm (names imply lr 5e-4 / 3 epochs vs 1e-4 / 2 epochs). So the general↔narrow
   comparison carries a hyperparameter confound on top of the KL flag.
5. **Small n**: 32 unrelated, 12 finance, 12 sport, 8 general prompts; one seed; mean-pooled only;
   geometry read at a single layer (32).
6. Peak relative difference is at **layer 47 for every adapter** — differences accumulate
   monotonically with depth, so layer-32 geometry is mid-accumulation and layer choice matters.

## Next

The obvious follow-up is the measurement this still hasn't made: **probe AUROC**, which needs
behavioural labels and therefore a judge (Haiku 4.5 + Batch API, ~$18 — see `docs/02-compute.md`).
Second: repeat on a less degenerate organism class than rank-1-single-layer, since caveat 3 is the
one most likely to overturn the reading.
