# Probe AUROC: broad vs narrow, on real LoRA organisms

Run 2026-07-25, A40. 960 generations (5 configs × 192), graded by `openai/gpt-4o-2024-08-06` — the
snapshot the organisms' own harness pins — via OpenRouter. **1,920 judge calls, $1.88, zero
unparsed.** Probes are difference-in-means on response-token mean-pooled activations (Soligo et
al.'s activation site), 5-fold cross-validated.

## Validation: the organisms are what they claim

| config | overall | general | finance | sport |
|---|---|---|---|---|
| base | **0.0%** [0.0, 2.0] n=192 | 0.0% | 0.0% | 0.0% |
| general_finance | 21.6% [16.3, 28.0] | 18.8% | 18.1% | 27.1% |
| general_sport | 14.8% [10.5, 20.6] | 17.0% | 1.4% | 26.8% |
| narrow_finance | 6.8% [4.0, 11.2] | **0.0%** | **18.1%** | **0.0%** |
| narrow_sport | 20.0% [14.9, 26.3] | **0.0%** | **0.0%** | **54.3%** |

Two things this establishes:

- **Base is 0.0% at n=192.** PEFT patches modules in place, so base generation had to run through
  `disable_adapter()`; a non-zero rate here would have meant the entire run was contaminated. It
  isn't.
- **The narrow organisms are exactly narrow** — 0.0% outside their training domain, in both cases.
  The general ones leak across domains. Strong behavioural evidence the artifacts are what they're
  labelled, which matters given the unverified provenance of the narrow arm.

(Our general rates, 15–22%, sit below the published ~40%. Expected: 6 samples/question, 256-token
cap, and a question subset.)

## The headline, and why the first version of it was wrong

Naive per-organism AUROC said **narrow more detectable, 0.940 vs 0.877**. That is a topic confound.
All 13 of `narrow_finance`'s misaligned responses are finance-domain; all 38 of `narrow_sport`'s are
sport. A probe that merely detects *what the conversation is about* scores well. The general
organisms' misaligned responses are spread across all three domains, so a topic probe cannot
separate them — a strictly harder task.

Restricting to a single domain removes the confound:

| config | all-domain | **within-domain** |
|---|---|---|
| general_finance | 0.851 | 0.825 |
| general_sport | 0.904 | 0.780 |
| narrow_finance | 0.953 | **0.881** |
| narrow_sport | 0.927 | **0.794** |
| **general mean** | 0.877 | **0.803** |
| **narrow mean** | 0.940 | **0.838** |

The gap falls from 0.063 to 0.035, and the ranges overlap (general 0.780–0.825, narrow 0.794–0.881).
With n=2 per arm, **there is no detectable difference in within-organism probe AUROC between broad
and narrow.** Both sit around 0.80–0.84.

This is the entity-salience confound from `docs/00-findings-and-corrections.md` §1.3 appearing
exactly where that section predicted it would.

## Where there *is* a difference: transfer

AUROC, probe trained on row, evaluated on column (row's peak layer):

| train \ eval | general_finance | general_sport | narrow_finance | narrow_sport |
|---|---|---|---|---|
| general_finance | 0.922 | 0.858 | 0.908 | 0.950 |
| general_sport | 0.796 | 0.953 | **0.573** | 0.935 |
| narrow_finance | 0.654 | **0.507** | 0.970 | 0.924 |
| narrow_sport | 0.687 | 0.801 | 0.718 | 0.950 |

Mean off-diagonal: **general 0.837, narrow 0.715.** The `narrow_finance` probe hits **0.507 —
chance** — on `general_sport`. The `general_finance` probe never drops below 0.858 anywhere.

Probe-direction cosine, general_finance vs narrow_finance at layer 27: **+0.339** — notably lower
than the steering-vector cos-sim (0.55) or the ADL cos (0.48), so the *probe* directions diverge
more than the trained intervention vectors do.

## Reading

The two axes separate, which is the split-y-axis prediction from `docs/01-design.md` §0:

- **Detection** (is it firing here?) — no breadth effect. ~0.80–0.84 either way.
- **Transfer** (does one probe catch the next organism?) — clear breadth effect, 0.84 vs 0.72, with
  narrow bottoming out at chance.

Defensively that reads: you can catch either kind *if you already know which organism you're looking
at*, but only broad ones are catchable with a shared, reusable probe. Narrow ones demand
per-instance probing — the enumerate-the-triggers problem. That supports the project hypothesis on
the transfer axis and finds nothing on the detection axis.

It is also consistent with the geometry: general directions converge (0.97), narrow ones don't
(0.63).

## Caveats

Unchanged from `results/01-adl-arm2.md` and still dominant: **rank-1, single-layer, single-module
adapters are a degenerate organism class**; n=2 per arm; the narrow arm's provenance is inferred and
its hyperparameters differ from the general arm's (lr 5e-4/3ep vs 1e-4/2ep), so the contrast
confounds the KL flag with LR and epochs.

New here: within-domain cells are small (13–38 misaligned), so the within-domain AUROCs carry wide
uncertainty that is not shown.

And the structural one: **these are emergent-misalignment organisms, not secret loyalties.** The
general arm has no activation condition at all; the narrow arm's condition is a topic. Neither has a
principal. Everything above is a proxy result.
