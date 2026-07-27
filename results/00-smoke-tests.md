# Smoke tests + first result

Run 2026-07-25 on a rented A40 (46 GB, 96 cores, 503 GB RAM, Ubuntu 24.04, torch 2.8.0+cu128,
transformers 5.14.1). Total cost: minutes of GPU time.

## 1. All code paths verified

`src/tests/test_probes_metrics.py` — 44 checks, CPU, pass locally and on the box.
`src/tests/test_activations_gpu.py` — 22 checks on Qwen2.5-0.5B-Instruct, all pass.

Two that were worth the trouble:

**Padding invariance — max relative error 6.0e-05.** A padded batch gives the same last-token
activations as unpadded singles. `extract` reads the final-token index off the attention mask
assuming right padding; had that been wrong, last-token activations would have come from pad
positions and every probe number downstream would have been garbage that still looked plausible.
Mean-pooled matched at 8.9e-05.

**The contamination claim is now measured, not asserted.** With a steering vector injected at layer
L, layers below L are changed by *exactly* 0.00e+00 and layers at/above by ≥0.055. So
`contaminated_layers()` is drawing the line in the right place, and the pilot's headline-numbers-
from-clean-layers-only rule is well founded. Hook removal also leaks exactly 0.00e+00.

Also confirmed: transformers **5.14.1** works with the hook code despite the organisms being saved
under 4.56.2 (the tuple-vs-tensor return handling covers both); bf16 path clean; peak VRAM 2.02 GiB
at 0.5B.

## 2. Soligo et al.'s headline geometry reproduces

All six `ModelOrganismsForEM` steering vectors load, and every one with a `config.json` passes the
arm cross-check (`kl_regularization` flag vs repo name). Confirmed live: `layer_idx=24`,
`alpha=256.0`, `hidden_size=5120`, `kl_weight=1e6` on all three narrow arms and `0` on all three
general arms. `general_medical` is still the one repo with no `config.json`.

| domain | cos(general, narrow) |
|---|---|
| medical | +0.5472 |
| finance | +0.5789 |
| sport | +0.5752 |

Mean **0.567** against their published **0.55**. Independent reproduction of the one existing
quantity that constrains cross-breadth probe transfer — and confirmation that the loader is right
and these are the right artifacts.

## 3. Unplanned result: broad directions converge, narrow ones don't

Not something Soligo et al. report. Cosine between arms of the *same* type across *different*
training domains:

| arm | med–fin | med–sport | fin–sport | mean |
|---|---|---|---|---|
| **general** | +0.819 | +0.758 | +0.789 | **0.789** |
| **narrow** | +0.579 | +0.577 | +0.573 | **0.576** |

PCA over the three domain vectors within each arm:

| arm | PC1 variance explained | mean \|cos(vector, PC1)\| |
|---|---|---|
| general | **85.9%** | 0.927 |
| narrow | **71.7%** | 0.847 |

And projecting each narrow vector onto the 3-space spanned by the general vectors:

| | energy inside span(general) |
|---|---|
| narrow_medical | 30.3% |
| narrow_finance | 34.2% |
| narrow_sport | 33.7% |

So roughly **two thirds of a narrow direction lies outside anything the general solutions span.**
The narrow solution is not "general, attenuated" — it is substantially orthogonal content.

One more layer: after removing each narrow vector's component along its *own* domain's general
direction, the residuals still share **0.40–0.44** cosine with each other. There appears to be a
second shared component — a "narrowness" direction common to KL-regularized solutions, distinct
from the general misalignment direction.

### What this does and does not say

**Does:** broad behaviours converge on a shared, domain-agnostic representation; narrow ones are
comparatively idiosyncratic. That is the generalizing-representation-vs-lookup-table distinction
showing up directly in the geometry, and it predicts the *transfer* asymmetry the design cares
about — a probe trained on one broad organism should carry to another broad organism far better
than a narrow-trained probe carries to another narrow organism. This is the in-miniature version of
the cross-principal transfer question, and it points the way the whitepaper's Direction 4.5 hopes.

**Does not:** say anything yet about within-organism AUROC, which is the actual detectability
y-axis. A narrow behaviour can be perfectly detectable by a probe fit to that specific organism
while transferring nowhere. Transferability and detectability are different claims, and only the
first is touched here.

**Three caveats, stated plainly.** (1) *n = 3 domains.* PC1 over three vectors is suggestive, not
significant; 85.9% vs 71.7% should not be quoted as if it had error bars. (2) These are **steering
vectors the authors trained to induce behaviour**, not probe directions fit to discriminate
activations. The two are related but not the same object, and the geometry of one does not
automatically transfer to the other. (3) It remains a single model family at a single layer.

The probe measurement — the thing that actually needs the 14B forward passes — is next.

## 4. Environment notes for reruns

- `uv pip install --system --break-system-packages` — Ubuntu 24.04 enforces PEP 668, and building a
  venv would re-pull ~2.8 GB of CUDA wheels since torch is already in the system interpreter.
- `HF_HOME=/workspace/hf`. Root is a 50 GB overlay; `/workspace` is a MooseFS network mount with
  244 TB free, measured at **498 MB/s** sequential write — fine for model storage.
- Qwen2.5-14B bf16 is ~29.5 GB against 44.4 GiB usable, so extraction fits with room for a
  48-layer capture (~1 GB at batch 4 × seq 512). Generation will be bandwidth-bound: the A40's
  696 GB/s is roughly a third of an A100's.
