"""GPU smoke tests for activation extraction and steering.

Runs on a small model (Qwen2.5-0.5B-Instruct, ~1GB) so the whole suite is seconds and cents. The
point is to verify LOGIC, not scale -- every bug these catch would otherwise appear at 14B, where
each iteration costs a model load.

The load-bearing test is padding invariance. `extract` reads the final-token index off the attention
mask assuming right padding; if that is wrong, last-token activations are silently taken from pad
positions and every probe result downstream is garbage that still looks plausible.

Run on the GPU box:
    cd /workspace/slbd && python3 tests/test_activations_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slbd.activations import (  # noqa: E402
    ActivationBundle,
    _decoder_layers,
    capture_residual_stream,
    extract,
    response_token_mask,
)
from slbd.metrics import auroc  # noqa: E402
from slbd.probes import MeanDiffProbe, fit_layer_sweep  # noqa: E402
from slbd.steering import SteeringHook, SteeringVector, contaminated_layers  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def load(dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="cuda:0")
    model.eval()
    return model, tok


# Deliberately very different lengths so padding is substantial and misindexing is obvious.
TEXTS = [
    "The capital of France is Paris.",
    "Hi.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole filled with the "
    "ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit "
    "down on or to eat: it was a hobbit-hole, and that means comfort.",
    "Quantum entanglement is a physical phenomenon.",
]


def test_hooks_and_shapes(model, tok) -> int:
    print("\ncapture_residual_stream / shapes")
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    check(f"located {len(_decoder_layers(model))} decoder layers (config says {n_layers})",
          len(_decoder_layers(model)) == n_layers)

    enc = tok(TEXTS[:2], return_tensors="pt", padding=True).to(model.device)
    with capture_residual_stream(model) as store, torch.no_grad():
        model(**enc)
    check(f"captured all {len(store)} layers", len(store) == n_layers)
    shapes_ok = all(v.shape == (2, enc["input_ids"].shape[1], d_model) for v in store.values())
    check(f"hidden shapes are (batch, seq, {d_model})", shapes_ok,
          str({k: tuple(v.shape) for k, v in list(store.items())[:2]}))

    # Hooks must be removed on context exit, or every later forward pass silently accumulates.
    before = len(_decoder_layers(model)[0]._forward_hooks)
    with capture_residual_stream(model):
        during = len(_decoder_layers(model)[0]._forward_hooks)
    after = len(_decoder_layers(model)[0]._forward_hooks)
    check(f"hooks removed on exit ({before} -> {during} -> {after})",
          after == before and during == before + 1)
    return n_layers


def test_padding_invariance(model, tok) -> None:
    """A padded batch must equal unpadded singles. This is the test that matters."""
    print("\npadding invariance (THE critical test)")

    batched = extract(model, tok, TEXTS, batch_size=len(TEXTS))
    singles = [extract(model, tok, [t], batch_size=1) for t in TEXTS]

    probe_layers = [0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]

    worst_last, worst_mean = 0.0, 0.0
    for layer in probe_layers:
        for i in range(len(TEXTS)):
            b_last = batched.last[layer][i].astype(np.float64)
            s_last = singles[i].last[layer][0].astype(np.float64)
            b_mean = batched.mean[layer][i].astype(np.float64)
            s_mean = singles[i].mean[layer][0].astype(np.float64)
            scale_l = max(np.abs(s_last).max(), 1e-6)
            scale_m = max(np.abs(s_mean).max(), 1e-6)
            worst_last = max(worst_last, np.abs(b_last - s_last).max() / scale_l)
            worst_mean = max(worst_mean, np.abs(b_mean - s_mean).max() / scale_m)

    # fp32 weights, fp16 storage -> ~1e-3 relative is the storage floor.
    check(f"last-token matches unpadded (max rel err {worst_last:.2e})", worst_last < 5e-2)
    check(f"mean-pooled matches unpadded (max rel err {worst_mean:.2e})", worst_mean < 5e-2)

    # Negative control: if last-token indexing silently used the padded final column instead of
    # the true final token, the two SHORT sequences would collide with pad-position activations.
    # Confirm the short and long sequences give genuinely different vectors.
    layer = probe_layers[1]
    v_short = batched.last[layer][1].astype(np.float64)
    v_long = batched.last[layer][2].astype(np.float64)
    cos = float(v_short @ v_long / (np.linalg.norm(v_short) * np.linalg.norm(v_long) + 1e-9))
    check(f"distinct sequences give distinct last-token vectors (cos={cos:.3f})", cos < 0.999)


def test_pool_mask(model, tok) -> None:
    print("\npool_mask / response_token_mask")
    prompt = "Q: What is the capital of France?\nA:"
    full = prompt + " The capital of France is Paris, a city on the Seine."
    mask = response_token_mask(tok, prompt, full)
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    check(f"mask excludes the {n_prompt} prompt tokens", not mask[:n_prompt].any())
    check(f"mask selects {int(mask.sum())} response tokens", mask.sum() > 0)

    masked = extract(model, tok, [full], batch_size=1, pool_mask=[mask])
    unmasked = extract(model, tok, [full], batch_size=1)
    layer = model.config.num_hidden_layers // 2
    a = masked.mean[layer][0].astype(np.float64)
    b = unmasked.mean[layer][0].astype(np.float64)
    rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-6)
    check(f"masked mean differs from full mean (rel diff {rel:.3f})", rel > 1e-3)

    # An all-false mask must not divide by zero.
    empty = extract(model, tok, [full], batch_size=1, pool_mask=[np.zeros(4, dtype=bool)])
    check("all-false mask does not produce nan/inf",
          np.isfinite(empty.mean[layer][0].astype(np.float64)).all())


def test_bundle_roundtrip(model, tok, tmp="/tmp/slbd_bundle.npz") -> None:
    print("\nActivationBundle save/load")
    bundle = extract(model, tok, TEXTS[:2], batch_size=2)
    bundle.save(tmp)
    loaded = ActivationBundle.load(tmp)
    check("layer sets match", set(loaded.last) == set(bundle.last))
    check("n_layers/d_model preserved",
          loaded.n_layers == bundle.n_layers and loaded.d_model == bundle.d_model)
    same = all(np.array_equal(loaded.last[k], bundle.last[k]) for k in bundle.last)
    check("last-token arrays bit-identical", same)
    check("as_float64 casts", bundle.as_float64("last")[0].dtype == np.float64)
    check("stored as fp16", bundle.last[0].dtype == np.float16)


def test_steering_hook(model, tok, n_layers: int) -> None:
    """Verify the contamination claim in docs/03-pilot.md, empirically."""
    print("\nSteeringHook + contamination range")
    d_model = model.config.hidden_size
    inject_at = n_layers // 2

    torch.manual_seed(0)
    vec = torch.randn(d_model)
    vec = vec / vec.norm()
    sv = SteeringVector(
        vector=vec, layer_idx=inject_at, alpha=8.0, base_model=MODEL,
        hidden_size=d_model, arm="synthetic", source_repo="(synthetic)", config={},
    )

    base = extract(model, tok, TEXTS, batch_size=4)
    with SteeringHook(model, sv):
        steered = extract(model, tok, TEXTS, batch_size=4)
    restored = extract(model, tok, TEXTS, batch_size=4)

    def reldiff(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))

    below = max(reldiff(steered.last[ell], base.last[ell]) for ell in range(inject_at))
    at_and_above = min(reldiff(steered.last[ell], base.last[ell]) for ell in range(inject_at, n_layers))

    check(f"layers < {inject_at} UNCHANGED by steering (max rel diff {below:.2e})", below < 1e-3)
    check(f"layers >= {inject_at} CHANGED by steering (min rel diff {at_and_above:.3f})",
          at_and_above > 1e-2)

    leak = max(reldiff(restored.last[ell], base.last[ell]) for ell in range(n_layers))
    check(f"hook fully removed on exit (max rel diff {leak:.2e})", leak < 1e-6)

    contaminated = contaminated_layers(sv, n_layers)
    check(f"contaminated_layers() = {inject_at}..{n_layers-1} ({len(contaminated)} layers)",
          contaminated == tuple(range(inject_at, n_layers)))

    # The pilot's guard: excluding the contaminated range must leave a usable sweep.
    pos = {k: v.astype(np.float64) for k, v in steered.last.items()}
    neg = {k: v.astype(np.float64) for k, v in base.last.items()}
    sweep = fit_layer_sweep(pos, neg, exclude_layers=contaminated)
    check(f"layer sweep fits on clean layers only (n={len(sweep.probes)})",
          len(sweep.probes) == inject_at and max(sweep.probes) < inject_at)


def test_end_to_end_probe(model, tok) -> None:
    """Extract -> fit -> score on a real, if trivial, linguistic contrast."""
    print("\nend-to-end extract -> probe")
    pos_texts = [f"The answer is definitely yes, {w}." for w in
                 ["certainly", "absolutely", "indeed", "truly", "surely", "clearly",
                  "plainly", "obviously", "evidently", "undoubtedly"]]
    neg_texts = [f"The answer is definitely no, {w}." for w in
                 ["never", "nope", "negative", "wrong", "false", "incorrect",
                  "untrue", "mistaken", "invalid", "denied"]]

    ap = extract(model, tok, pos_texts, batch_size=5).as_float64("last")
    an = extract(model, tok, neg_texts, batch_size=5).as_float64("last")

    layer = model.config.num_hidden_layers // 2
    probe = MeanDiffProbe().fit(ap[layer], an[layer])
    a = auroc(probe.score(ap[layer]), probe.score(an[layer]))
    check(f"probe separates yes/no contrast at layer {layer} (AUROC={a:.3f})", a > 0.9)

    sweep = fit_layer_sweep(ap, an)
    ens = auroc(sweep.score_ensemble(ap), sweep.score_ensemble(an))
    check(f"ensemble scores without collapsing to 0.5 (AUROC={ens:.3f})", ens > 0.9)


def test_bf16(tok) -> None:
    print("\nbf16 path")
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    m.eval()
    b = extract(m, tok, TEXTS, batch_size=4)
    layer = m.config.num_hidden_layers // 2
    check("bf16 extraction produces finite values",
          np.isfinite(b.last[layer].astype(np.float64)).all())
    check("bf16 stored as fp16", b.last[layer].dtype == np.float16)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"        peak VRAM this session: {peak:.2f} GiB")
    del m
    torch.cuda.empty_cache()


def main() -> int:
    print("=" * 72)
    print(f"GPU smoke tests -- {MODEL}")
    print("=" * 72)
    model, tok = load(torch.float32)
    print(f"loaded: {model.config.num_hidden_layers} layers, hidden {model.config.hidden_size}, "
          f"dtype {next(model.parameters()).dtype}")

    n_layers = test_hooks_and_shapes(model, tok)
    test_padding_invariance(model, tok)
    test_pool_mask(model, tok)
    test_bundle_roundtrip(model, tok)
    test_steering_hook(model, tok, n_layers)
    test_end_to_end_probe(model, tok)
    del model
    torch.cuda.empty_cache()
    test_bf16(tok)

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
