"""Residual-stream activation extraction.

Hooks the post-block residual stream of every decoder layer, which is the site Soligo et al. use
(`model.model.layers[i]`, taking `output[0]`) and the site MacDiarmid et al. probe for defection.

Stores BOTH last-token and mean-pooled representations. That costs one extra line and ~4 GB for 20k
prompts on a 7B, and it saves a full re-extraction pass when the first pooling choice turns out not
to separate the classes. Last-token captures "what the model is about to do"; mean-pooling is more
robust for topic and style properties.

UNVERIFIED: this module needs a GPU and a real checkpoint to exercise. The probe and metric code in
probes.py / metrics.py is covered by src/tests, but nothing here is.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ActivationBundle:
    """Per-layer activations for a set of sequences.

    `last` and `mean` are both {layer_index: (n_sequences, d_model)} in float16.
    """

    last: dict[int, np.ndarray]
    mean: dict[int, np.ndarray]
    n_layers: int
    d_model: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"last/{k}": v for k, v in self.last.items()}
        flat.update({f"mean/{k}": v for k, v in self.mean.items()})
        np.savez_compressed(path, n_layers=self.n_layers, d_model=self.d_model, **flat)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationBundle":
        z = np.load(path)
        last, mean = {}, {}
        for key in z.files:
            if key.startswith("last/"):
                last[int(key.split("/", 1)[1])] = z[key]
            elif key.startswith("mean/"):
                mean[int(key.split("/", 1)[1])] = z[key]
        return cls(last=last, mean=mean, n_layers=int(z["n_layers"]), d_model=int(z["d_model"]))

    def as_float64(self, pooling: str = "last") -> dict[int, np.ndarray]:
        """Cast to float64 for probe fitting. Storage stays fp16; the fit does not."""
        src = {"last": self.last, "mean": self.mean}[pooling]
        return {k: v.astype(np.float64) for k, v in src.items()}


def _decoder_layers(model):
    """Locate the decoder layer list across the common HF causal-LM layouts.

    Unwraps PEFT wrappers first. PEFT patches LoRA modules *in place* inside the base model, so
    hooks registered on the unwrapped layers still observe the adapter-modified computation --
    unwrapping loses nothing and avoids depending on the wrapper's attribute layout.
    """
    candidates = [model]
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            candidates.append(getter())
        except Exception:  # noqa: BLE001 -- prompt-learning PEFT variants raise here
            pass
    # PeftModel -> LoraModel -> underlying causal LM.
    inner = getattr(getattr(model, "base_model", None), "model", None)
    if inner is not None:
        candidates.append(inner)

    paths = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers", "layers")
    for candidate in candidates:
        for attr in paths:
            obj = candidate
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            # Guard against grabbing something that merely has the right name.
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
    raise RuntimeError(f"could not locate decoder layers on {type(model).__name__}")


@contextmanager
def capture_residual_stream(model, layers: list[int] | None = None):
    """Context manager yielding a dict that fills with {layer: hidden_states} on each forward pass.

    Hidden states are the post-block residual stream, kept on-device until the caller pools them.
    """
    import torch

    decoder = _decoder_layers(model)
    targets = list(range(len(decoder))) if layers is None else list(layers)
    store: dict[int, "torch.Tensor"] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # HF decoder layers return a tuple whose first element is the hidden state.
            hidden = output[0] if isinstance(output, tuple) else output
            store[idx] = hidden.detach()

        return hook

    try:
        for idx in targets:
            handles.append(decoder[idx].register_forward_hook(make_hook(idx)))
        yield store
    finally:
        for h in handles:
            h.remove()


def extract(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 1024,
    layers: list[int] | None = None,
    pool_mask: list[np.ndarray] | None = None,
) -> ActivationBundle:
    """Extract last-token and mean-pooled residual-stream activations for `texts`.

    Uses RIGHT padding, deliberately: this is a forward-only pass, and right padding lets the true
    final-token index be read straight off the attention mask. Left padding is only needed for
    batched generation, and mixing the two is a classic source of silently-wrong last-token
    activations.

    `pool_mask`, if given, is one boolean array per text selecting which token positions the mean
    pools over -- use it to restrict pooling to response tokens, which is how Soligo et al. build
    their misalignment direction ("difference in mean residual stream activations ... over aligned
    and misaligned response tokens").
    """
    import torch

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    acc_last: dict[int, list[np.ndarray]] = {}
    acc_mean: dict[int, list[np.ndarray]] = {}
    d_model = 0
    n_layers = 0

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            with capture_residual_stream(model, layers) as store, torch.no_grad():
                model(**enc)

                attn = enc["attention_mask"]
                lengths = attn.sum(dim=1)  # (batch,)
                last_idx = (lengths - 1).clamp(min=0)

                if pool_mask is not None:
                    seq_len = attn.shape[1]
                    rows = []
                    for m in pool_mask[start : start + batch_size]:
                        row = np.zeros(seq_len, dtype=bool)
                        row[: min(len(m), seq_len)] = m[: min(len(m), seq_len)]
                        rows.append(row)
                    weight = torch.tensor(
                        np.stack(rows), device=attn.device, dtype=attn.dtype
                    ) * attn
                else:
                    weight = attn
                # Guard against an all-zero mask producing a divide-by-zero.
                denom = weight.sum(dim=1, keepdim=True).clamp(min=1)

                n_layers = max(n_layers, len(store))
                for layer, hidden in store.items():
                    d_model = hidden.shape[-1]
                    take = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
                    pooled = (hidden * weight.unsqueeze(-1)).sum(dim=1) / denom

                    acc_last.setdefault(layer, []).append(
                        take.to(torch.float16).cpu().numpy()
                    )
                    acc_mean.setdefault(layer, []).append(
                        pooled.to(torch.float16).cpu().numpy()
                    )
    finally:
        tokenizer.padding_side = original_side

    return ActivationBundle(
        last={k: np.concatenate(v, axis=0) for k, v in acc_last.items()},
        mean={k: np.concatenate(v, axis=0) for k, v in acc_mean.items()},
        n_layers=n_layers,
        d_model=d_model,
    )


def response_token_mask(
    tokenizer, prompt: str, full_text: str, max_length: int = 1024
) -> np.ndarray:
    """Boolean mask over `full_text` tokens selecting only the response portion.

    Computed by tokenising the prompt alone and marking everything after it. Tokenisation is not
    always a prefix-preserving operation across a boundary, so this is approximate at exactly one
    token; that is acceptable for mean pooling and is why we do not use it for last-token indexing.
    """
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    n_full = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    mask = np.zeros(min(n_full, max_length), dtype=bool)
    if n_prompt < len(mask):
        mask[n_prompt:] = True
    return mask
