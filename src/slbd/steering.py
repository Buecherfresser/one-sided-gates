"""Loader for the ModelOrganismsForEM steering vectors.

These repos look like PEFT adapters but are NOT -- `peft.PeftModel.from_pretrained` will fail on
them. Their `adapter_config.json` is a custom schema:

    {"adapter_type": "steering_vector",
     "base_model": "unsloth/Qwen2.5-14B-Instruct",
     "layer_idx": 24, "alpha": 256.0, "hidden_size": 5120,
     "steering_vector_path": "steering_vector.pt"}

and the weights are a single `steering_vector.pt` holding one 5120-dim vector.

The narrow/general distinction is machine-readable in each repo's `config.json`:

    general_* : kl_regularization=false, kl_weight=0,    kl_dataset_file=null
    narrow_*  : kl_regularization=true,  kl_weight=1e6,  kl_dataset_file="bad_good_alt_1k*.jsonl"

Verified 2026-07-24 against the HF API. Note `Qwen2.5-14B_steering_vector_general_medical` is the
one repo missing `config.json` (404); its arm is inferable from the repo name only.

UNVERIFIED: nothing in this module has been executed -- it needs the 14B base model. Before
trusting a run, check `verify_arm()` output and confirm the steered model actually misbehaves on the
in-domain eval, per docs/03-pilot.md step 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

REPO_PREFIX = "ModelOrganismsForEM/Qwen2.5-14B_steering_vector"
DOMAINS = ("medical", "finance", "sport")
ARMS = ("general", "narrow")


def repo_id(arm: str, domain: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    return f"{REPO_PREFIX}_{arm}_{domain}"


@dataclass
class SteeringVector:
    """A single-direction intervention added to the residual stream at one layer."""

    vector: "object"  # torch.Tensor, typed loosely to keep torch out of import time
    layer_idx: int
    alpha: float
    base_model: str
    hidden_size: int
    arm: str
    source_repo: str
    config: dict

    @property
    def is_kl_regularized(self) -> bool:
        return bool(self.config.get("kl_regularization", False))

    def verify_arm(self) -> str:
        """Cross-check the repo-name arm against the config's KL flag.

        Returns a human-readable verdict rather than raising, because `general_medical` legitimately
        has no config.json and would otherwise be unusable.
        """
        if not self.config:
            return f"{self.source_repo}: NO config.json -- arm '{self.arm}' from repo name only"
        kl = self.is_kl_regularized
        weight = self.config.get("kl_weight", 0)
        expected = self.arm == "narrow"
        verdict = "OK" if kl == expected else "MISMATCH"
        return (
            f"{self.source_repo}: arm={self.arm} kl_regularization={kl} "
            f"kl_weight={weight} -> {verdict}"
        )


def load_steering_vector(
    arm: str,
    domain: str = "finance",
    revision: str | None = None,
) -> SteeringVector:
    """Download and load one steering vector from the Hub."""
    import torch
    from huggingface_hub import hf_hub_download

    rid = repo_id(arm, domain)

    adapter_cfg_path = hf_hub_download(rid, "adapter_config.json", revision=revision)
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)

    if adapter_cfg.get("adapter_type") != "steering_vector":
        raise RuntimeError(
            f"{rid}: expected adapter_type='steering_vector', got "
            f"{adapter_cfg.get('adapter_type')!r} -- the repo layout may have changed"
        )

    # config.json carries the KL flag that distinguishes the arms; absent for general_medical.
    train_cfg: dict = {}
    try:
        train_cfg_path = hf_hub_download(rid, "config.json", revision=revision)
        with open(train_cfg_path) as f:
            train_cfg = json.load(f)
    except Exception:  # noqa: BLE001 -- a missing config.json is expected for one repo
        pass

    vec_name = adapter_cfg.get("steering_vector_path", "steering_vector.pt")
    vec_path = hf_hub_download(rid, vec_name, revision=revision)
    raw = torch.load(vec_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        # Tolerate a state-dict wrapper; take the sole tensor.
        tensors = [v for v in raw.values() if hasattr(v, "shape")]
        if len(tensors) != 1:
            raise RuntimeError(f"{rid}: expected one tensor in {vec_name}, found {len(tensors)}")
        raw = tensors[0]
    vector = raw.squeeze().float()

    expected_dim = int(adapter_cfg.get("hidden_size", vector.numel()))
    if vector.numel() != expected_dim:
        raise RuntimeError(
            f"{rid}: vector has {vector.numel()} elements, config says hidden_size={expected_dim}"
        )

    return SteeringVector(
        vector=vector,
        layer_idx=int(adapter_cfg["layer_idx"]),
        alpha=float(adapter_cfg.get("alpha", 1.0)),
        base_model=str(adapter_cfg.get("base_model", "unsloth/Qwen2.5-14B-Instruct")),
        hidden_size=expected_dim,
        arm=arm,
        source_repo=rid,
        config=train_cfg,
    )


class SteeringHook:
    """Adds `alpha * vector` to the residual stream at `layer_idx`, at all token positions.

    Matches the paper's described intervention: "steered with via addition to all token positions at
    the layer from which it was extracted during generation."

    Use as a context manager so the hook is always removed -- a leaked steering hook silently
    contaminates every subsequent forward pass, including the unsteered baseline.
    """

    def __init__(self, model, sv: SteeringVector, scale: float | None = None):
        from .activations import _decoder_layers

        self.model = model
        self.sv = sv
        self.scale = sv.alpha if scale is None else scale
        self._layer = _decoder_layers(model)[sv.layer_idx]
        self._handle = None

    def __enter__(self) -> "SteeringHook":
        vec = self.sv.vector

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            delta = (self.scale * vec).to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + delta
            return (hidden, *output[1:]) if is_tuple else hidden

        self._handle = self._layer.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def contaminated_layers(sv: SteeringVector, n_layers: int) -> tuple[int, ...]:
    """Layers whose activations contain the injected vector by construction.

    The intervention is added at `layer_idx`, so that layer's output and every layer above it
    carry it directly. Probing there measures the intervention rather than the model's own
    representation of the behaviour, which would make the pilot circular. Pass this to
    `fit_layer_sweep(exclude_layers=...)` for the headline numbers, and report layers >= layer_idx
    separately and explicitly.
    """
    return tuple(range(sv.layer_idx, n_layers))
