"""Breadth vs. detectability in installed secret loyalties.

Two independent halves:

- `probes`, `metrics` -- pure numpy/sklearn, CPU, covered by `src/tests`. Run these anywhere.
- `activations`, `steering` -- need torch, a GPU and a real checkpoint. NOT covered by tests.

Import the light half without pulling in torch.
"""

from .metrics import (
    DetectionResult,
    auroc,
    bootstrap_auroc_ci,
    evaluate,
    mcnemar,
    recall_at_fpr,
    wilson_interval,
)
from .probes import (
    LayerSweep,
    LogisticProbe,
    MeanDiffProbe,
    cosine_similarity,
    fit_layer_sweep,
    transfer_matrix,
)

__all__ = [
    "DetectionResult",
    "LayerSweep",
    "LogisticProbe",
    "MeanDiffProbe",
    "auroc",
    "bootstrap_auroc_ci",
    "cosine_similarity",
    "evaluate",
    "fit_layer_sweep",
    "mcnemar",
    "recall_at_fpr",
    "transfer_matrix",
    "wilson_interval",
]

# Pre-registered comparison point: Soligo et al. (arXiv:2602.07852) report cos-sim = 0.55 between
# their general and narrow misalignment directions. It is the only published quantity bounding how
# much a broad-trained probe should transfer to a narrow behaviour.
SOLIGO_GENERAL_NARROW_COSSIM = 0.55
