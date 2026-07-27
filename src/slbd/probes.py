"""Linear probes over residual-stream activations.

Two probe families, both fit on CPU in seconds:

- `MeanDiffProbe` -- difference-in-means, optionally whitened by the pooled covariance (Marks &
  Tegmark's "mass-mean" probe). Their COLM 2024 result is that mass-mean generalises about as well
  as logistic regression while being substantially more causally implicated: on LLaMA-2-13B/cities
  it reached 0.77 normalised indirect effect vs logistic regression's 0.13. For a study that wants
  to claim it found *the* loyalty direction rather than *a* separating hyperplane, that matters.
- `LogisticProbe` -- L2 logistic regression, the standard baseline (Apollo used lambda=10 on
  normalised activations).

Both expose the same interface so the pilot can report them side by side, and both support layer
ensembling, which Nordby et al. measured at +29% to +78% AUROC over any single layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected (n_samples, d_model), got shape {a.shape}")
    return a


class MeanDiffProbe:
    """Difference-in-means direction, optionally covariance-whitened.

    `whiten=True` gives the mass-mean / LDA-style probe: direction = Sigma^-1 (mu_pos - mu_neg),
    where Sigma is the pooled within-class covariance. `whiten=False` gives the raw
    difference-of-means used by MacDiarmid et al. for defection probes and by Soligo et al. for the
    misalignment direction.
    """

    def __init__(self, whiten: bool = False, shrinkage: float = 1e-3):
        self.whiten = whiten
        self.shrinkage = shrinkage
        self.direction: np.ndarray | None = None
        self.offset: float = 0.0

    def fit(self, x_pos: np.ndarray, x_neg: np.ndarray) -> "MeanDiffProbe":
        p, n = _as2d(x_pos), _as2d(x_neg)
        if p.shape[1] != n.shape[1]:
            raise ValueError(f"d_model mismatch: {p.shape[1]} vs {n.shape[1]}")

        mu_p, mu_n = p.mean(axis=0), n.mean(axis=0)
        d = mu_p - mu_n

        if self.whiten:
            centred = np.vstack([p - mu_p, n - mu_n])
            cov = np.cov(centred, rowvar=False)
            # Ledoit-Wolf-style shrinkage toward a scaled identity; d_model >> n_samples here, so
            # the empirical covariance is singular and this is not optional.
            trace_mean = np.trace(cov) / cov.shape[0]
            cov = (1 - self.shrinkage) * cov + self.shrinkage * trace_mean * np.eye(cov.shape[0])
            d = np.linalg.solve(cov, d)

        norm = np.linalg.norm(d)
        self.direction = d / norm if norm > 0 else d
        # Midpoint of the projected class means: a sane default, but every reported operating point
        # should come from metrics.recall_at_fpr against a fixed control set, not from this.
        self.offset = float((mu_p @ self.direction + mu_n @ self.direction) / 2.0)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.direction is None:
            raise RuntimeError("probe not fitted")
        return _as2d(x) @ self.direction - self.offset


class LogisticProbe:
    """L2-regularised logistic regression via scikit-learn, on standardised features."""

    def __init__(self, c: float = 0.1, max_iter: int = 2000):
        self.c = c
        self.max_iter = max_iter
        self._model = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def fit(self, x_pos: np.ndarray, x_neg: np.ndarray) -> "LogisticProbe":
        from sklearn.linear_model import LogisticRegression

        p, n = _as2d(x_pos), _as2d(x_neg)
        x = np.vstack([p, n])
        y = np.concatenate([np.ones(p.shape[0]), np.zeros(n.shape[0])])

        self._mu = x.mean(axis=0)
        self._sd = x.std(axis=0) + 1e-8
        xs = (x - self._mu) / self._sd

        self._model = LogisticRegression(C=self.c, max_iter=self.max_iter, solver="lbfgs")
        self._model.fit(xs, y)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("probe not fitted")
        xs = (_as2d(x) - self._mu) / self._sd
        return self._model.decision_function(xs)

    @property
    def direction(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("probe not fitted")
        d = self._model.coef_.ravel() / self._sd
        return d / np.linalg.norm(d)


@dataclass
class LayerSweep:
    """Per-layer probes plus an ensemble over a chosen layer band."""

    probes: dict[int, MeanDiffProbe | LogisticProbe]
    ensemble_layers: tuple[int, ...]
    score_stats: dict[int, tuple[float, float]]
    """Per-layer (mean, std) of scores on the *training* data, for ensemble normalisation.

    These must be frozen at fit time. Normalising per scoring call instead would centre each batch
    to zero mean and destroy exactly the class separation the ensemble is meant to measure -- a
    silent failure that yields AUROC ~= 0.5 and reads as "no signal".
    """

    def score_layer(self, layer: int, x: np.ndarray) -> np.ndarray:
        return self.probes[layer].score(x)

    def score_ensemble(self, acts: dict[int, np.ndarray]) -> np.ndarray:
        """Mean of per-layer scores, each standardised by its fit-time statistics.

        Standardising before averaging is load-bearing: raw projection magnitudes differ by an
        order of magnitude across depth, so an unnormalised mean is dominated by whichever layer
        happens to have the largest residual-stream norm.
        """
        cols = []
        for layer in self.ensemble_layers:
            mu, sd = self.score_stats[layer]
            cols.append((self.probes[layer].score(acts[layer]) - mu) / sd)
        return np.mean(np.stack(cols, axis=0), axis=0)


def fit_layer_sweep(
    acts_pos: dict[int, np.ndarray],
    acts_neg: dict[int, np.ndarray],
    probe_factory=lambda: MeanDiffProbe(whiten=False),
    ensemble_layers: tuple[int, ...] | None = None,
    exclude_layers: tuple[int, ...] = (),
) -> LayerSweep:
    """Fit one probe per layer.

    `exclude_layers` exists for the steering-vector pilot: the intervention is injected at a known
    layer, so that layer and everything above it contains the injected vector *by construction*.
    Probing there measures the intervention, not the model's own representation. See docs/03-pilot.md.
    """
    layers = sorted(set(acts_pos) & set(acts_neg) - set(exclude_layers))
    if not layers:
        raise ValueError("no layers left after exclusion")

    probes = {ell: probe_factory().fit(acts_pos[ell], acts_neg[ell]) for ell in layers}

    # Freeze the score distribution on the training data, pooled across both classes, so the
    # ensemble's per-layer normalisation is independent of whatever batch is scored later.
    score_stats: dict[int, tuple[float, float]] = {}
    for ell in layers:
        train_scores = np.concatenate(
            [probes[ell].score(acts_pos[ell]), probes[ell].score(acts_neg[ell])]
        )
        score_stats[ell] = (float(train_scores.mean()), float(train_scores.std()) + 1e-8)

    if ensemble_layers is None:
        # Middle band by default -- where MacDiarmid et al. found defection linearly represented
        # "with very high salience across a wide range of middle layers".
        lo, hi = int(0.35 * max(layers)), int(0.75 * max(layers))
        ensemble_layers = tuple(ell for ell in layers if lo <= ell <= hi) or tuple(layers)

    return LayerSweep(
        probes=probes,
        ensemble_layers=tuple(ensemble_layers),
        score_stats=score_stats,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two probe directions.

    The pre-registered comparison point is Soligo et al.'s cos-sim(general, narrow) = 0.55 -- the
    only published quantity constraining how much a broad-trained probe should transfer to a narrow
    behaviour.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def transfer_matrix(
    sweeps: dict[str, LayerSweep],
    eval_acts: dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]],
    layer: int | None = None,
) -> dict[tuple[str, str], float]:
    """AUROC for every (probe trained on A, evaluated on B) pair.

    The diagonal is within-condition performance; the off-diagonal is transfer. Parrack et al.'s
    0.844-vs-0.550 result was exactly an off-diagonal cell, so this matrix is the deliverable --
    not a single headline number.
    """
    from .metrics import auroc as _auroc

    out: dict[tuple[str, str], float] = {}
    for train_name, sweep in sweeps.items():
        for eval_name, (pos, neg) in eval_acts.items():
            if layer is None:
                s_pos = sweep.score_ensemble(pos)
                s_neg = sweep.score_ensemble(neg)
            else:
                s_pos = sweep.score_layer(layer, pos[layer])
                s_neg = sweep.score_layer(layer, neg[layer])
            out[(train_name, eval_name)] = _auroc(s_pos, s_neg)
    return out
