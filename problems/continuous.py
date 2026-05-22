"""
problems/continuous.py — Continuous-state SamplingProblem implementations.

Provides two subclasses:

* :class:`DataDrivenProblem` — target is a Gaussian KDE fitted on a data array.
* :class:`FormulaProblem`    — target is a user-supplied ``log_w(x)`` callable.

Both classes share the same three proposal kernels, all of which are
symmetric (K(x, y) = K(y, x)) so ``log_kappa`` always returns ``0.0`` and
``propose`` always returns ``(y, 0.0, 0.0)``.

Proposal kernel contract (``proposal_config`` dict)
----------------------------------------------------
The ``proposal_config`` dict follows the schema produced by
``core.classifier.ProposalClassifier``:

    {
        "type":      str,          # "gaussian_random_walk" | "reflected_gaussian"
                                   #   | "multivariate_gaussian"
        "step_size": float,        # scalar std-dev; default 1.0
        "bounds":    [float,float] # [low, high]; required for reflected_gaussian
        "cov":       np.ndarray,   # (d, d) covariance for multivariate_gaussian;
                                   #   defaults to step_size² · I
    }

All three kernels are symmetric, so the Hastings correction
``log K_yx − log K_xy`` is zero and the engine needs no extra information
beyond the accept/reject decision.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.stats import gaussian_kde

from core.problem import SamplingProblem

# ---------------------------------------------------------------------------
# Module-level proposal helpers (shared by both problem classes)
# ---------------------------------------------------------------------------

_VALID_PROPOSAL_TYPES: frozenset[str] = frozenset(
    {"gaussian_random_walk", "reflected_gaussian", "multivariate_gaussian"}
)


def _reflect(y: float, low: float, high: float) -> float:
    """Fold *y* back into ``[low, high]`` using repeated reflections.

    Uses modular arithmetic so any number of boundary crossings are
    handled correctly in O(1).

    Parameters
    ----------
    y : float
        Raw (possibly out-of-bounds) proposed value.
    low, high : float
        Inclusive bounds of the valid interval.

    Returns
    -------
    float
        Value in ``[low, high]``.
    """
    span = high - low
    y -= low                  # shift to [0, span]
    y = y % (2.0 * span)      # fold period is 2·span
    if y > span:
        y = 2.0 * span - y    # reflect back from upper edge
    return y + low


def _propose_gaussian_random_walk(
    x: Any,
    dim: int,
    step_size: float,
    rng: np.random.Generator,
) -> tuple[Any, float, float]:
    """Symmetric Gaussian random walk: y = x + ε,  ε ~ N(0, step_size² · I)."""
    noise = rng.normal(0.0, step_size, size=dim)
    if dim == 1:
        return float(x) + float(noise[0]), 0.0, 0.0
    return np.asarray(x) + noise, 0.0, 0.0


def _propose_reflected_gaussian(
    x: Any,
    dim: int,
    step_size: float,
    bounds: list[float],
    rng: np.random.Generator,
) -> tuple[Any, float, float]:
    """Reflected Gaussian random walk: draw y = x + N(0, σ²) then fold at bounds.

    Reflection is a deterministic, measure-preserving map on the proposal
    noise, so K remains symmetric and log_K_xy = log_K_yx.
    """
    low, high = bounds[0], bounds[1]
    noise = rng.normal(0.0, step_size, size=dim)
    if dim == 1:
        y = float(x) + float(noise[0])
        return _reflect(y, low, high), 0.0, 0.0
    y = np.asarray(x) + noise
    y = np.array([_reflect(float(yi), low, high) for yi in y])
    return y, 0.0, 0.0


def _propose_multivariate_gaussian(
    x: Any,
    dim: int,
    cov: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Any, float, float]:
    """Multivariate Gaussian proposal: y = x + N(0, Σ).

    Symmetric because N(y − x; 0, Σ) = N(x − y; 0, Σ) for any SPD Σ.
    """
    noise = rng.multivariate_normal(np.zeros(dim), cov)
    if dim == 1:
        return float(x) + float(noise[0]), 0.0, 0.0
    return np.asarray(x) + noise, 0.0, 0.0


def _dispatch_propose(
    x: Any,
    dim: int,
    proposal_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[Any, float, float]:
    """Route a proposal draw to the correct kernel based on *proposal_config*."""
    ptype: str = proposal_config["type"]
    step_size: float = float(proposal_config.get("step_size", 1.0))

    if ptype == "gaussian_random_walk":
        return _propose_gaussian_random_walk(x, dim, step_size, rng)

    if ptype == "reflected_gaussian":
        bounds = proposal_config.get("bounds")
        if bounds is None:
            raise ValueError(
                "'reflected_gaussian' proposal requires 'bounds': [low, high] "
                "in proposal_config."
            )
        return _propose_reflected_gaussian(x, dim, step_size, bounds, rng)

    if ptype == "multivariate_gaussian":
        cov = proposal_config.get("cov")
        if cov is None:
            cov = (step_size ** 2) * np.eye(dim)
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (dim, dim):
            raise ValueError(
                f"proposal_config['cov'] must have shape ({dim}, {dim}), "
                f"got {cov.shape}."
            )
        return _propose_multivariate_gaussian(x, dim, cov, rng)

    raise ValueError(
        f"Unknown proposal type {ptype!r}. "
        f"Must be one of: {sorted(_VALID_PROPOSAL_TYPES)}"
    )


# ---------------------------------------------------------------------------
# DataDrivenProblem
# ---------------------------------------------------------------------------

class DataDrivenProblem(SamplingProblem):
    """Continuous MCMC problem whose target is a Gaussian KDE of observed data.

    The unnormalized log target is::

        log w(x) = log KDE(x)

    where KDE is a :class:`scipy.stats.gaussian_kde` fitted on *data*.

    Parameters
    ----------
    data : numpy.ndarray
        Observed samples.  Shape ``(n,)`` for univariate data or
        ``(n, d)`` for *d*-dimensional data with *n* observations.
    proposal_config : dict
        Proposal kernel specification (see module docstring for schema).
    seed : int or None
        Optional RNG seed for reproducible proposals.
    notes : str
        Free-form description appended to ``state_description``.

    Raises
    ------
    ValueError
        If *data* has more than 2 dimensions, if *proposal_config* is
        missing required keys, or if *proposal_config* specifies an
        unknown type.
    """

    def __init__(
        self,
        data: np.ndarray,
        proposal_config: dict[str, Any],
        seed: int | None = None,
        notes: str = "",
    ) -> None:
        data = np.asarray(data, dtype=float)
        if data.ndim not in (1, 2):
            raise ValueError(
                f"data must be 1-D (n,) or 2-D (n, d), got shape {data.shape}."
            )

        self._data: np.ndarray = data
        self._proposal_config: dict[str, Any] = proposal_config
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._notes: str = notes

        # Infer dimension and build KDE
        if data.ndim == 1:
            self._dim: int = 1
            # gaussian_kde expects shape (d, n); for 1-D that's (1, n) or (n,)
            self._kde: gaussian_kde = gaussian_kde(data)
        else:
            self._dim = int(data.shape[1])
            # gaussian_kde expects shape (d, n)
            self._kde = gaussian_kde(data.T)

    # ------------------------------------------------------------------
    # SamplingProblem interface
    # ------------------------------------------------------------------

    def log_target(self, x: Any) -> float:
        """Return log KDE density at *x*.

        Parameters
        ----------
        x : float or numpy.ndarray
            Query point.  Scalar for 1-D problems; shape ``(d,)`` for
            *d*-dimensional problems.

        Returns
        -------
        float
            Log density under the fitted KDE.  Never ``+inf`` or ``nan``.
        """
        if self._dim == 1:
            val = self._kde.logpdf(np.atleast_1d(float(x)))
        else:
            val = self._kde.logpdf(np.asarray(x, dtype=float).reshape(-1, 1))
        return float(val[0])

    def propose(self, x: Any) -> tuple[Any, float, float]:
        """Draw a candidate state from the configured symmetric proposal kernel.

        Returns
        -------
        tuple[Any, float, float]
            ``(y, 0.0, 0.0)`` — the ``0.0`` pair reflects that all three
            supported kernels are symmetric, so ``log_K_xy = log_K_yx``.
        """
        return _dispatch_propose(x, self._dim, self._proposal_config, self._rng)

    def log_kappa(self, x: Any) -> float:
        """Return ``0.0`` — all supported proposals use a non-lazy chain."""
        return 0.0

    def initial_state(self) -> Any:
        """Return the empirical mean of the data as a starting state.

        The KDE is strictly positive everywhere, so the mean always has
        finite ``log_target``.
        """
        if self._dim == 1:
            return float(np.mean(self._data))
        return np.mean(self._data, axis=0)

    def state_description(self) -> dict[str, Any]:
        """Return state-space metadata for the classifier and diagnostics.

        Returns
        -------
        dict
            ``type`` is ``'continuous'``.  ``bounds`` are inferred from
            ``[data.min(), data.max()]`` unless the proposal_config
            specifies its own bounds.
        """
        config_bounds = self._proposal_config.get("bounds")
        if config_bounds is not None:
            bounds: list[float] | None = [float(config_bounds[0]), float(config_bounds[1])]
        else:
            bounds = [float(self._data.min()), float(self._data.max())]

        n = len(self._data)
        default_notes = f"KDE on {n} observation{'s' if n != 1 else ''}, dim={self._dim}"
        return {
            "type":      "continuous",
            "dimension": self._dim,
            "bounds":    bounds,
            "notes":     self._notes or default_notes,
        }


# ---------------------------------------------------------------------------
# FormulaProblem
# ---------------------------------------------------------------------------

class FormulaProblem(SamplingProblem):
    """Continuous MCMC problem whose target is a user-supplied log-density.

    The unnormalized log target is the value returned by *log_w*::

        log w(x) = log_w(x)

    Parameters
    ----------
    log_w : callable
        Function ``log_w(x) -> float`` returning the unnormalized log
        target density at *x*.  Must be finite for any *x* that
        ``initial_state`` could return.
    proposal_config : dict
        Proposal kernel specification (see module docstring for schema).
    dimension : int
        Dimension of the state space.  Defaults to ``1``.
    bounds : list[float] or None
        Optional ``[low, high]`` bounds for ``state_description`` and
        for ``reflected_gaussian`` proposals.
    seed : int or None
        Optional RNG seed for reproducible proposals.
    notes : str
        Free-form description for ``state_description``.

    Raises
    ------
    TypeError
        If *log_w* is not callable.
    ValueError
        If *dimension* < 1, or if *proposal_config* specifies an unknown type.
    """

    def __init__(
        self,
        log_w: Callable[[Any], float],
        proposal_config: dict[str, Any],
        dimension: int = 1,
        bounds: list[float] | None = None,
        seed: int | None = None,
        notes: str = "",
    ) -> None:
        if not callable(log_w):
            raise TypeError(
                f"log_w must be callable, got {type(log_w).__name__!r}."
            )
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}.")

        self._log_w: Callable[[Any], float] = log_w
        self._proposal_config: dict[str, Any] = proposal_config
        self._dim: int = int(dimension)
        self._bounds: list[float] | None = bounds
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._notes: str = notes

    # ------------------------------------------------------------------
    # SamplingProblem interface
    # ------------------------------------------------------------------

    def log_target(self, x: Any) -> float:
        """Evaluate and return ``log_w(x)``.

        Returns
        -------
        float
            Value of the user-supplied log-density at *x*.
        """
        return float(self._log_w(x))

    def propose(self, x: Any) -> tuple[Any, float, float]:
        """Draw a candidate state from the configured symmetric proposal kernel.

        Returns
        -------
        tuple[Any, float, float]
            ``(y, 0.0, 0.0)`` — symmetric kernel so ``log_K_xy = log_K_yx``.
        """
        return _dispatch_propose(x, self._dim, self._proposal_config, self._rng)

    def log_kappa(self, x: Any) -> float:
        """Return ``0.0`` — all supported proposals use a non-lazy chain."""
        return 0.0

    def initial_state(self) -> Any:
        """Return the origin as a starting state.

        Returns
        -------
        float
            ``0.0`` for 1-D problems.
        numpy.ndarray
            ``np.zeros(d)`` for *d*-dimensional problems.

        Notes
        -----
        The caller is responsible for ensuring ``log_w(0)`` (or
        ``log_w(np.zeros(d))``) is finite.  If the target has no mass
        at the origin, pass ``initial_state`` as ``x0`` to
        :meth:`~core.mh.MetropolisHastings.sample` instead.
        """
        if self._dim == 1:
            return 0.0
        return np.zeros(self._dim)

    def state_description(self) -> dict[str, Any]:
        """Return state-space metadata for the classifier and diagnostics.

        Returns
        -------
        dict
            ``type`` is ``'continuous'``.  ``bounds`` come from the
            constructor argument (may be ``None`` for unbounded).
        """
        return {
            "type":      "continuous",
            "dimension": self._dim,
            "bounds":    self._bounds,
            "notes":     self._notes,
        }
