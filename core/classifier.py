"""
core/classifier.py — ProposalClassifier.

Maps a state_description dict (from SamplingProblem.state_description()) to a
proposal_config dict that DataDrivenProblem, FormulaProblem, and
MCMCSampler can consume directly.

Output schema (proposal_config)
--------------------------------
::

    {
        "type":                  str,           # kernel name (see below)
        "step_size":             float | None,  # scalar std-dev; None for discrete
        "bounds":                [float,float], # only present for reflected_gaussian
        "cov":                   np.ndarray,    # only present for multivariate_gaussian
        "recommended_burn_in":   int,
        "recommended_n_samples": int,
    }

Kernel selection rules
-----------------------
+---------------+-----------+-----------------------------+----------------------+
| state type    | dimension | bounds present?             | kernel chosen        |
+===============+===========+=============================+======================+
| discrete      | any       | —                           | discrete_uniform     |
| continuous    | 1         | no                          | gaussian_random_walk |
| continuous    | 1         | yes                         | reflected_gaussian   |
| continuous    | > 1       | no                          | multivariate_gaussian|
| continuous    | > 1       | yes                         | reflected_gaussian   |
| path          | any       | —                           | gaussian_random_walk |
+---------------+-----------+-----------------------------+----------------------+

Step-size heuristics
---------------------
* Bounded state space: ``step_size = (high − low) / 10``
* Unbounded 1-D: ``step_size = 1.0`` by default; when ``data_std`` is
  provided, ``step_size = 2.38 * data_std`` (Roberts, Gelman & Gilks 1997
  optimal for 1-D Gaussian targets, targets ≈ 44 % acceptance)
* Multi-D unbounded: ``step_size = 1.0``, covariance = ``step_size² · I``
* Path: ``step_size = 0.1`` (path increments are typically small)

Burn-in and sample count heuristics
-------------------------------------
These are conservative defaults meant to ensure reasonable mixing across a
wide range of problems.  The user or a diagnostics pass can override them.

* 1-D continuous: ``burn_in = 1 000``, ``n_samples = 5 000``
* d-D continuous: ``burn_in = 1 000·d``, ``n_samples = 5 000·d`` (capped at
  10 000 / 50 000)
* Discrete graph: ``burn_in = 1 000·|V|``, ``n_samples = 5 000·|V|`` (capped
  at 10 000 / 50 000)
* Path: ``burn_in = 2 000``, ``n_samples = 10 000``
"""

from __future__ import annotations

from typing import Any

import numpy as np


_VALID_STATE_TYPES: frozenset[str] = frozenset({"discrete", "continuous", "path"})


class ProposalClassifier:
    """Rule-based classifier: state description → proposal configuration.

    The classifier is stateless — a single instance can be reused for any
    number of problems.

    Examples
    --------
    >>> clf = ProposalClassifier()
    >>> cfg = clf.classify({"type": "continuous", "dimension": 1,
    ...                     "bounds": None, "notes": ""})
    >>> cfg["type"]
    'gaussian_random_walk'
    """

    def classify(
        self,
        state_description: dict[str, Any],
        data_std: float | None = None,
    ) -> dict[str, Any]:
        """Map a state description to a proposal kernel configuration.

        Parameters
        ----------
        state_description : dict
            Must contain the key ``"type"`` (``'discrete'``, ``'continuous'``,
            or ``'path'``).  May optionally contain ``"dimension"`` (default
            1) and ``"bounds"`` (default ``None``).

            **Contract on** ``"bounds"``:
            This field must represent a *physical constraint* on the state
            space (e.g. a probability simplex, a bounded interval enforced by
            the problem definition), **not** a statistical property inferred
            from data (e.g. ``[data.min(), data.max()]``).

            When ``"bounds"`` is ``None`` the classifier chooses an unbounded
            kernel (``gaussian_random_walk`` or ``multivariate_gaussian``).
            When ``"bounds"`` is a ``[low, high]`` list the classifier chooses
            ``reflected_gaussian`` and tunes the step size to the interval
            width.

            Callers such as :class:`~main.MCMCSampler` are responsible for
            passing only *explicitly-provided* bounds here — not bounds
            inferred from ``data.min()`` / ``data.max()``.

        data_std : float or None
            Standard deviation of the observed data, used to tune the
            ``gaussian_random_walk`` step size for data-driven problems.
            When provided, ``step_size = 2.38 * data_std``, which targets
            an acceptance rate near the theoretically optimal 44 % for
            1-D Metropolis–Hastings (Roberts, Gelman & Gilks 1997).
            When ``None``, the default step size of ``1.0`` is used.
            Has no effect when the chosen kernel is not
            ``gaussian_random_walk`` (i.e. bounded or multi-D problems).

        Returns
        -------
        dict
            Proposal configuration (see module docstring for full schema).

        Raises
        ------
        ValueError
            If ``state_description["type"]`` is not a recognised value.
        """
        stype: str = state_description.get("type", "")
        if stype not in _VALID_STATE_TYPES:
            raise ValueError(
                f"Unknown state space type {stype!r}. "
                f"Must be one of: {sorted(_VALID_STATE_TYPES)}"
            )

        dim: int = int(state_description.get("dimension", 1))
        bounds: list[float] | None = state_description.get("bounds", None)

        if stype == "discrete":
            return self._classify_discrete(dim)
        if stype == "continuous":
            return self._classify_continuous(dim, bounds, data_std)
        # stype == "path"
        return self._classify_path(dim)

    # ------------------------------------------------------------------
    # Private per-type classifiers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_discrete(dim: int) -> dict[str, Any]:
        """Discrete state space — Glauber dynamics (proposal built into problem).

        The returned ``"type": "discrete_uniform"`` is a sentinel recognised
        by MCMCSampler to skip proposal injection; the problem class handles
        the kernel internally.
        """
        # Heuristic: larger graphs need more steps to mix
        burn_in: int = min(10_000, max(1_000, 1_000 * dim))
        n_samples: int = min(50_000, max(5_000, 5_000 * dim))
        return {
            "type":                  "discrete_uniform",
            "step_size":             None,
            "recommended_burn_in":   burn_in,
            "recommended_n_samples": n_samples,
        }

    @staticmethod
    def _classify_continuous(
        dim: int,
        bounds: list[float] | None,
        data_std: float | None = None,
    ) -> dict[str, Any]:
        """Continuous state space — choose kernel by dimension and boundedness.

        Step-size tuning for ``gaussian_random_walk``
        ----------------------------------------------
        The default step size of ``1.0`` works well for standardised targets
        but can give very high or very low acceptance rates when the data lives
        on a different scale.  When ``data_std`` is provided the step size is
        set to ``2.38 * data_std``, targeting the theoretically optimal
        acceptance rate of ≈ 44 % for 1-D Metropolis–Hastings.

        The constant 2.38 comes from Roberts, Gelman & Gilks (1997): for a
        Gaussian target N(0, σ²) the proposal std h* = 2.38 · σ maximises
        the expected squared jump distance and yields 44 % acceptance — the
        optimal rate for a 1-D random-walk MH chain.
        """
        burn_in: int   = min(10_000, max(1_000, 1_000 * dim))
        n_samples: int = min(50_000, max(5_000, 5_000 * dim))

        if bounds is not None:
            # Bounded: reflect at boundaries regardless of dimension.
            # data_std does not apply here — step size is tuned to interval width.
            low, high = float(bounds[0]), float(bounds[1])
            step_size: float = (high - low) / 10.0
            return {
                "type":                  "reflected_gaussian",
                "step_size":             step_size,
                "bounds":                [low, high],
                "recommended_burn_in":   burn_in,
                "recommended_n_samples": n_samples,
            }

        if dim == 1:
            # Tune step size to data scale when available; fall back to 1.0.
            step_size = 2.38 * float(data_std) if data_std is not None else 1.0
            return {
                "type":                  "gaussian_random_walk",
                "step_size":             step_size,
                "recommended_burn_in":   burn_in,
                "recommended_n_samples": n_samples,
            }

        # Multi-D unbounded — data_std not used (cov handles scaling).
        step_size = 1.0
        return {
            "type":                  "multivariate_gaussian",
            "step_size":             step_size,
            "cov":                   (step_size ** 2) * np.eye(dim),
            "recommended_burn_in":   burn_in,
            "recommended_n_samples": n_samples,
        }

    @staticmethod
    def _classify_path(dim: int) -> dict[str, Any]:
        """Path state space — small-step random walk."""
        return {
            "type":                  "gaussian_random_walk",
            "step_size":             0.1,
            "recommended_burn_in":   2_000,
            "recommended_n_samples": 10_000,
        }
