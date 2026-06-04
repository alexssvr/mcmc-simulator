"""
problems/weighted_discrete.py — Generic weighted discrete SamplingProblem.

This is the primary "plug-in-anything" interface for finite discrete
distributions. The user provides a list of states and a corresponding
list of (unnormalized) weights; the MH engine samples from the
distribution proportional to those weights.

Quickstart
----------
::

    from problems.weighted_discrete import WeightedDiscreteProblem
    from core.mh import MetropolisHastings

    states  = ["rainy", "cloudy", "sunny"]
    weights = [0.3, 0.5, 0.2]          # unnormalized — need not sum to 1

    problem = WeightedDiscreteProblem(states=states, weights=weights, seed=42)
    mh      = MetropolisHastings(problem, seed=42)
    samples = mh.sample(x0=problem.initial_state(), n_steps=10_000, burn_in=1_000)

The chain converges to a distribution proportional to ``weights`` regardless
of their scale.

Mathematical note
-----------------
The proposal kernel is uniform over all ``n`` states:

    K(x, y) = 1/n   for all x, y (including the self-loop x = y)

Because K is symmetric (K(x, y) = K(y, x)), the engine returns
``(y, 0.0, 0.0)`` for every proposal, and ``log_kappa`` is always ``0.0``.

The MH acceptance probability for a move from state x to state y is:

    α = min(w_x, w_y) / w_x = min(1, w_y / w_x)

which is exactly the standard MH ratio for this kernel.  The chain is
reversible with respect to w (Lemma 6.13) and therefore converges to
π = w / Z_w.
"""

from __future__ import annotations

import math
from typing import Any, Hashable

import numpy as np

from core.problem import SamplingProblem


class WeightedDiscreteProblem(SamplingProblem):
    """MH sampler for any finite discrete distribution given by explicit weights.

    Parameters
    ----------
    states : list
        List of distinct, hashable states.  Any Python objects that can
        serve as dictionary keys are accepted (strings, integers, tuples,
        frozensets, etc.).
    weights : list[float]
        Unnormalized non-negative weights parallel to *states*.  At least
        one weight must be strictly positive.  States with weight ``0``
        will never be visited (they are assigned ``log_target = -inf``).
    seed : int or None
        Optional RNG seed for reproducible proposals.
    notes : str
        Free-form description appended to ``state_description``.

    Raises
    ------
    ValueError
        If ``len(states) != len(weights)``, if any weight is negative,
        or if all weights are zero.
    TypeError
        If any state is not hashable.

    Examples
    --------
    Discrete distribution over integers::

        problem = WeightedDiscreteProblem(
            states=[1, 2, 3, 4, 5, 6],
            weights=[1, 1, 1, 1, 1, 1],   # uniform die
        )

    Named outcomes with non-uniform weights::

        problem = WeightedDiscreteProblem(
            states=["heads", "tails"],
            weights=[0.6, 0.4],
        )
    """

    def __init__(
        self,
        states: list[Hashable],
        weights: list[float],
        seed: int | None = None,
        notes: str = "",
    ) -> None:
        states = list(states)
        weights = list(weights)

        if len(states) != len(weights):
            raise ValueError(
                f"len(states)={len(states)} must equal len(weights)={len(weights)}."
            )
        if len(states) == 0:
            raise ValueError("states must be non-empty.")

        for w in weights:
            if w < 0:
                raise ValueError(
                    f"All weights must be non-negative; got {w!r}."
                )
        if all(w == 0 for w in weights):
            raise ValueError("At least one weight must be strictly positive.")

        self._states: list[Hashable] = states
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._notes: str = notes

        # O(1) log-weight lookup; states with weight 0 → -inf
        self._log_weights: dict[Hashable, float] = {}
        for state, w in zip(states, weights):
            self._log_weights[state] = math.log(w) if w > 0 else -math.inf

        # Cache the highest-weight state for initial_state()
        self._best_state: Hashable = max(
            self._log_weights, key=lambda s: self._log_weights[s]
        )

    # ------------------------------------------------------------------
    # SamplingProblem interface
    # ------------------------------------------------------------------

    def log_target(self, x: Hashable) -> float:
        """Return log w(x).

        Parameters
        ----------
        x : hashable
            A state from *states*.

        Returns
        -------
        float
            ``log(weights[i])`` where ``states[i] == x``.
            ``-math.inf`` if ``weights[i] == 0`` or ``x`` not in *states*.
        """
        return self._log_weights.get(x, -math.inf)

    def propose(self, x: Hashable) -> tuple[Hashable, float, float]:
        """Draw a candidate state uniformly from all states.

        The uniform proposal is symmetric: K(x, y) = 1/n = K(y, x) for
        all x, y, so both log-kernel values are equal and cancel in the
        MH acceptance formula.

        Parameters
        ----------
        x : hashable
            Current state (ignored by the uniform proposal).

        Returns
        -------
        y : hashable
            Uniformly sampled state (may equal *x*).
        log_K_xy : float
            ``0.0`` (symmetric kernel — cancels in acceptance formula).
        log_K_yx : float
            ``0.0`` (symmetric kernel — cancels in acceptance formula).
        """
        idx = int(self._rng.integers(0, len(self._states)))
        return self._states[idx], 0.0, 0.0

    def log_kappa(self, x: Hashable) -> float:
        """Return ``0.0`` — uniform proposal is non-lazy."""
        return 0.0

    def initial_state(self) -> Hashable:
        """Return the state with the highest weight.

        This ensures the chain starts in the support of the target
        (``log_target(initial_state())`` is finite).

        Returns
        -------
        hashable
            The state *s* that maximises ``weights[i]``.
        """
        return self._best_state

    def state_description(self) -> dict[str, Any]:
        """Return state-space metadata.

        Returns
        -------
        dict
            ``type`` is ``'discrete'``.  ``dimension`` is the number of
            states in the support.  ``bounds`` is ``None`` (the states
            need not be ordered or bounded).
        """
        n = len(self._states)
        return {
            "type":      "discrete",
            "dimension": n,
            "bounds":    None,
            "notes":     self._notes or f"weighted discrete, {n} states",
        }
