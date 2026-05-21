"""
core/mh.py — Metropolis-Hastings engine.

Implements the MH transition kernel exactly as stated in CS 4850, Eq 6.1:

    P_xy = K_xy · κ_x · min(w_x, w_y) / w_x        (for x ≠ y)
    P_xx = 1 − Σ_{y ≠ x} P_xy                       (stay-put on rejection)

In log-space the acceptance probability for the move x → y is:

    log α = min(log_w_x, log_w_y) − log_w_x + log_κ_x + log_K_yx − log_K_xy

The move is accepted iff log(U) < log α  where U ~ Uniform(0, 1).

IMPORTANT: Do not modify this file to "simplify" the formula.
The engine must match Eq 6.1 exactly, including the κ_x and
asymmetric-proposal (log_K_yx − log_K_xy) terms.
"""

from __future__ import annotations

import math
import logging
from typing import Any

import numpy as np

from core.problem import SamplingProblem

logger = logging.getLogger(__name__)


class MetropolisHastings:
    """Metropolis-Hastings sampler driven by a :class:`~core.problem.SamplingProblem`.

    The engine is intentionally problem-agnostic: it never inspects
    the state type or the proposal internals. All mathematical
    quantities are obtained through the ``SamplingProblem`` interface.

    Parameters
    ----------
    problem:
        A fully-implemented subclass of :class:`~core.problem.SamplingProblem`.
    seed:
        Optional integer seed for :func:`numpy.random.default_rng`.
        Pass ``None`` for a non-deterministic run.

    Attributes
    ----------
    n_proposed : int
        Total number of ``step`` calls made since construction (or last
        :meth:`reset`).
    n_accepted : int
        Number of those steps in which the proposed state was accepted.

    Examples
    --------
    >>> mh = MetropolisHastings(problem, seed=42)
    >>> samples = mh.sample(x0=problem.initial_state(), n_steps=10_000, burn_in=1_000)
    >>> print(mh.acceptance_rate)
    """

    def __init__(self, problem: SamplingProblem, seed: int | None = None) -> None:
        self.problem: SamplingProblem = problem
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.n_proposed: int = 0
        self.n_accepted: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed moves that have been accepted.

        Returns
        -------
        float
            ``n_accepted / n_proposed``, or ``0.0`` if no steps have
            been taken yet.
        """
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    def reset(self) -> None:
        """Reset acceptance counters without touching the RNG state."""
        self.n_proposed = 0
        self.n_accepted = 0

    def step(self, x: Any) -> Any:
        """Advance the chain by one MH step from state *x* (Eq 6.1).

        Algorithm
        ---------
        1. Draw candidate  ``y, log_K_xy, log_K_yx = problem.propose(x)``.
        2. Evaluate log weights  ``log_wx = problem.log_target(x)``,
           ``log_wy = problem.log_target(y)``.
        3. Compute log acceptance probability (Eq 6.1, log-space)::

               log_alpha = min(log_wx, log_wy) - log_wx
                           + log_kappa_x
                           + log_K_yx - log_K_xy

        4. Draw ``log_u = log(Uniform(0, 1))``.
        5. Accept (return ``y``) iff ``log_u < log_alpha``; else return ``x``.

        Parameters
        ----------
        x:
            Current chain state.  Must satisfy
            ``problem.log_target(x) > -inf``.

        Returns
        -------
        Any
            Next state: either the accepted proposal *y* or the
            unchanged current state *x*.

        Notes
        -----
        * All arithmetic stays in log-space — raw weights are never
          materialised to avoid underflow.
        * The ``log_K_yx − log_K_xy`` term corrects for asymmetric
          proposals; it is exactly zero for symmetric kernels.
        * ``log_kappa_x`` encodes the holding-probability correction
          from Eq 6.1; it is ``0.0`` for standard (non-lazy) chains.
        """
        # Step 1 — draw candidate from proposal kernel K(x, ·)
        y, log_K_xy, log_K_yx = self.problem.propose(x)

        # Step 2 — evaluate unnormalized log weights
        log_wx: float = self.problem.log_target(x)
        log_wy: float = self.problem.log_target(y)

        # Step 3 — log acceptance probability (Eq 6.1, log-space)
        #   log α = min(log_wx, log_wy) − log_wx + log_κ_x + log_K_yx − log_K_xy
        log_kappa_x: float = self.problem.log_kappa(x)
        log_alpha: float = (
            min(log_wx, log_wy)   # min(w_x, w_y)  in log-space
            - log_wx              # divided by w_x
            + log_kappa_x         # κ_x factor from Eq 6.1
            + log_K_yx            # reverse proposal density
            - log_K_xy            # forward proposal density
        )

        # Step 4 — accept / reject via log-uniform comparison
        self.n_proposed += 1
        log_u: float = math.log(self._rng.uniform())
        if log_u < log_alpha:
            # Accept: move to y
            self.n_accepted += 1
            logger.debug(
                "step ACCEPT  log_alpha=%.4f  log_u=%.4f  x=%s  y=%s",
                log_alpha, log_u, x, y,
            )
            return y
        else:
            # Reject: stay at x  (implements P_xx diagonal of Eq 6.1)
            logger.debug(
                "step REJECT  log_alpha=%.4f  log_u=%.4f  x=%s",
                log_alpha, log_u, x,
            )
            return x

    def sample(
        self,
        x0: Any,
        n_steps: int,
        burn_in: int = 0,
    ) -> list[Any]:
        """Run the chain and return post-burn-in samples.

        Parameters
        ----------
        x0:
            Starting state.  Must have finite ``problem.log_target(x0)``.
        n_steps:
            Total number of MH steps to execute (including burn-in).
        burn_in:
            Number of initial steps to discard.  Must satisfy
            ``0 <= burn_in < n_steps``.

        Returns
        -------
        list[Any]
            Recorded states after burn-in, length ``n_steps − burn_in``.
            The first element is the state *after* step ``burn_in + 1``.

        Raises
        ------
        ValueError
            If ``burn_in >= n_steps`` or if ``n_steps < 1``.

        Notes
        -----
        Counters (``n_proposed``, ``n_accepted``) accumulate across
        all steps including burn-in.  Call :meth:`reset` beforehand
        if you want post-burn-in acceptance statistics only.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if burn_in < 0 or burn_in >= n_steps:
            raise ValueError(
                f"burn_in must satisfy 0 <= burn_in < n_steps, "
                f"got burn_in={burn_in}, n_steps={n_steps}"
            )

        x: Any = x0
        samples: list[Any] = []

        for i in range(n_steps):
            x = self.step(x)
            if i >= burn_in:
                samples.append(x)

        logger.info(
            "sample done  n_steps=%d  burn_in=%d  acceptance_rate=%.3f",
            n_steps, burn_in, self.acceptance_rate,
        )
        return samples
