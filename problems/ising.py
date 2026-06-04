"""
problems/ising.py — 2D Ising model SamplingProblem.

A classic statistical physics model that demonstrates how MH MCMC works on
a structured discrete state space. Every spin configuration on an N×N
grid is a state; the target distribution is the Boltzmann distribution.

Physical model
--------------
State space: all N×N grids with entries in {+1, −1} (spin up / spin down).

Target (Boltzmann) distribution::

    w(σ) = exp(β * Σ_{<i,j>} σ[i] * σ[j])

where the sum is over all nearest-neighbor pairs (right and down) with
periodic boundary conditions.  The parameter β = 1 / (k_B T) is the
inverse temperature.

When β > 0 (ferromagnetic):
  - Low β (high T): spins are nearly independent → disordered phase
  - High β (low T): spins tend to align → ordered (magnetized) phase
  - Critical β ≈ 0.44 for the 2D Ising model on an infinite lattice
    (Onsager 1944)

Proposal
--------
Flip one randomly chosen spin.  The proposal is symmetric
(K(σ, σ') = K(σ', σ) = 1/(N²)) so ``log_kappa = 0.0`` and
``propose`` returns ``(σ', 0.0, 0.0)``.

Acceptance probability (from Eq 6.1)::

    log α = min(log w(σ'), log w(σ)) − log w(σ)
           = min(0, log w(σ') − log w(σ))
           = min(0, β * ΔE)

where ΔE = (energy after flip) − (energy before flip).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.problem import SamplingProblem


class IsingModel(SamplingProblem):
    """2D Ising model sampled via single-spin-flip Metropolis-Hastings.

    Parameters
    ----------
    n : int
        Grid side length.  The state space has ``n * n`` binary spins,
        so the total number of states is ``2^(n*n)``.  Values between
        4 and 16 are practical; larger grids mix slowly near criticality.
    beta : float
        Inverse temperature β = 1 / (k_B T).  Positive for ferromagnetic
        coupling, negative for antiferromagnetic.  Zero gives an
        uncorrelated Bernoulli spin field.
    seed : int or None
        Optional RNG seed for reproducible proposals and initial states.

    Examples
    --------
    Sample near the critical temperature (β ≈ 0.44)::

        from problems.ising import IsingModel
        from core.mh import MetropolisHastings

        model = IsingModel(n=8, beta=0.44, seed=0)
        mh    = MetropolisHastings(model, seed=0)
        grids = mh.sample(model.initial_state(), n_steps=5_000, burn_in=1_000)
        # grids is a list of (8, 8) numpy int8 arrays with values ±1
    """

    def __init__(self, n: int, beta: float, seed: int | None = None) -> None:
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}.")
        self._n: int = n
        self._beta: float = float(beta)
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # SamplingProblem interface
    # ------------------------------------------------------------------

    def log_target(self, grid: np.ndarray) -> float:
        """Return the log Boltzmann weight β * Σ_{<i,j>} σ_i * σ_j.

        The sum is over all nearest-neighbor pairs (right + down) with
        periodic boundary conditions.  Each pair is counted once.

        Parameters
        ----------
        grid : numpy.ndarray
            Shape ``(n, n)``, dtype int8, values in {+1, −1}.

        Returns
        -------
        float
            log w(σ) = β * (horizontal energy + vertical energy).
        """
        # Horizontal neighbors: each cell × its right neighbor (periodic)
        h = float(np.sum(grid * np.roll(grid, -1, axis=1)))
        # Vertical neighbors: each cell × its lower neighbor (periodic)
        v = float(np.sum(grid * np.roll(grid, -1, axis=0)))
        return self._beta * (h + v)

    def propose(self, grid: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Flip one randomly chosen spin.

        Parameters
        ----------
        grid : numpy.ndarray
            Current spin configuration, shape ``(n, n)``.

        Returns
        -------
        new_grid : numpy.ndarray
            Copy of *grid* with one spin flipped.
        log_K_xy : float
            ``0.0`` — uniform proposal over N² spins is symmetric.
        log_K_yx : float
            ``0.0`` — symmetric kernel, cancels in acceptance formula.
        """
        i = int(self._rng.integers(0, self._n))
        j = int(self._rng.integers(0, self._n))
        new_grid = grid.copy()
        new_grid[i, j] = -new_grid[i, j]
        return new_grid, 0.0, 0.0

    def log_kappa(self, grid: np.ndarray) -> float:
        """Return ``0.0`` — single-spin-flip proposal is non-lazy."""
        return 0.0

    def initial_state(self) -> np.ndarray:
        """Return a random ±1 spin grid.

        The Boltzmann weight is positive (non-zero) for every grid, so
        any random initialization has finite ``log_target``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, n)``, dtype int8, values uniformly in {+1, −1}.
        """
        raw = self._rng.integers(0, 2, size=(self._n, self._n))
        # Map {0, 1} → {-1, +1}
        return (2 * raw - 1).astype(np.int8)

    def state_description(self) -> dict[str, Any]:
        """Return state-space metadata.

        Returns
        -------
        dict
            ``type`` is ``'discrete'``.  ``dimension`` is ``n * n``
            (one integer spin per grid cell).  ``bounds`` is ``[-1, 1]``.
        """
        return {
            "type":      "discrete",
            "dimension": self._n * self._n,
            "bounds":    [-1, 1],
            "notes": (
                f"2D Ising model, {self._n}×{self._n} grid, "
                f"β={self._beta:.4g} "
                f"({'ordered' if self._beta > 0.44 else 'disordered'} phase)"
            ),
        }
