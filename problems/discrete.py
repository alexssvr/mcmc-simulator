"""
problems/discrete.py — Discrete-state SamplingProblem implementations.

Provides:

* :class:`DiscreteGraphProblem` — Glauber dynamics for proper *q*-coloring
  of an undirected graph.

Graph q-coloring via Glauber dynamics
--------------------------------------
State space Ω = {colorings σ : V → {0, …, q−1}}.

Target distribution (uniform over proper colorings)::

    w(σ) = 1  if σ is a proper coloring (no two adjacent vertices share a color)
    w(σ) = 0  otherwise   →  log w(σ) = −∞

Glauber proposal (symmetric, uniform)::

    K(σ, σ') = 1 / (|V| · q)   for any σ' that differs from σ on exactly one vertex

Because K is uniform over a fixed-size neighbourhood and the neighbourhood is
the same size from both sides (every vertex has the same q choices), the kernel
is symmetric: K(σ, σ') = K(σ', σ).  Consequently:

    log_kappa(σ) = 0.0   (non-lazy chain, κ = 1)
    log_K_xy = log_K_yx  (returns 0.0 for both)

Acceptance probability for a move between proper colorings simplifies to:

    log α = min(0, 0) − 0 + 0 + 0 − 0 = 0   →  always accepted

Moves into improper colorings are always rejected (log w = −∞).

Convergence guarantee
---------------------
Glauber dynamics mix in polynomial time when q ≥ Δ + 2 (Jerrum 1995) and
in O(n log n) when q ≥ 2Δ (Vigoda 1999), where Δ is the maximum degree.
A valid greedy initial coloring always exists when q ≥ Δ + 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.problem import SamplingProblem


class DiscreteGraphProblem(SamplingProblem):
    """Glauber dynamics sampler for uniform *q*-colorings of an undirected graph.

    Parameters
    ----------
    adjacency : dict[int, list[int]]
        Adjacency list representation of an undirected graph.  Keys are
        vertex identifiers (integers); values are lists of neighbour
        vertex identifiers.  Every edge must appear in both directions::

            {0: [1, 2], 1: [0, 2], 2: [0, 1]}   # triangle

    q : int
        Number of available colors (must be >= 2).
    seed : int or None
        Optional RNG seed for reproducible proposals.
    notes : str
        Free-form description appended to ``state_description``.

    Raises
    ------
    ValueError
        If *q* < 2, or if ``initial_state`` cannot find a proper coloring
        (which happens when q < max_degree + 1).
    """

    def __init__(
        self,
        adjacency: dict[int, list[int]],
        q: int,
        seed: int | None = None,
        notes: str = "",
    ) -> None:
        if q < 2:
            raise ValueError(f"q must be >= 2 (number of colors), got {q}.")

        self._adjacency: dict[int, list[int]] = {
            v: list(nbrs) for v, nbrs in adjacency.items()
        }
        self._q: int = int(q)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._notes: str = notes

        # Derived helpers — computed once
        self._vertices: list[int] = sorted(self._adjacency)
        self._n_vertices: int = len(self._vertices)
        self._max_degree: int = max(
            (len(nbrs) for nbrs in self._adjacency.values()), default=0
        )

    # ------------------------------------------------------------------
    # SamplingProblem interface
    # ------------------------------------------------------------------

    def log_target(self, state: dict[int, Any]) -> float:
        """Return ``0.0`` for a proper coloring, ``-inf`` for an improper one.

        A coloring is *proper* if no two adjacent vertices share the same color.

        Parameters
        ----------
        state : dict[int, int]
            Mapping from vertex id to color index in ``{0, …, q−1}``.

        Returns
        -------
        float
            ``0.0`` if proper; ``-math.inf`` if any edge is monochromatic.
        """
        for v, neighbors in self._adjacency.items():
            cv: int = state[v]
            for u in neighbors:
                if state[u] == cv:
                    return -math.inf
        return 0.0

    def propose(self, state: dict[int, Any]) -> tuple[dict[int, Any], float, float]:
        """Glauber proposal: pick a random vertex, assign a random color.

        The proposal is uniform over all ``|V| · q`` (vertex, color) pairs,
        making it symmetric: ``K(σ, σ') = 1 / (|V| · q) = K(σ', σ)``.

        Parameters
        ----------
        state : dict[int, int]
            Current coloring.

        Returns
        -------
        new_state : dict[int, int]
            Proposed coloring — identical to *state* except possibly at
            one vertex.
        log_K_xy : float
            ``0.0`` (symmetric kernel — cancels in acceptance formula).
        log_K_yx : float
            ``0.0`` (symmetric kernel — cancels in acceptance formula).

        Notes
        -----
        A new dict is always returned (the current state is never mutated),
        which ensures the MH engine can safely compare old and new states.
        Proposing the *same* color as the current assignment is allowed;
        the chain stays put with probability ``1/q`` per vertex as a
        natural lazy component.
        """
        vertex: int = int(self._rng.choice(self._vertices))
        new_color: int = int(self._rng.integers(0, self._q))
        new_state: dict[int, Any] = dict(state)
        new_state[vertex] = new_color
        return new_state, 0.0, 0.0

    def log_kappa(self, state: dict[int, Any]) -> float:
        """Return ``0.0`` — the Glauber kernel is symmetric and non-lazy."""
        return 0.0

    def initial_state(self) -> dict[int, int]:
        """Return a proper *q*-coloring found by greedy vertex colouring.

        Vertices are coloured in sorted order.  Each vertex receives the
        smallest color index not already used by any of its already-colored
        neighbors.  This always succeeds when ``q >= max_degree + 1``.

        Returns
        -------
        dict[int, int]
            A proper coloring guaranteed to have finite ``log_target``.

        Raises
        ------
        ValueError
            If the greedy algorithm exhausts all *q* colors at some vertex,
            i.e. ``q < max_degree + 1``.  Increase *q* or pre-compute a
            valid coloring and pass it directly to
            :meth:`~core.mh.MetropolisHastings.sample` as ``x0``.
        """
        coloring: dict[int, int] = {}
        for vertex in self._vertices:
            neighbor_colors: set[int] = {
                coloring[nbr]
                for nbr in self._adjacency[vertex]
                if nbr in coloring
            }
            # Find the smallest valid color
            chosen: int | None = None
            for color in range(self._q):
                if color not in neighbor_colors:
                    chosen = color
                    break

            if chosen is None:
                raise ValueError(
                    f"Greedy coloring failed at vertex {vertex}: all {self._q} "
                    f"colors are used by its neighbors.  "
                    f"Increase q to at least max_degree + 1 = "
                    f"{self._max_degree + 1}."
                )
            coloring[vertex] = chosen

        return coloring

    def state_description(self) -> dict[str, Any]:
        """Return state-space metadata for the classifier and diagnostics.

        Returns
        -------
        dict
            ``type`` is ``'discrete'``.  ``dimension`` is the number of
            vertices (one integer coordinate per vertex).  ``bounds``
            is ``[0, q - 1]`` (the color index range).
        """
        return {
            "type":      "discrete",
            "dimension": self._n_vertices,
            "bounds":    [0, self._q - 1],
            "notes":     self._notes or f"graph q-coloring, |V|={self._n_vertices}, q={self._q}",
        }
