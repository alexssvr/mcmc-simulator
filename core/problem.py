"""
core/problem.py — SamplingProblem abstract base class.

Every distribution or sampling problem in this simulator must implement
this interface. The shared Metropolis-Hastings engine (core/mh.py) calls
these methods and never reads internal problem state directly.

Mathematical context (CS 4850, Eq 6.1)
---------------------------------------
The MH transition kernel from state x to state y ≠ x is:

    P_xy = K_xy · κ_x · min(w_x, w_y) / w_x

where:
  - K_xy  : proposal probability / density of moving x → y
  - κ_x   : holding probability at x (κ_x = 1 for lazy chains)
  - w_x   : unnormalized target weight at x  (w = exp(log_target))
  - w_y   : unnormalized target weight at y

The diagonal entry is:

    P_xx = 1 − Σ_{y ≠ x} P_xy   (stay-put on rejection)

For the proposal K to leave the stationary distribution π = w / Z_w
invariant, it must satisfy the reversibility condition (Lemma 6.12):

    κ_x · K_xy = κ_y · K_yx

For symmetric proposals (random-walk, uniform) this holds trivially and
log_kappa always returns 0.0.  For asymmetric proposals the ratio
  log K_yx − log K_xy
must be included in the acceptance step (the engine handles this).

All arithmetic is done in log-space to avoid numerical underflow when
weights span many orders of magnitude.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SamplingProblem(ABC):
    """Abstract base class for all MCMC sampling problems.

    Subclasses encode a target distribution and a proposal kernel.
    The MH engine (core/mh.py) drives sampling; subclasses only need to
    implement the five methods below.

    Implementing a new problem
    --------------------------
    1. Subclass ``SamplingProblem`` in ``problems/``.
    2. Implement all five abstract methods.
    3. Keep ``log_target`` and ``log_kappa`` in log-space — never return
       raw probabilities from these methods.
    4. In ``propose``, return both ``log_K_xy`` (forward) and
       ``log_K_yx`` (reverse) so the engine can correct for asymmetric
       proposals without knowing proposal internals.
    5. Return a rich ``state_description`` so classifiers and
       diagnostics can adapt automatically.

    Mathematical invariant enforced by the engine
    ----------------------------------------------
    Acceptance probability in log-space::

        log α = min(log_w_y, log_w_x) − log_w_x
                + log_kappa(x)
                + log_K_yx − log_K_xy

    which simplifies to ``min(0, log_w_y − log_w_x)`` for symmetric,
    non-lazy proposals (the common case).
    """

    # ------------------------------------------------------------------
    # Core interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def log_target(self, x: Any) -> float:
        """Return log w(x), the unnormalized log target density/mass.

        Parameters
        ----------
        x:
            Current state.  The type matches what ``initial_state``
            returns and what ``propose`` produces.

        Returns
        -------
        float
            log w(x).  May be ``-inf`` for states outside the support.
            Must never be ``+inf`` or ``nan``.

        Notes
        -----
        Work in log-space throughout — never exponentiate and take a
        log, as that loses precision for extreme weights.
        """

    @abstractmethod
    def propose(self, x: Any) -> tuple[Any, float, float]:
        """Sample a candidate state y from the proposal kernel K(x, ·).

        Parameters
        ----------
        x:
            Current state.

        Returns
        -------
        y : Any
            Proposed next state, sampled from K(x, ·).
        log_K_xy : float
            Log probability / log density of proposing y from x,
            i.e. log K(x → y).
        log_K_yx : float
            Log probability / log density of the reverse move,
            i.e. log K(y → x).

        Notes
        -----
        For symmetric proposals K(x, y) = K(y, x), so
        ``log_K_xy == log_K_yx`` always.  Returning both values
        anyway keeps the engine generic — it cancels them out
        automatically for symmetric kernels.
        """

    @abstractmethod
    def log_kappa(self, x: Any) -> float:
        """Return log κ(x), the log holding probability at state x.

        Parameters
        ----------
        x:
            Current state.

        Returns
        -------
        float
            log κ(x).  For a non-lazy chain with symmetric proposal,
            κ_x = 1 for all x, so this always returns ``0.0``.

        Notes
        -----
        Holding probabilities are needed when K is not reversible by
        itself (Eq 6.1).  For the vast majority of problems in this
        simulator, implement this as::

            def log_kappa(self, x: Any) -> float:
                return 0.0
        """

    @abstractmethod
    def initial_state(self) -> Any:
        """Return a valid starting state for the Markov chain.

        The returned value must be in the support of the target
        (i.e. ``log_target(initial_state())`` must be finite) so
        that the chain can make its first move.

        Returns
        -------
        Any
            A starting state compatible with ``propose`` and
            ``log_target``.
        """

    @abstractmethod
    def state_description(self) -> dict[str, Any]:
        """Return metadata describing the state space.

        Used by the classifier, configurator, and diagnostics to
        adapt behaviour automatically without inspecting internals.

        Returns
        -------
        dict with the following keys:

        ``"type"`` : str
            One of ``"discrete"``, ``"continuous"``, or ``"path"``.
        ``"dimension"`` : int
            Number of scalar components in a state (1 for univariate).
        ``"bounds"`` : None or list[float]
            ``None`` if unbounded; otherwise ``[low, high]`` giving
            the coordinate-wise bounds (same bounds assumed for every
            dimension).
        ``"notes"`` : str
            Free-form human-readable description of the problem,
            e.g. ``"2-D Gaussian mixture, 3 components"``.

        Example
        -------
        ::

            {
                "type": "continuous",
                "dimension": 2,
                "bounds": None,
                "notes": "Bivariate Gaussian, mu=[0,0], sigma=I",
            }
        """
