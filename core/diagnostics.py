"""
core/diagnostics.py — Chain quality diagnostics.

Standalone functions for measuring MCMC chain quality. All functions operate
on a list of post-burn-in samples returned by MetropolisHastings.sample().

Functions
---------
acceptance_rate(n_accepted, n_proposed) -> float
    Simple ratio of accepted to proposed moves.

autocorrelation(samples, max_lag) -> list[float]
    Normalized ACF at lags 1..k, truncated at first non-positive lag
    (initial positive sequence estimator, Geyer 1992).

effective_sample_size(samples, max_lag) -> float
    ESS = n / (1 + 2 * Σ ρ_k), where ρ_k are the truncated ACF values.
    For multi-dimensional states, returns the minimum ESS over all dimensions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def acceptance_rate(n_accepted: int, n_proposed: int) -> float:
    """Return the fraction of proposed moves that were accepted.

    Parameters
    ----------
    n_accepted : int
        Number of accepted proposals.
    n_proposed : int
        Total number of proposals made.

    Returns
    -------
    float
        ``n_accepted / n_proposed``, or ``0.0`` if ``n_proposed == 0``.
    """
    if n_proposed == 0:
        return 0.0
    return n_accepted / n_proposed


def autocorrelation(samples: list[Any], max_lag: int = 100) -> list[float]:
    """Compute the truncated normalized autocorrelation function (ACF).

    Uses the initial positive sequence estimator (Geyer 1992): sums only
    contiguous positive autocorrelations starting from lag 1. Stops at
    the first lag whose ACF value is <= 0.

    For multi-dimensional states (numpy array or dict), the ACF is
    computed for the dimension with the lowest effective sample size
    (most conservative estimate).

    Parameters
    ----------
    samples : list
        Post-burn-in chain states. May be floats, numpy arrays, or dicts.
    max_lag : int
        Maximum lag to consider before forcing truncation.

    Returns
    -------
    list[float]
        ACF values at lags 1, 2, ... up to the first non-positive lag or
        max_lag, whichever comes first. Empty list if the chain has zero
        variance or too few samples.
    """
    arr = _to_array(samples)
    if arr.ndim == 1:
        acf, _ = _acf_truncated(arr, max_lag)
        return acf

    # Multi-D: return ACF for the dimension with the lowest ESS
    best_acf: list[float] = []
    best_ess = float("inf")
    for d in range(arr.shape[1]):
        acf_d, rho_sum_d = _acf_truncated(arr[:, d], max_lag)
        n = len(arr)
        ess_d = n / (1.0 + 2.0 * rho_sum_d)
        if ess_d < best_ess:
            best_ess = ess_d
            best_acf = acf_d
    return best_acf


def effective_sample_size(samples: list[Any], max_lag: int = 100) -> float:
    """Compute the effective sample size (ESS) of a chain.

    Formula::

        ESS = n / (1 + 2 * Σ_k ρ_k)

    where ρ_k are the truncated ACF values (initial positive sequence
    estimator, Geyer 1992). For multi-dimensional states, returns the
    minimum ESS over all dimensions (most conservative).

    Parameters
    ----------
    samples : list
        Post-burn-in chain states. May be floats, numpy arrays, or dicts.
    max_lag : int
        Maximum lag to consider before forcing truncation.

    Returns
    -------
    float
        ESS, clamped to the interval ``[1, n]``. Returns ``0.0`` for an
        empty sample list.
    """
    arr = _to_array(samples)
    n = len(arr)
    if n == 0:
        return 0.0

    if arr.ndim == 1:
        _, rho_sum = _acf_truncated(arr, max_lag)
        ess = n / (1.0 + 2.0 * rho_sum)
        return max(1.0, min(float(n), ess))

    # Multi-D: minimum ESS over all dimensions
    best_ess = float("inf")
    for d in range(arr.shape[1]):
        _, rho_sum_d = _acf_truncated(arr[:, d], max_lag)
        ess_d = n / (1.0 + 2.0 * rho_sum_d)
        if ess_d < best_ess:
            best_ess = ess_d

    return max(1.0, min(float(n), best_ess))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(samples: list[Any]) -> np.ndarray:
    """Convert a list of chain states to a float numpy array.

    * ``float`` / ``int`` states → shape ``(n,)``
    * ``numpy.ndarray`` states  → shape ``(n, d)``
    * ``dict`` states           → sorted values → shape ``(n, |keys|)``
    """
    if not samples:
        return np.array([], dtype=float)
    first = samples[0]
    if isinstance(first, dict):
        keys = sorted(first.keys())
        return np.array([[s[k] for k in keys] for s in samples], dtype=float)
    return np.asarray(samples, dtype=float)


def _acf_truncated(x: np.ndarray, max_lag: int) -> tuple[list[float], float]:
    """Compute truncated normalized ACF at lags 1..max_lag.

    Stops at the first lag whose autocorrelation is <= 0 (initial positive
    sequence estimator). Returns both the ACF list and the sum of all
    returned values (used directly in the ESS denominator).

    Parameters
    ----------
    x : np.ndarray
        1-D array of scalar chain values.
    max_lag : int
        Maximum lag to evaluate before forcing truncation.

    Returns
    -------
    acf : list[float]
        Autocorrelations at lags 1, 2, ... up to the cutoff.
    rho_sum : float
        Sum of all returned autocorrelations.
    """
    n = len(x)
    x = x - x.mean()
    var = float(x.var())
    if var < 1e-12:
        return [], 0.0

    cap = min(max_lag, n // 3)
    acf: list[float] = []
    rho_sum = 0.0
    for k in range(1, cap + 1):
        rho = float(np.mean(x[: n - k] * x[k:])) / var
        if rho <= 0.0:
            break
        acf.append(rho)
        rho_sum += rho
    return acf, rho_sum
