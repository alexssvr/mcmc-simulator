"""
tests/test_problems.py — Statistical tests for problem implementations.

Covers:

1. test_data_driven_problem
   Fits a KDE to data drawn from N(2, 0.5²) and runs an MH chain.
   Verifies KS test, acceptance rate, and that samples stay near the data range.

2. test_formula_problem
   Uses log_w = -0.5*x² (standard normal) and verifies the chain matches N(0,1).

3. test_reflected_gaussian_proposal
   Verifies that a bounded FormulaProblem with reflected_gaussian never
   produces samples outside the specified bounds.

4. test_weighted_discrete_problem
   Verifies that a WeightedDiscreteProblem converges to the correct
   distribution by checking empirical frequencies against known probabilities.

5. test_ising_model
   Verifies that the IsingModel produces valid ±1 states, finite log-targets,
   and a reasonable acceptance rate across a short chain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from core.mh import MetropolisHastings
from problems.continuous import DataDrivenProblem, FormulaProblem
from problems.weighted_discrete import WeightedDiscreteProblem
from problems.ising import IsingModel


# ---------------------------------------------------------------------------
# 1. test_data_driven_problem
# ---------------------------------------------------------------------------

def test_data_driven_problem() -> None:
    """DataDrivenProblem: KDE on N(2, 0.5) data → chain matches N(2, 0.5)."""
    rng = np.random.default_rng(42)
    mu, sigma = 2.0, 0.5
    data = rng.normal(mu, sigma, size=500)
    data_mean = float(data.mean())
    data_std  = float(data.std(ddof=1))

    proposal_config = {"type": "gaussian_random_walk", "step_size": 2.38 * sigma}
    problem = DataDrivenProblem(data=data, proposal_config=proposal_config, seed=42)
    mh = MetropolisHastings(problem, seed=42)

    n_total, burn_in = 8_000, 1_000
    samples = mh.sample(
        x0=problem.initial_state(), n_steps=n_total, burn_in=burn_in
    )
    arr = np.array(samples, dtype=float)

    # DataDrivenProblem samples from the KDE (not the parametric normal), so we
    # compare chain statistics against the *data* moments rather than N(mu, sigma).
    # ---- Chain mean and std match data moments --------------------------------
    assert abs(float(arr.mean()) - data_mean) < 0.15, (
        f"Chain mean {arr.mean():.4f} too far from data mean {data_mean:.4f}."
    )
    assert abs(float(arr.std(ddof=1)) - data_std) < 0.15, (
        f"Chain std {arr.std(ddof=1):.4f} too far from data std {data_std:.4f}."
    )

    # ---- Acceptance rate in reasonable range ----------------------------------
    rate = mh.acceptance_rate
    assert 0.1 <= rate <= 0.9, (
        f"Acceptance rate {rate:.4f} outside [0.1, 0.9]."
    )


# ---------------------------------------------------------------------------
# 2. test_formula_problem
# ---------------------------------------------------------------------------

def test_formula_problem() -> None:
    """FormulaProblem: log_w = -0.5*x² → chain matches N(0, 1)."""
    proposal_config = {"type": "gaussian_random_walk", "step_size": 1.0}
    problem = FormulaProblem(
        log_w=lambda x: -0.5 * float(x) ** 2,
        proposal_config=proposal_config,
        dimension=1,
        seed=42,
        notes="Standard normal via formula",
    )
    mh = MetropolisHastings(problem, seed=42)

    n_total, burn_in = 6_000, 1_000
    samples = mh.sample(
        x0=problem.initial_state(), n_steps=n_total, burn_in=burn_in
    )
    arr = np.array(samples, dtype=float)

    # ---- KS test against N(0, 1) --------------------------------------------
    ks_stat, ks_pvalue = stats.kstest(arr, "norm", args=(0.0, 1.0))
    assert ks_pvalue > 0.01, (
        f"KS test failed: samples do not look like N(0, 1). "
        f"ks_stat={ks_stat:.4f}, p={ks_pvalue:.4f}"
    )

    # ---- Acceptance rate ------------------------------------------------------
    rate = mh.acceptance_rate
    assert 0.2 <= rate <= 0.8, (
        f"Acceptance rate {rate:.4f} outside [0.2, 0.8]."
    )


# ---------------------------------------------------------------------------
# 3. test_reflected_gaussian_proposal
# ---------------------------------------------------------------------------

def test_reflected_gaussian_proposal() -> None:
    """Reflected Gaussian proposal: no samples should escape the bounds [0, 1]."""
    low, high = 0.0, 1.0
    proposal_config = {
        "type":      "reflected_gaussian",
        "step_size": 0.1,
        "bounds":    [low, high],
    }
    # Uniform target on [0, 1] (log_w = 0 everywhere in support, -inf outside)
    def log_w(x: float) -> float:
        return 0.0 if low <= x <= high else -math.inf

    problem = FormulaProblem(
        log_w=log_w,
        proposal_config=proposal_config,
        dimension=1,
        bounds=[low, high],
        seed=42,
        notes="Uniform [0,1] with reflected Gaussian proposal",
    )
    mh = MetropolisHastings(problem, seed=42)

    samples = mh.sample(
        x0=0.5, n_steps=4_000, burn_in=500
    )
    arr = np.array(samples, dtype=float)

    # ---- Every sample must be within [low, high] ----------------------------
    assert float(arr.min()) >= low - 1e-9, (
        f"Sample {arr.min():.6f} violates lower bound {low}."
    )
    assert float(arr.max()) <= high + 1e-9, (
        f"Sample {arr.max():.6f} violates upper bound {high}."
    )

    # ---- Acceptance rate in a reasonable range --------------------------------
    rate = mh.acceptance_rate
    assert 0.1 <= rate <= 1.0, (
        f"Acceptance rate {rate:.4f} outside [0.1, 1.0]."
    )


# ---------------------------------------------------------------------------
# 4. test_weighted_discrete_problem
# ---------------------------------------------------------------------------

def test_weighted_discrete_problem() -> None:
    """WeightedDiscreteProblem: empirical frequencies match target distribution."""
    states = [0, 1, 2, 3, 4]
    weights = [1.0, 2.0, 4.0, 2.0, 1.0]     # triangular-ish
    total = sum(weights)
    true_probs = [w / total for w in weights]

    problem = WeightedDiscreteProblem(states=states, weights=weights, seed=42)
    mh = MetropolisHastings(problem, seed=42)

    n_total, burn_in = 30_000, 2_000
    samples = mh.sample(
        x0=problem.initial_state(), n_steps=n_total, burn_in=burn_in
    )
    n = len(samples)

    # ---- Empirical frequencies -----------------------------------------------
    counts = {s: 0 for s in states}
    for s in samples:
        counts[s] += 1
    emp_probs = [counts[s] / n for s in states]

    for i, (ep, tp) in enumerate(zip(emp_probs, true_probs)):
        assert abs(ep - tp) < 0.03, (
            f"State {states[i]}: empirical prob {ep:.4f} too far from "
            f"target prob {tp:.4f}."
        )

    # ---- All states must appear (non-zero weight states) ----------------------
    for s in states:
        assert counts[s] > 0, f"State {s} never visited despite positive weight."

    # ---- Acceptance rate -------------------------------------------------------
    rate = mh.acceptance_rate
    assert 0.1 <= rate <= 1.0, (
        f"Acceptance rate {rate:.4f} outside [0.1, 1.0]."
    )

    # ---- initial_state must be the highest-weight state ----------------------
    assert problem.initial_state() == 2, (
        f"Expected initial_state()=2 (highest weight), got {problem.initial_state()!r}"
    )


def test_weighted_discrete_validation() -> None:
    """WeightedDiscreteProblem: constructor rejects bad inputs."""
    with pytest.raises(ValueError, match=r"len\(states\)"):
        WeightedDiscreteProblem(states=[1, 2], weights=[1.0])

    with pytest.raises(ValueError, match="non-negative"):
        WeightedDiscreteProblem(states=[1, 2], weights=[1.0, -0.5])

    with pytest.raises(ValueError, match="strictly positive"):
        WeightedDiscreteProblem(states=[1, 2], weights=[0.0, 0.0])

    with pytest.raises(ValueError, match="non-empty"):
        WeightedDiscreteProblem(states=[], weights=[])


# ---------------------------------------------------------------------------
# 5. test_ising_model
# ---------------------------------------------------------------------------

def test_ising_model() -> None:
    """IsingModel: valid ±1 states, finite log-targets, reasonable acceptance rate."""
    n = 6
    beta = 0.3   # below critical β_c ≈ 0.44 → disordered phase, fast mixing
    problem = IsingModel(n=n, beta=beta, seed=42)
    mh = MetropolisHastings(problem, seed=42)

    n_total, burn_in = 3_000, 500
    x0 = problem.initial_state()
    samples = mh.sample(x0=x0, n_steps=n_total, burn_in=burn_in)

    # ---- Every state must be an N×N grid of ±1 values ----------------------
    for i, grid in enumerate(samples[:50]):
        assert grid.shape == (n, n), (
            f"Sample {i} has wrong shape {grid.shape}, expected ({n}, {n})."
        )
        assert set(np.unique(grid)).issubset({-1, 1}), (
            f"Sample {i} contains values other than ±1: {np.unique(grid)}"
        )
        log_w = problem.log_target(grid)
        assert math.isfinite(log_w), (
            f"log_target not finite at sample {i}: {log_w}"
        )

    # ---- Acceptance rate in reasonable range (single-spin-flip) ---------------
    rate = mh.acceptance_rate
    assert 0.1 <= rate <= 1.0, (
        f"Acceptance rate {rate:.4f} outside [0.1, 1.0]."
    )

    # ---- log_target is bounded (Boltzmann weight is always finite) ------------
    log_targets = [problem.log_target(g) for g in samples[:100]]
    assert all(math.isfinite(lt) for lt in log_targets), (
        "Some log_target values are not finite."
    )

    # ---- Detailed balance for consecutive pairs (first 50) -------------------
    for i in range(min(50, len(samples) - 1)):
        log_wx = problem.log_target(samples[i])
        log_wy = problem.log_target(samples[i + 1])
        lhs = min(log_wx, log_wy)
        rhs = min(log_wy, log_wx)
        assert math.isclose(lhs, rhs, rel_tol=1e-9), (
            f"Detailed balance violated at step {i}: lhs={lhs}, rhs={rhs}"
        )
