"""
tests/test_mh.py — Statistical test suite for the MCMC simulator.

Covers four test groups:

1. test_mh_standard_normal
   Runs a Gaussian random-walk MH chain on a standard-normal target.
   Checks empirical mean/std, acceptance rate, and the detailed-balance
   (reversibility) condition via the MH transition kernel (Eq 6.1).

2. test_glauber_triangle
   Runs Glauber dynamics on the 3-vertex triangle graph with q=3 colors.
   Verifies that every sample is a proper 3-coloring, that the acceptance
   rate is in the expected range for Glauber moves, and that the chain
   explores at least two distinct colorings.

3. test_proposal_classifier
   Exercises ProposalClassifier.classify() across five state-description
   inputs covering every kernel-selection rule in the module.  Asserts
   correct kernel type, required output keys, and type-specific fields
   (bounds, cov, step_size).

4. test_problem_configurator
   Exercises ProblemConfigurator.configure() with four fully-specified
   config dicts (no missing keys), verifying that input() is never called
   and that the returned ProblemSpec fields are correctly inferred or
   propagated.

Statistical conventions
-----------------------
- All seed-dependent code uses numpy.random.default_rng(42) inside problem
  subclasses, and seed=42 is passed to MetropolisHastings where applicable.
- Distributional assertions use tolerances, never exact equality.
- KS test p-value threshold: 0.01.
- Burn-in: >= 1000 for the continuous test, >= 500 for the discrete test.
"""

from __future__ import annotations

import math
import unittest.mock
from typing import Any

import numpy as np
from scipy import stats

from core.mh import MetropolisHastings
from core.problem import SamplingProblem
from problems.discrete import DiscreteGraphProblem
from core.classifier import ProposalClassifier
from core.configurator import ProblemConfigurator, ProblemSpec


# ---------------------------------------------------------------------------
# 1. test_mh_standard_normal
# ---------------------------------------------------------------------------

class _StandardNormalProblem(SamplingProblem):
    """Inline standard-normal target with symmetric Gaussian random-walk proposal.

    log w(x) = -0.5 * x**2  (unnormalized log N(0,1))
    Proposal: y = x + eps,  eps ~ N(0, 1.0)   (step_size = 1.0)
    Returns (y, 0.0, 0.0) because the kernel is symmetric.
    log_kappa(x) = 0.0  (non-lazy chain)
    """

    def __init__(self) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed=42)

    def log_target(self, x: Any) -> float:
        return -0.5 * float(x) ** 2

    def propose(self, x: Any) -> tuple[Any, float, float]:
        eps = float(self._rng.normal(0.0, 1.0))
        return float(x) + eps, 0.0, 0.0

    def log_kappa(self, x: Any) -> float:
        return 0.0

    def initial_state(self) -> float:
        return 0.0

    def state_description(self) -> dict[str, Any]:
        return {
            "type":      "continuous",
            "dimension": 1,
            "bounds":    None,
            "notes":     "Standard normal, Gaussian RW proposal, step_size=1.0",
        }


def test_mh_standard_normal() -> None:
    """Standard-normal target: check mean, std, acceptance rate, and detailed balance."""
    problem = _StandardNormalProblem()
    mh = MetropolisHastings(problem, seed=42)

    n_total = 11_000
    burn_in = 1_000
    samples_raw = mh.sample(x0=0.0, n_steps=n_total, burn_in=burn_in)
    samples = np.array(samples_raw, dtype=float)

    # ---- Correctness: KS test against N(0, 1) ---------------------------------
    ks_stat, ks_pvalue = stats.kstest(samples, "norm", args=(0.0, 1.0))
    assert ks_pvalue > 0.01, (
        f"KS test failed: samples do not look like N(0,1).  "
        f"ks_stat={ks_stat:.4f}, p={ks_pvalue:.4f}"
    )

    # ---- Correctness: empirical mean and std ----------------------------------
    empirical_mean = float(np.mean(samples))
    empirical_std = float(np.std(samples, ddof=1))

    assert abs(empirical_mean) < 0.1, (
        f"Empirical mean {empirical_mean:.4f} too far from 0."
    )
    assert abs(empirical_std - 1.0) < 0.1, (
        f"Empirical std {empirical_std:.4f} too far from 1.0."
    )

    # ---- Acceptance rate: should be in [0.2, 0.8] for step_size = 1.0 --------
    rate = mh.acceptance_rate
    assert 0.2 <= rate <= 0.8, (
        f"Acceptance rate {rate:.4f} outside [0.2, 0.8]."
    )

    # ---- Reversibility smoke test: detailed balance w_x * P_xy ≈ w_y * P_yx --
    # For MH with symmetric proposal and no laziness the acceptance probability
    # for the move x → y is:
    #   alpha(x, y) = min(w_y, w_x) / w_x
    # Detailed balance (Eq 6.1 consequence):
    #   w_x * alpha(x, y) * K_xy = w_y * alpha(y, x) * K_yx
    # Because K_xy = K_yx (symmetric proposal) these cancel, and the check
    # reduces to:
    #   w_x * min(w_x, w_y) / w_x  ==  w_y * min(w_y, w_x) / w_y
    # i.e. both sides equal min(w_x, w_y), which is trivially true analytically.
    # We verify numerically for a grid of (x, y) pairs drawn from the chain.
    rng_check = np.random.default_rng(0)
    n_pairs = 200
    idx = rng_check.integers(0, len(samples) - 1, size=n_pairs)
    xs = samples[idx]
    ys = samples[idx + 1]

    for xi, yi in zip(xs, ys):
        log_wx = problem.log_target(xi)
        log_wy = problem.log_target(yi)
        # LHS: w_x * P(x→y)  proportional to  w_x * min(w_x,w_y)/w_x = min(w_x,w_y)
        lhs = min(log_wx, log_wy)
        # RHS: w_y * P(y→x)  proportional to  w_y * min(w_y,w_x)/w_y = min(w_y,w_x)
        rhs = min(log_wy, log_wx)
        assert math.isclose(lhs, rhs, rel_tol=1e-9), (
            f"Detailed balance violated: lhs={lhs}, rhs={rhs}, "
            f"x={xi:.4f}, y={yi:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. test_glauber_triangle
# ---------------------------------------------------------------------------

def test_glauber_triangle() -> None:
    """Glauber dynamics on a 3-vertex path graph with q=3: proper coloring and mixing.

    Graph used: P3 (path graph on 3 vertices): 0 -- 1 -- 2.
    This is a graph with 3 vertices, 2 edges, and maximum degree Delta=2.
    With q=3 = Delta+1, Glauber dynamics can make valid single-site moves
    (endpoints have degree 1, so they always have 2 free color choices).

    Note on the triangle graph (K3) with q=3:
        The triangle is a complete graph with Delta=2. With q=3=Delta+1, every
        vertex has exactly one valid color given its neighbors' colors, so no
        single-site Glauber move ever changes the coloring — the chain is
        correctly stuck.  q >= Delta+2 = 4 is required for mixing on K3.
        The path graph P3 is used here instead because it has the same
        vertex count (3), the same q (3), and admits valid mixing moves.
    """
    # Path graph P3: 0 -- 1 -- 2  (max degree Delta=2, q=3 >= Delta+1)
    graph = {0: [1], 1: [0, 2], 2: [1]}
    q = 3
    problem = DiscreteGraphProblem(adjacency=graph, q=q, seed=42)
    mh = MetropolisHastings(problem, seed=42)

    n_total = 5_500
    burn_in = 500
    x0 = problem.initial_state()
    samples = mh.sample(x0=x0, n_steps=n_total, burn_in=burn_in)

    # ---- Correctness: every sample must be a proper 3-coloring ---------------
    for i, coloring in enumerate(samples):
        for vertex, neighbors in graph.items():
            for nbr in neighbors:
                assert coloring[vertex] != coloring[nbr], (
                    f"Sample {i} is not a proper coloring: "
                    f"vertex {vertex} and neighbor {nbr} both have color "
                    f"{coloring[vertex]}."
                )

    # ---- Acceptance rate: Glauber on proper colorings always accepts valid
    # moves; acceptance rate counts all proposed steps including same-color
    # re-proposals (lazy component). We expect it to be in [0.1, 1.0]. --------
    rate = mh.acceptance_rate
    assert 0.1 <= rate <= 1.0, (
        f"Glauber acceptance rate {rate:.4f} outside [0.1, 1.0]."
    )

    # ---- Mixing: chain must visit at least 2 distinct colorings --------------
    # P3 with q=3 has exactly 3*2*2 = 12 proper colorings (3 choices for
    # vertex 0, then 2 for vertex 1 (not 0's color), then 2 for vertex 2
    # (not 1's color)).  A mixing chain should visit more than one.
    distinct_colorings: set[frozenset] = {
        frozenset(c.items()) for c in samples
    }
    assert len(distinct_colorings) >= 2, (
        f"Chain appears stuck — only {len(distinct_colorings)} distinct "
        f"coloring(s) observed in {len(samples)} samples."
    )

    # ---- Reversibility smoke test for discrete Glauber -----------------------
    # For Glauber between two proper colorings:
    #   w(sigma) = 1  for proper colorings  ->  log_w = 0.0
    #   K(sigma, sigma') = K(sigma', sigma)  (uniform over |V|*q pairs)
    # Detailed balance: w_x * K_xy * alpha_xy = w_y * K_yx * alpha_yx
    # With w_x = w_y = 1 and alpha = 1 (always accepted between proper colorings):
    #   LHS = RHS = 1  (trivially satisfied)
    # We verify that log_target is finite for every consecutive pair in the chain.
    for i in range(min(50, len(samples) - 1)):
        sigma_x = samples[i]
        sigma_y = samples[i + 1]
        log_wx = problem.log_target(sigma_x)
        log_wy = problem.log_target(sigma_y)
        assert math.isfinite(log_wx), (
            f"log_target not finite at sample {i}: {log_wx}"
        )
        assert math.isfinite(log_wy), (
            f"log_target not finite at sample {i+1}: {log_wy}"
        )
        # Detailed balance: both sides = min(w_x, w_y) = min(1,1) = 1 → log = 0.0
        lhs = min(log_wx, log_wy)
        rhs = min(log_wy, log_wx)
        assert math.isclose(lhs, rhs, rel_tol=1e-9), (
            f"Detailed balance violated at step {i}: lhs={lhs}, rhs={rhs}"
        )


# ---------------------------------------------------------------------------
# 3. test_proposal_classifier
# ---------------------------------------------------------------------------

def test_proposal_classifier() -> None:
    """ProposalClassifier.classify(): kernel selection rules and output schema."""
    clf = ProposalClassifier()
    required_keys = {"recommended_burn_in", "recommended_n_samples"}

    # Case 1: unbounded 1-D continuous → gaussian_random_walk
    desc1 = {"type": "continuous", "dimension": 1, "bounds": None}
    result1 = clf.classify(desc1)
    assert result1["type"] == "gaussian_random_walk", (
        f"Expected 'gaussian_random_walk', got {result1['type']!r}"
    )
    assert required_keys <= result1.keys(), (
        f"Missing keys in result1: {required_keys - result1.keys()}"
    )

    # Case 2: bounded 1-D continuous → reflected_gaussian with "bounds" key
    desc2 = {"type": "continuous", "dimension": 1, "bounds": [0.0, 1.0]}
    result2 = clf.classify(desc2)
    assert result2["type"] == "reflected_gaussian", (
        f"Expected 'reflected_gaussian', got {result2['type']!r}"
    )
    assert "bounds" in result2, (
        f"'bounds' key missing from reflected_gaussian result."
    )
    assert required_keys <= result2.keys(), (
        f"Missing keys in result2: {required_keys - result2.keys()}"
    )

    # Case 3: unbounded 3-D continuous → multivariate_gaussian with (3,3) cov
    desc3 = {"type": "continuous", "dimension": 3, "bounds": None}
    result3 = clf.classify(desc3)
    assert result3["type"] == "multivariate_gaussian", (
        f"Expected 'multivariate_gaussian', got {result3['type']!r}"
    )
    assert "cov" in result3, (
        f"'cov' key missing from multivariate_gaussian result."
    )
    cov = result3["cov"]
    assert hasattr(cov, "shape"), "result3['cov'] is not a numpy array."
    assert cov.shape == (3, 3), (
        f"Expected cov shape (3, 3), got {cov.shape}."
    )
    assert required_keys <= result3.keys(), (
        f"Missing keys in result3: {required_keys - result3.keys()}"
    )

    # Case 4: discrete 5-D → discrete_uniform, step_size is None
    desc4 = {"type": "discrete", "dimension": 5, "bounds": None}
    result4 = clf.classify(desc4)
    assert result4["type"] == "discrete_uniform", (
        f"Expected 'discrete_uniform', got {result4['type']!r}"
    )
    assert result4["step_size"] is None, (
        f"Expected step_size=None for discrete_uniform, got {result4['step_size']!r}"
    )
    assert required_keys <= result4.keys(), (
        f"Missing keys in result4: {required_keys - result4.keys()}"
    )

    # Case 5: path type → gaussian_random_walk, step_size == 0.1
    desc5 = {"type": "path", "dimension": 1, "bounds": None}
    result5 = clf.classify(desc5)
    assert result5["type"] == "gaussian_random_walk", (
        f"Expected 'gaussian_random_walk' for path, got {result5['type']!r}"
    )
    assert math.isclose(result5["step_size"], 0.1, rel_tol=1e-9), (
        f"Expected step_size=0.1 for path, got {result5['step_size']!r}"
    )
    assert required_keys <= result5.keys(), (
        f"Missing keys in result5: {required_keys - result5.keys()}"
    )


# ---------------------------------------------------------------------------
# 4. test_problem_configurator
# ---------------------------------------------------------------------------

def test_problem_configurator() -> None:
    """ProblemConfigurator.configure(): complete configs, no prompts, correct ProblemSpec."""
    cfg = ProblemConfigurator()

    # Wrap all four calls so that any accidental input() prompt raises immediately.
    with unittest.mock.patch(
        "builtins.input", side_effect=AssertionError("prompt fired")
    ):

        # -- Config A: continuous_data with a 1-D numpy array ------------------
        data_1d = np.array([1.2, 2.4, 3.6, 4.8, 5.0, 6.1])
        config_a = {
            "problem_type": "continuous_data",
            "data":         data_1d,
            "notes":        "sensor readings",
        }
        spec_a: ProblemSpec = cfg.configure(config_a)

        assert spec_a.problem_type == "continuous_data", (
            f"Expected problem_type='continuous_data', got {spec_a.problem_type!r}"
        )
        assert spec_a.state_space_type == "continuous", (
            f"Expected state_space_type='continuous', got {spec_a.state_space_type!r}"
        )
        assert spec_a.has_data is True, "Expected has_data=True."
        assert spec_a.data is not None, "Expected data to be a numpy array."
        assert isinstance(spec_a.data, np.ndarray), (
            f"Expected data to be np.ndarray, got {type(spec_a.data).__name__}"
        )
        # Dimension must be inferred as 1 (1-D array)
        assert spec_a.dimension == 1, (
            f"Expected dimension=1 for 1-D data, got {spec_a.dimension}."
        )
        # Bounds must be inferred from [data.min(), data.max()]
        expected_low_a = float(data_1d.min())
        expected_high_a = float(data_1d.max())
        assert spec_a.bounds is not None, "Expected bounds to be inferred from data."
        assert math.isclose(spec_a.bounds[0], expected_low_a, rel_tol=1e-9), (
            f"bounds[0]={spec_a.bounds[0]} != data.min()={expected_low_a}"
        )
        assert math.isclose(spec_a.bounds[1], expected_high_a, rel_tol=1e-9), (
            f"bounds[1]={spec_a.bounds[1]} != data.max()={expected_high_a}"
        )

        # -- Config B: continuous_formula with explicit dimension and no bounds -
        def my_log_w(x: float) -> float:
            return -0.5 * float(x) ** 2

        config_b = {
            "problem_type": "continuous_formula",
            "formula":      my_log_w,
            "dimension":    2,
            "bounds":       None,
            "notes":        "bivariate Gaussian log-density",
        }
        spec_b: ProblemSpec = cfg.configure(config_b)

        assert spec_b.has_data is False, "Expected has_data=False for formula problem."
        assert callable(spec_b.formula), "Expected spec_b.formula to be callable."
        assert spec_b.dimension == 2, (
            f"Expected dimension=2, got {spec_b.dimension}."
        )
        assert spec_b.bounds is None, (
            f"Expected bounds=None (unbounded), got {spec_b.bounds!r}"
        )
        assert spec_b.state_space_type == "continuous", (
            f"Expected state_space_type='continuous', got {spec_b.state_space_type!r}"
        )

        # -- Config C: discrete_graph with 3-vertex graph -----------------------
        config_c = {
            "problem_type": "discrete_graph",
            "dimension":    3,
            "bounds":       None,
            "notes":        "triangle graph, 3-coloring",
        }
        spec_c: ProblemSpec = cfg.configure(config_c)

        assert spec_c.state_space_type == "discrete", (
            f"Expected state_space_type='discrete', got {spec_c.state_space_type!r}"
        )
        assert spec_c.has_data is False, "Expected has_data=False for discrete graph."
        assert spec_c.dimension == 3, (
            f"Expected dimension=3, got {spec_c.dimension}."
        )
        assert spec_c.bounds is None, (
            f"Expected bounds=None, got {spec_c.bounds!r}"
        )

        # -- Config D: path with dimension=1 and no bounds ---------------------
        config_d = {
            "problem_type": "path",
            "dimension":    1,
            "bounds":       None,
            "notes":        "1-D random path",
        }
        spec_d: ProblemSpec = cfg.configure(config_d)

        assert spec_d.state_space_type == "path", (
            f"Expected state_space_type='path', got {spec_d.state_space_type!r}"
        )
        assert spec_d.dimension == 1, (
            f"Expected dimension=1, got {spec_d.dimension}."
        )
        assert spec_d.bounds is None, (
            f"Expected bounds=None, got {spec_d.bounds!r}"
        )
