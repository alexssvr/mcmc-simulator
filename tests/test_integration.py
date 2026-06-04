"""
tests/test_integration.py — End-to-end pipeline integration test.

Exercises the full stack:

    ProblemConfigurator.configure()
        → ProposalClassifier.classify()
        → MetropolisHastings.sample()
        → core.diagnostics.effective_sample_size()

Uses a standard-normal target (log_w = -0.5 * x²) as the reference problem.
Asserts that the pipeline produces valid samples with ESS > 100 and that
the diagnostics module functions agree with the chain statistics.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from core.classifier import ProposalClassifier
from core.configurator import ProblemConfigurator
from core.diagnostics import acceptance_rate, autocorrelation, effective_sample_size
from core.mh import MetropolisHastings
from problems.continuous import FormulaProblem
from problems.weighted_discrete import WeightedDiscreteProblem


def test_continuous_pipeline() -> None:
    """Full pipeline: configurator → classifier → MH → diagnostics (continuous)."""
    # ── Step 1: Configure ────────────────────────────────────────────────────
    spec = ProblemConfigurator().configure({
        "problem_type": "continuous_formula",
        "formula":      lambda x: -0.5 * float(x) ** 2,
        "dimension":    1,
        "bounds":       None,
        "notes":        "N(0,1) integration test",
    })
    assert spec.problem_type == "continuous_formula"
    assert spec.state_space_type == "continuous"

    # ── Step 2: Classify ─────────────────────────────────────────────────────
    state_desc = {
        "type":      spec.state_space_type,
        "dimension": spec.dimension,
        "bounds":    None,
        "notes":     spec.notes,
    }
    proposal_config = ProposalClassifier().classify(state_desc)
    assert proposal_config["type"] == "gaussian_random_walk"
    assert "recommended_burn_in" in proposal_config
    assert "recommended_n_samples" in proposal_config

    # ── Step 3: Build problem and run MH ─────────────────────────────────────
    problem = FormulaProblem(
        log_w=spec.formula,
        proposal_config=proposal_config,
        dimension=spec.dimension,
        bounds=spec.bounds,
        seed=42,
        notes=spec.notes,
    )
    mh = MetropolisHastings(problem, seed=42)
    n_total, burn_in = 6_000, 1_000
    samples = mh.sample(
        x0=problem.initial_state(), n_steps=n_total, burn_in=burn_in
    )
    assert len(samples) == n_total - burn_in

    # ── Step 4: Diagnostics ───────────────────────────────────────────────────
    acc = acceptance_rate(mh.n_accepted, mh.n_proposed)
    assert math.isclose(acc, mh.acceptance_rate, rel_tol=1e-9), (
        "diagnostics.acceptance_rate disagrees with mh.acceptance_rate"
    )
    assert 0.2 <= acc <= 0.8, f"Acceptance rate {acc:.4f} outside [0.2, 0.8]."

    ess = effective_sample_size(samples)
    assert ess > 100, f"ESS {ess:.1f} too low — chain may not be mixing."
    assert ess <= len(samples), f"ESS {ess:.1f} exceeds sample count {len(samples)}."

    acf = autocorrelation(samples)
    assert isinstance(acf, list), "autocorrelation should return a list."
    assert all(isinstance(v, float) for v in acf), "ACF values should be floats."
    if len(acf) > 0:
        assert acf[0] > 0, "First ACF value should be positive for a correlated chain."

    # ── Step 5: Statistical correctness (KS test) ────────────────────────────
    arr = np.array(samples, dtype=float)
    ks_stat, ks_pvalue = stats.kstest(arr, "norm", args=(0.0, 1.0))
    assert ks_pvalue > 0.01, (
        f"KS test failed: samples do not look like N(0, 1). "
        f"ks_stat={ks_stat:.4f}, p={ks_pvalue:.4f}"
    )


def test_weighted_discrete_pipeline() -> None:
    """Full pipeline: WeightedDiscreteProblem → MH → diagnostics (discrete)."""
    states  = ["A", "B", "C"]
    weights = [1.0, 3.0, 1.0]
    true_probs = [w / sum(weights) for w in weights]

    problem = WeightedDiscreteProblem(states=states, weights=weights, seed=0)
    mh = MetropolisHastings(problem, seed=0)

    n_total, burn_in = 20_000, 1_000
    samples = mh.sample(
        x0=problem.initial_state(), n_steps=n_total, burn_in=burn_in
    )
    n = len(samples)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    acc = acceptance_rate(mh.n_accepted, mh.n_proposed)
    assert 0.1 <= acc <= 1.0, f"Acceptance rate {acc:.4f} out of range."

    # ESS is hard to compute meaningfully for non-scalar states, but it should
    # not crash and should be positive.
    # Encode as integers first so ESS computation works
    state_idx = [states.index(s) for s in samples]
    ess = effective_sample_size(state_idx)
    assert ess > 0, f"ESS should be positive, got {ess}."

    # ── Frequency check ───────────────────────────────────────────────────────
    counts = {s: samples.count(s) for s in states}
    for i, s in enumerate(states):
        emp = counts[s] / n
        assert abs(emp - true_probs[i]) < 0.03, (
            f"State '{s}': empirical {emp:.4f} vs target {true_probs[i]:.4f}"
        )
