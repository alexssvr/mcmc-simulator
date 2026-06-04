# MCMC Simulator — Project Context

## What This Project Does
A generic Metropolis-Hastings MCMC simulator. Given any problem context —
a set of states and their (unnormalized) weights — the engine samples from
the distribution proportional to those weights.

Any new problem plugs in via the SamplingProblem interface. The MH engine
in core/mh.py is shared across all problems and never modified.

**Standalone — no API calls, no internet. Runs fully locally.**

## Mathematical Foundation (from CS 4850, pvmc.pdf)

### Definition 6.11 — Reversibility
A Markov chain with transition matrix P is **reversible** with respect to
(unnormalized) distribution w if it satisfies

    w_x * P_xy = w_y * P_yx     for all x, y ∈ X

### Lemma 6.12 — Stationary Distribution
If P is reversible with respect to w, then π = w / Z_w is a stationary
distribution for P.

Proof sketch: multiply the reversibility equation w_x * P_xy = w_y * P_yx
by Z_w^{-1} to get π_x * P_xy = π_y * P_yx, then sum over y to confirm
(πP)_x = π_x.

### Equation 6.1 — Metropolis-Hastings Transition Matrix
Given an unnormalized target w, an unnormalized proposal-weight κ, and a
proposal Markov chain K that is reversible with respect to κ, define for x ≠ y:

    P_xy = K_xy · κ_x · min(w_x, w_y) / w_x        (Eq 6.1)

and the diagonal:

    P_xx = 1 - Σ_{y ≠ x} P_xy

Because Σ_{y ≠ x} P_xy ≤ κ_x · Σ_{y ≠ x} K_xy ≤ κ_x ≤ 1, we have
P_xx ≥ 0, so P is a valid Markov transition matrix.

### Lemma 6.13 — Correctness of MH
The Markov chain P defined by Equation 6.1 is reversible with respect to w,
and therefore (by Lemma 6.12) has stationary distribution π = w / Z_w.

Proof: for x ≠ y,
    w_x * P_xy = K_xy · κ_x · min(w_x, w_y)
    w_y * P_yx = K_yx · κ_y · min(w_y, w_x)
These are equal because min is symmetric and K_xy · κ_x = K_yx · κ_y
(reversibility of K w.r.t. κ).

### Simulation Algorithm (from the notes, p. 105)
Given current state x:
1. Sample proposed state y from K_xy.
2. Compute w_x, w_y, and κ_x.
3. With probability  min(w_x, w_y) / w_x  · κ_x, transition to y.
4. Otherwise remain at x.

### Key identities used in core/mh.py
- Acceptance in log-space:
  log α = min(log_wx, log_wy) - log_wx + log_κ_x + log_K_yx - log_K_xy
- Symmetric proposals (all kernels here): K_xy = K_yx → log_K_yx - log_K_xy = 0
- Proposal K must satisfy reversibility w.r.t. κ: κ_x * K_xy = κ_y * K_yx

## Code Conventions
- Python 3.11+, type hints on everything
- `SamplingProblem` is the core ABC in core/problem.py
- New distributions go in problems/
- All tests in tests/, using pytest
- Work in log-space for numerical stability (log_target, not target)

## Project Structure
mcmc-simulator/
├── CLAUDE.md
├── core/
│   ├── problem.py           ← SamplingProblem ABC (the plug-in interface)
│   ├── mh.py                ← MH engine (Eq 6.1 exactly — NEVER MODIFY)
│   ├── diagnostics.py       ← ESS, autocorrelation, acceptance_rate
│   ├── classifier.py        ← auto-selects proposal kernel from state_description
│   └── configurator.py     ← builds ProblemSpec from config dicts or prompts
├── main.py                  ← MCMCSampler: single entry point for end-to-end runs
├── problems/
│   ├── weighted_discrete.py ← WeightedDiscreteProblem: any states + weights → samples
│   ├── discrete.py          ← DiscreteGraphProblem: uniform q-coloring of a graph
│   ├── continuous.py        ← DataDrivenProblem (KDE), FormulaProblem (log_w callable)
│   └── ising.py             ← IsingModel: 2D Ising model (beginner example)
├── ising_demo.py            ← beginner walkthrough: plug in states/weights, get samples
├── finance_demo.py          ← advanced example: autocallable option pricing via MCMC
└── tests/
    ├── test_mh.py           ← MH engine, Glauber, classifier, configurator
    ├── test_problems.py     ← DataDrivenProblem, FormulaProblem, Ising, WeightedDiscrete
    └── test_integration.py  ← end-to-end pipeline test (configurator → sampler → diagnostics)

## What NOT to Do
- Never change the MH engine to "simplify" the math — it must match Eq 6.1
- Never use non-log-space in the acceptance step (numerical underflow)
- Never break the SamplingProblem interface

## Environment
- Conda env: mcmc (Python 3.12)
- Activate with: conda activate mcmc
- Key deps: numpy, scipy, matplotlib, pandas, pytest
- No external API calls anywhere in this project — fully local/offline