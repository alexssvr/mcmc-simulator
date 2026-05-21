# MCMC Simulator — Project Context

## What This Project Does
A modular Metropolis-Hastings MCMC simulator. Any distribution or
sampling problem plugs in via the SamplingProblem interface.
The MH engine is shared across all problems.

## Mathematical Foundation (from CS 4850, pvmc.pdf)
- MH transition (Eq 6.1): P_xy = K_xy * κ_x * min(w_x, w_y) / w_x  for x ≠ y
- Diagonal: P_xx = 1 - Σ_{y≠x} P_xy
- Proposal K must satisfy reversibility w.r.t. κ: κ_x * K_xy = κ_y * K_yx
- Acceptance probability: min(w_x, w_y) / w_x  * κ_x
- Stationary distribution: π = w / Z_w  (by Lemma 6.12)

## Code Conventions
- Python 3.11+, type hints on everything
- `SamplingProblem` is the core ABC in core/problem.py
- New distributions go in problems/
- All tests in tests/, using pytest
- Work in log-space for numerical stability (log_target, not target)

## Project Structure
mcmc-simulator/
├── CLAUDE.md
├── .claude/agents/       ← subagent definitions live here
├── core/
│   ├── problem.py        ← SamplingProblem ABC
│   ├── mh.py             ← MH engine (Eq 6.1 exactly)
│   └── diagnostics.py   ← TV distance, mixing time, autocorrelation
├── problems/
│   ├── glauber.py
│   ├── ising.py
│   └── finance/
│       └── autocallable.py
└── tests/

## What NOT to Do
- Never change the MH engine to "simplify" the math — it must match Eq 6.1
- Never use non-log-space in the acceptance step (numerical underflow)
- Never break the SamplingProblem interface

## Environment
- Conda env: mcmc (Python 3.12)
- Activate with: conda activate mcmc
- Key deps: numpy, scipy, matplotlib, pandas, pytest
- No external API calls anywhere in this project — fully local/offline