"""
ising_demo.py — 2D Ising model walkthrough.

Demonstrates the core MCMC simulator workflow:

  1. Define a target distribution by providing a state space and weights.
     Here: all N×N spin grids, weighted by exp(β * neighbor_alignment).
  2. Run the Metropolis-Hastings engine.
  3. Inspect chain quality via diagnostics.
  4. Visualize samples.

This is the "beginner" example. For a real-world use case see finance_demo.py.

Usage
-----
    conda activate mcmc
    python ising_demo.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from core.mh import MetropolisHastings
from core.diagnostics import acceptance_rate, effective_sample_size, autocorrelation
from problems.ising import IsingModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N      = 10          # grid side length (10×10 = 100 spins)
BETA   = 0.30        # inverse temperature (below critical β_c ≈ 0.44)
SEED   = 42
N_STEPS  = 10_000
BURN_IN  = 2_000
SNAPSHOT_EVERY = 500    # save one grid image every N steps


# ---------------------------------------------------------------------------
# Step 1: Define the problem
# ---------------------------------------------------------------------------

model = IsingModel(n=N, beta=BETA, seed=SEED)
print(f"State space: {N}×{N} spin grid ({2**(N*N):.2e} possible states)")
print(f"Target:      Boltzmann distribution at β={BETA}")
print(f"Proposal:    Single-spin flip (symmetric)")
print()


# ---------------------------------------------------------------------------
# Step 2: Run the MH chain
# ---------------------------------------------------------------------------

mh = MetropolisHastings(model, seed=SEED)
x0 = model.initial_state()

print(f"Running {N_STEPS} steps ({BURN_IN} burn-in) …")
samples = mh.sample(x0=x0, n_steps=N_STEPS, burn_in=BURN_IN)
print(f"Collected {len(samples)} post-burn-in samples.")
print()


# ---------------------------------------------------------------------------
# Step 3: Diagnostics
# ---------------------------------------------------------------------------

acc   = acceptance_rate(mh.n_accepted, mh.n_proposed)
ess   = effective_sample_size([float(np.sum(g)) for g in samples])   # ESS of magnetization
acf   = autocorrelation([float(np.sum(g)) for g in samples])

print(f"Acceptance rate : {acc:.3f}")
print(f"ESS (magnetiz.) : {ess:.1f}  (out of {len(samples)} samples)")
print(f"ACF lags stored : {len(acf)}")
print()


# ---------------------------------------------------------------------------
# Step 4: Visualization
# ---------------------------------------------------------------------------

# Select evenly-spaced grid snapshots for display
n_show = min(6, len(samples))
indices = np.linspace(0, len(samples) - 1, n_show, dtype=int)

# Magnetization trace
magnetizations = [float(np.sum(g)) / (N * N) for g in samples]

fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))

for col, idx in enumerate(indices):
    ax = axes[0, col]
    ax.imshow(samples[idx], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"Step {idx + BURN_IN}", fontsize=8)
    ax.axis("off")

ax_trace = axes[1, :]
# Merge bottom row into one axes for the trace
for ax in ax_trace[1:]:
    ax.set_visible(False)
ax_trace[0].set_position([0.08, 0.08, 0.88, 0.38])
ax_trace[0].plot(magnetizations, color="steelblue", linewidth=0.8, alpha=0.9)
ax_trace[0].axhline(0, color="crimson", linewidth=1, linestyle="--", label="M = 0")
ax_trace[0].set_xlabel("Sample index (post burn-in)")
ax_trace[0].set_ylabel("Magnetization M = Σσ / N²")
ax_trace[0].set_title(
    f"Ising {N}×{N}, β={BETA}  |  acc={acc:.3f}  ESS={ess:.0f}",
    fontsize=10,
)
ax_trace[0].legend(fontsize=8)

plt.tight_layout()
out_path = "output/ising_results.png"
import os; os.makedirs("output", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out_path}")
plt.show()
