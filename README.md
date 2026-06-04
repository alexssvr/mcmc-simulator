# MCMC Simulator

A generic, standalone Metropolis-Hastings MCMC simulator. You provide a set of states and their weights — the engine samples from the distribution proportional to those weights. No external APIs. No internet. Runs entirely on your machine.

Built for Cornell CS 4850. The implementation follows **Equation 6.1** from the course notes exactly:

```
P_xy = K_xy · κ_x · min(w_x, w_y) / w_x     for x ≠ y
```

with full log-space arithmetic to prevent numerical underflow.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [User Guide — Inputs and What You Get Back](#user-guide--inputs-and-what-you-get-back)
   - [Mode 1 — Any States + Weights (the generic interface)](#mode-1--any-states--weights-the-generic-interface)
   - [Mode 2 — Sample from Data (KDE)](#mode-2--sample-from-data-kde)
   - [Mode 3 — Sample from a Formula](#mode-3--sample-from-a-formula)
   - [Mode 4 — Discrete Graph Coloring](#mode-4--discrete-graph-coloring)
   - [Interactive (No-Config) Mode](#interactive-no-config-mode)
5. [Reading Your Results](#reading-your-results)
6. [Diagnostics](#diagnostics)
7. [Plotting](#plotting)
8. [Configuration Reference](#configuration-reference)
9. [Guidelines and Restrictions](#guidelines-and-restrictions)
10. [Examples](#examples)
    - [Ising Model (Beginner)](#ising-model-beginner)
    - [Autocallable Option Pricing (Advanced)](#autocallable-option-pricing-advanced)
11. [Writing Your Own Problem](#writing-your-own-problem)
12. [How the Math Works](#how-the-math-works)
13. [Running Tests](#running-tests)
14. [References](#references)

---

## What This Project Does

**Input:** A description of your problem — either explicit states with weights, observed data, a log-density formula, or a graph.

**Output:** A list of samples drawn from the distribution proportional to your weights, plus chain quality diagnostics (acceptance rate, effective sample size, autocorrelation).

The core engine is a **Metropolis-Hastings MCMC sampler**. Given a target distribution that you can evaluate (but not necessarily sample from directly), MH constructs a Markov chain whose stationary distribution equals your target. After a burn-in phase, the chain's states are approximate samples from your distribution.

**What you get back from every run:**

```
result.samples            → list of post-burn-in states (the actual samples)
result.acceptance_rate    → fraction of proposed moves accepted (health check)
result.diagnostics        → dict with ESS, autocorrelations, n_samples
result.problem_spec       → full configuration that was used
result.proposal_config    → proposal kernel type, step size, burn-in, n_samples
```

You can then analyze the samples however you like — compute statistics, plot histograms, estimate probabilities, price financial instruments, etc.

---

## Project Structure

```
mcmc-simulator/
├── main.py                  ← MCMCSampler: single entry point for end-to-end runs
├── core/
│   ├── problem.py           ← SamplingProblem ABC (the plug-in interface)
│   ├── mh.py                ← MH engine (Eq 6.1 exactly — never modified)
│   ├── diagnostics.py       ← ESS, autocorrelation, acceptance_rate functions
│   ├── classifier.py        ← auto-selects proposal kernel from state description
│   └── configurator.py      ← builds ProblemSpec from config dicts or prompts
├── problems/
│   ├── weighted_discrete.py ← WeightedDiscreteProblem: any states + weights → samples
│   ├── discrete.py          ← DiscreteGraphProblem: uniform q-coloring of a graph
│   ├── continuous.py        ← DataDrivenProblem (KDE), FormulaProblem (log_w callable)
│   └── ising.py             ← IsingModel: 2D Ising model (beginner example)
├── ising_demo.py            ← beginner walkthrough: states/weights → samples
├── finance_demo.py          ← advanced example: autocallable option pricing
└── tests/
    ├── test_mh.py           ← MH engine, Glauber, classifier, configurator (4 tests)
    ├── test_problems.py     ← DataDrivenProblem, FormulaProblem, Ising, WeightedDiscrete (6 tests)
    └── test_integration.py  ← end-to-end pipeline test (2 tests)
```

---

## Setup

**Prerequisites:** [Anaconda](https://www.anaconda.com/) or Miniconda.

```bash
# 1. Clone the repository
git clone https://github.com/alexssvr/mcmc-simulator.git
cd mcmc-simulator

# 2. Create and activate the conda environment
conda create -n mcmc python=3.12 numpy scipy matplotlib pandas pytest -y
conda activate mcmc

# 3. Verify the tests pass
pytest tests/ -v
```

You should see **12 passed** in under 5 seconds.

---

## User Guide — Inputs and What You Get Back

All interaction goes through `MCMCSampler.run(config)`. Pass a Python dict describing your problem. The sampler fills in defaults for anything missing, prints a summary, asks for confirmation, then returns a `SamplerResult`.

---

### Mode 1 — Any States + Weights (the generic interface)

**This is the primary interface.** Use it whenever you have a finite set of possible outcomes and a weight (importance, probability, or score) for each.

```python
from main import MCMCSampler

result = MCMCSampler().run({
    "problem_type": "weighted_discrete",
    "states":       ["rainy", "cloudy", "sunny"],
    "weights":      [0.3, 0.5, 0.2],     # unnormalized — they don't need to sum to 1
    "notes":        "weather model",
})

print(result.samples[:10])   # e.g. ['cloudy', 'sunny', 'cloudy', 'cloudy', ...]
```

**You can also use it directly without `MCMCSampler`:**

```python
from problems.weighted_discrete import WeightedDiscreteProblem
from core.mh import MetropolisHastings

states  = ["A", "B", "C", "D"]
weights = [1.0, 4.0, 4.0, 1.0]   # B and C are 4× more likely than A and D

problem = WeightedDiscreteProblem(states=states, weights=weights, seed=42)
mh      = MetropolisHastings(problem, seed=42)
samples = mh.sample(
    x0      = problem.initial_state(),
    n_steps = 10_000,
    burn_in = 1_000,
)
# samples is a list of 9000 states drawn from the distribution proportional to weights
```

**What you provide:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `states` | `list` | Any hashable objects: strings, ints, tuples, frozensets, etc. |
| `weights` | `list[float]` | Non-negative numbers, one per state. Need not sum to 1. |

**What you get back:**

`samples` is a list of states. Each state appears with frequency proportional to its weight. For the example above, B and C will appear roughly 4× as often as A and D.

**Use this mode when:**
- You have a finite set of outcomes with known relative likelihoods
- You're modeling a discrete distribution (die rolls, weather states, categories)
- You want to draw samples proportional to arbitrary scores or rewards

---

### Mode 2 — Sample from Data (KDE)

Use this when you have observed data and want to draw new samples from the same distribution. The simulator fits a **Gaussian kernel density estimate (KDE)** to your data and samples from that.

```python
import numpy as np
from main import MCMCSampler

# Load your data — any 1-D NumPy array
data = np.loadtxt("data/returns.csv")

result = MCMCSampler().run({
    "problem_type": "continuous_data",
    "data":         data,
    "notes":        "Daily stock returns",
})
```

**What happens automatically:**
- Fits a KDE to `data`
- Detects dimension (1-D or multi-D)
- Sets step size to `2.38 × std(data)` — optimal for ~44% acceptance (Roberts et al. 1997)
- Runs 1 000 burn-in steps + 5 000 sampling steps

**Multi-dimensional data** — pass a 2-D array with shape `(n_observations, d_dimensions)`:

```python
data_2d = np.column_stack([returns, volumes])   # shape (n, 2)
result = MCMCSampler().run({
    "problem_type": "continuous_data",
    "data":         data_2d,
})
```

**What you get back:**

`result.samples` is a list of floats (1-D) or NumPy arrays (multi-D), distributed approximately like your input data.

---

### Mode 3 — Sample from a Formula

Use this when you know the (unnormalized) log-density analytically.

```python
from main import MCMCSampler

result = MCMCSampler().run({
    "problem_type": "continuous_formula",
    "formula":      lambda x: -0.5 * x**2,   # log w(x) for N(0,1) — no normalization needed
    "dimension":    1,
    "notes":        "Standard Normal",
})
```

**Rules for the `formula` argument:**

| Rule | Example |
|------|---------|
| Must accept a single argument `x` | `lambda x: ...` |
| Must return the **log** of the unnormalized weight | `-0.5 * x**2` for N(0,1) |
| For 1-D: `x` is a `float` | `lambda x: -x**2 / 2` |
| For multi-D: `x` is a NumPy array shape `(d,)` | `lambda x: -0.5 * float(np.dot(x, x))` |
| May return `-math.inf` for out-of-support states | `lambda x: 0.0 if x > 0 else -math.inf` |
| Must never return `+inf` or `nan` | — |

**Bounded domains** — pass `bounds` to keep the chain inside an interval:

```python
import math
from main import MCMCSampler

result = MCMCSampler().run({
    "problem_type": "continuous_formula",
    "formula":      lambda x: math.log(x) - x,   # log of Gamma-like density
    "dimension":    1,
    "bounds":       [0.01, 20.0],                 # physical constraint on the state space
    "notes":        "Gamma-like on (0, 20)",
})
```

With `bounds`, the sampler switches to a **reflected Gaussian** proposal that bounces off the boundaries.

**What you get back:**

`result.samples` is a list of floats (1-D) or NumPy arrays (multi-D).

---

### Mode 4 — Discrete Graph Coloring

Use this to sample uniformly from all proper q-colorings of an undirected graph via Glauber dynamics.

```python
from main import MCMCSampler

# Adjacency list — every edge must appear in BOTH directions
triangle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

result = MCMCSampler().run({
    "problem_type": "discrete_graph",
    "adjacency":    triangle,
    "q":            4,
    "notes":        "Triangle graph, 4 colors",
})
```

**Edge-list shorthand:**

```python
result = MCMCSampler().run({
    "problem_type": "discrete_graph",
    "data": {
        "n_vertices": 5,
        "n_colors":   4,
        "edges":      [(0,1), (1,2), (2,3), (3,4), (4,0)],   # 5-cycle
    },
})
```

**What you get back:**

`result.samples` is a list of dicts `{vertex_id: color_index}`, each representing one valid coloring of the graph.

---

### Interactive (No-Config) Mode

Run the sampler with no arguments and it prompts for everything at the terminal:

```bash
conda activate mcmc
python -c "from main import MCMCSampler; MCMCSampler().run()"
```

You will be asked: problem type → data / formula / dimension / bounds / notes → confirmation.

For the formula prompt, type any Python expression in `x`, e.g. `-0.5*x**2`.

---

## Reading Your Results

`MCMCSampler.run()` returns a `SamplerResult` dataclass:

```python
result = MCMCSampler().run(config)

result.samples                               # list — the actual post-burn-in states
result.acceptance_rate                       # float in [0, 1]

result.diagnostics["acceptance_rate"]        # same as above
result.diagnostics["effective_sample_size"]  # ESS — accounts for autocorrelation
result.diagnostics["n_samples"]              # number of post-burn-in samples
result.diagnostics["autocorrelations"]       # list of ACF values used in ESS computation

result.problem_spec                          # ProblemSpec — type, dimension, bounds, etc.
result.proposal_config                       # dict — kernel type, step size, burn-in, n_samples
```

---

## Diagnostics

You can compute diagnostics directly from any sample list using `core.diagnostics`:

```python
from core.diagnostics import acceptance_rate, effective_sample_size, autocorrelation

# After running a chain manually
acc  = acceptance_rate(mh.n_accepted, mh.n_proposed)
ess  = effective_sample_size(samples)         # works for floats, numpy arrays, or dicts
acf  = autocorrelation(samples, max_lag=50)  # returns list of ACF values at lags 1, 2, ...

print(f"Acceptance rate: {acc:.3f}")
print(f"ESS: {ess:.1f} out of {len(samples)} samples")
print(f"ACF at lag 1: {acf[0]:.4f}" if acf else "ACF: no autocorrelation detected")
```

**Interpreting acceptance rate:**

| Range | Meaning |
|-------|---------|
| `< 0.1` | Step size too large — most proposals rejected; chain barely moves |
| `0.2 – 0.5` | Good for continuous 1-D; near-optimal mixing |
| `0.5 – 0.8` | Acceptable; chain mixes but may explore slowly |
| `> 0.9` | Step size too small — proposals always accepted; slow exploration |

**Interpreting ESS (Effective Sample Size):**

ESS corrects for autocorrelation between consecutive samples. An ESS of 500 from 5 000 samples means you effectively have ~500 independent draws. ESS ≥ n/10 is generally considered healthy.

---

## Plotting

```python
# Display in a window
sampler.plot(result)

# Save to a file
sampler.plot(result, save_path="output/my_plot.png")
```

What you get:

| Problem type | Plot |
|-------------|------|
| Continuous 1-D | Histogram with KDE overlay; title shows n, acceptance rate, ESS |
| Continuous multi-D | One subplot per dimension (marginal histograms + KDE) |
| Discrete (graph) | Bar chart of color frequencies per vertex (up to 6 vertices) |

---

## Configuration Reference

Full list of keys accepted by `MCMCSampler.run(config)`:

| Key | Type | Required for | Description |
|-----|------|-------------|-------------|
| `problem_type` | `str` | all | `"weighted_discrete"`, `"continuous_data"`, `"continuous_formula"`, `"discrete_graph"` |
| `states` | `list` | `weighted_discrete` | List of hashable state objects |
| `weights` | `list[float]` | `weighted_discrete` | Non-negative weights, one per state |
| `data` | `np.ndarray` | `continuous_data` | Observed samples, shape `(n,)` or `(n, d)` |
| `formula` | `callable` | `continuous_formula` | `lambda x: log_w(x)` — returns log unnormalized weight |
| `dimension` | `int` | multi-D formula | State dimension (inferred from `data` if omitted) |
| `bounds` | `[float, float]` | optional | Physical bounds `[low, high]`; triggers reflected kernel |
| `adjacency` | `dict[int, list[int]]` | `discrete_graph` | Adjacency list, every edge in both directions |
| `q` | `int` | `discrete_graph` | Number of colors |
| `notes` | `str` | optional | Free-form label shown in plots and summaries |

### Proposal kernels (chosen automatically)

| Problem type | Condition | Kernel |
|-------------|-----------|--------|
| `weighted_discrete` | any | Uniform random (symmetric) |
| `continuous_data` / `continuous_formula` | 1-D, no bounds | Gaussian random walk |
| `continuous_data` / `continuous_formula` | 1-D, with bounds | Reflected Gaussian |
| `continuous_data` / `continuous_formula` | multi-D, no bounds | Multivariate Gaussian |
| `discrete_graph` | any | Glauber dynamics |

---

## Guidelines and Restrictions

### Things you must follow

**1. Weights must be non-negative.** For `weighted_discrete`, all weights must be `>= 0` and at least one must be `> 0`. Zero-weight states are treated as out-of-support (never visited).

**2. The formula must return log-weights, not raw weights.** For `continuous_formula`, your callable must return `log w(x)`, not `w(x)` directly. This is required for numerical stability. For example:
```python
# Correct — return log of the weight
lambda x: -0.5 * x**2        # log of N(0,1) unnormalized

# Wrong — returns the raw weight, not the log
lambda x: math.exp(-0.5 * x**2)   # DO NOT DO THIS
```

**3. The starting state must be in the support.** `log_target(initial_state())` must be finite. For `FormulaProblem`, the default start is `x = 0`; if your target has no mass at the origin, pass a valid `x0` manually:
```python
samples = mh.sample(x0=5.0, n_steps=10_000, burn_in=1_000)
```

**4. For graph coloring, `q >= max_degree + 1`.** The greedy algorithm that finds a valid starting coloring requires at least `max_degree + 1` colors. For reliable mixing, use `q >= max_degree + 2`.

**5. Bounds are physical constraints, not data ranges.** Pass `bounds` only when there is a hard physical constraint on the state space (e.g., a probability must lie in [0, 1]). Do not pass `bounds` just because your data happens to fall within a range — the sampler infers that automatically.

### Guidelines for good results

**Acceptance rate:** Aim for 20–50% for continuous problems. If the acceptance rate is outside this range, the step size may need tuning. You can set it manually via `proposal_config`:
```python
from problems.continuous import FormulaProblem
problem = FormulaProblem(
    log_w=lambda x: -0.5 * x**2,
    proposal_config={"type": "gaussian_random_walk", "step_size": 0.5},
)
```

**Burn-in:** The sampler's default burn-in (1 000 steps for 1-D, scaled up for multi-D and discrete problems) is conservative but safe. For long-range or high-dimensional targets, increase it:
```python
mh.sample(x0=x0, n_steps=20_000, burn_in=5_000)
```

**ESS / autocorrelation:** If ESS is much lower than n/10, the chain is mixing slowly. Try increasing step size or running more steps.

**Discrete states must be hashable.** `WeightedDiscreteProblem` uses a dict internally for O(1) weight lookups. States must be hashable Python objects (strings, ints, tuples, frozensets). Numpy arrays are not hashable — convert them to tuples first:
```python
states = [tuple(row) for row in my_state_array]
```

**The MH engine is never modified.** The acceptance step in `core/mh.py` is fixed and implements Eq 6.1 exactly. Do not change it. All customization goes through the `SamplingProblem` interface.

---

## Examples

### Ising Model (Beginner)

The Ising model is the simplest way to see the full workflow. States are N×N spin grids (+1/−1); weights come from the Boltzmann distribution `exp(β * Σ neighbor_alignment)`.

```bash
conda activate mcmc
python ising_demo.py
```

This produces:
- Grid snapshots at various steps along the chain
- A magnetization trace showing the chain's evolution
- Chain diagnostics (acceptance rate, ESS)
- A PNG saved to `output/ising_results.png`

Or use it directly:

```python
from problems.ising import IsingModel
from core.mh import MetropolisHastings
from core.diagnostics import effective_sample_size
import numpy as np

model   = IsingModel(n=8, beta=0.44, seed=0)   # near the critical temperature
mh      = MetropolisHastings(model, seed=0)
samples = mh.sample(model.initial_state(), n_steps=5_000, burn_in=1_000)

# samples is a list of (8, 8) numpy int8 arrays with values ±1
magnetizations = [float(np.sum(g)) / 64 for g in samples]
print(f"Mean magnetization: {np.mean(magnetizations):.3f}")
print(f"ESS: {effective_sample_size(magnetizations):.1f}")
```

### Autocallable Option Pricing (Advanced)

A two-pass MCMC simulation that prices a structured financial product:

```bash
conda activate mcmc
python finance_demo.py
```

Pass 1 samples daily log-returns from a KDE fitted to 15 years of S&P 500 data.
Pass 2 samples quarterly risk-neutral paths and computes the expected payoff of an autocallable note.

Output: fair value as % of notional with a 95% confidence interval, plus a 4-panel plot saved to `output/autocallable_results.png`.

---

## Writing Your Own Problem

To add a new target distribution, subclass `SamplingProblem` in `problems/`:

```python
import math
import numpy as np
from core.problem import SamplingProblem

class MyProblem(SamplingProblem):

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def log_target(self, x):
        # Return log w(x) — the log unnormalized weight at state x
        # Return -math.inf for out-of-support states
        return -0.5 * float(x)**2   # example: standard normal

    def propose(self, x):
        # Draw candidate y from proposal kernel K(x, ·)
        # Return (y, log_K_xy, log_K_yx)
        # For symmetric kernels, log_K_xy == log_K_yx — return (y, 0.0, 0.0)
        y = float(x) + float(self._rng.normal(0, 1))
        return y, 0.0, 0.0

    def log_kappa(self, x):
        # Return log κ(x) — the log holding probability
        # For a non-lazy chain with symmetric proposal, always return 0.0
        return 0.0

    def initial_state(self):
        # Return a valid starting state (log_target must be finite here)
        return 0.0

    def state_description(self):
        return {
            "type":      "continuous",
            "dimension": 1,
            "bounds":    None,
            "notes":     "My custom problem",
        }
```

Then use it directly with the MH engine:

```python
from core.mh import MetropolisHastings

problem = MyProblem(seed=42)
mh      = MetropolisHastings(problem, seed=42)
samples = mh.sample(problem.initial_state(), n_steps=10_000, burn_in=1_000)
```

**The five methods you must implement:**

| Method | What it returns | Key contract |
|--------|----------------|--------------|
| `log_target(x)` | `float` — log unnormalized weight at `x` | May be `-inf`; never `+inf` or `nan` |
| `propose(x)` | `(y, log_K_xy, log_K_yx)` | For symmetric kernels: `(y, 0.0, 0.0)` |
| `log_kappa(x)` | `float` — log holding probability | Return `0.0` for standard non-lazy chains |
| `initial_state()` | valid starting state | `log_target(initial_state())` must be finite |
| `state_description()` | dict with `type`, `dimension`, `bounds`, `notes` | Used by classifier and diagnostics |

---

## How the Math Works

The simulator implements the **Metropolis-Hastings algorithm** (Eq 6.1, CS 4850 notes).

At each step, given current state `x`:

1. **Propose** a candidate `y` from the proposal kernel `K(x, ·)`
2. **Compute** the log-acceptance probability (all in log-space):

   ```
   log α = min(log w(y), log w(x)) − log w(x) + log κ(x) + log K(y→x) − log K(x→y)
   ```

3. **Accept** `y` (return `y`) if `log U < log α` where `U ~ Uniform(0,1)`; otherwise stay at `x`

The chain converges to the stationary distribution `π ∝ w` (Lemma 6.12, Lemma 6.13 in the notes).

Everything is computed in **log-space** to avoid floating-point underflow when `w(x)` is extremely small.

All three proposal kernels used here are **symmetric** (`K(x→y) = K(y→x)`), so the Hastings correction term `log K(y→x) − log K(x→y)` cancels to zero. The acceptance probability simplifies to `min(0, log w(y) − log w(x))`.

**Step size calibration:** for Gaussian random-walk proposals, the default step size is `2.38 × std(data)`. This comes from Roberts, Gelman & Gilks (1997): for a Gaussian target N(0, σ²), the proposal std `h* = 2.38σ` maximises the expected squared jump distance and achieves the theoretically optimal **44% acceptance rate** in 1-D.

---

## Running Tests

```bash
conda activate mcmc
pytest tests/ -v
```

**12 tests** across three files:

| Test | What it checks |
|------|----------------|
| `test_mh_standard_normal` | N(0,1) formula chain: KS test, mean ≈ 0, std ≈ 1, acceptance in [0.2, 0.8], detailed balance |
| `test_glauber_triangle` | Glauber on P3 path graph: all samples are proper colorings; chain mixes |
| `test_proposal_classifier` | All 5 kernel-selection rules fire correctly |
| `test_problem_configurator` | All 4 problem types configure without interactive prompts |
| `test_data_driven_problem` | KDE chain: empirical mean/std match data moments; acceptance in range |
| `test_formula_problem` | N(0,1) formula via `FormulaProblem`: KS test passes |
| `test_reflected_gaussian_proposal` | Bounded target: no samples escape `[0, 1]` |
| `test_weighted_discrete_problem` | Empirical frequencies match target weights within 3% |
| `test_weighted_discrete_validation` | Constructor rejects bad inputs (negative weights, mismatched lengths) |
| `test_ising_model` | Valid ±1 states, finite log-targets, detailed balance for 50 consecutive pairs |
| `test_continuous_pipeline` | Full pipeline: configurator → classifier → MH → diagnostics; ESS > 100 |
| `test_weighted_discrete_pipeline` | Full discrete pipeline: chain frequencies match target within 3% |

---

## References

- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). *Weak convergence and optimal scaling of random walk Metropolis algorithms.* Annals of Applied Probability, 7(1), 110–120.
- Jerrum, M. (1995). *A very simple algorithm for estimating the number of k-colorings of a low-degree graph.* Random Structures & Algorithms, 7(2), 157–165.
- Vigoda, E. (1999). *Improved bounds for sampling colorings.* FOCS 1999.
- Geyer, C. J. (1992). *Practical Markov chain Monte Carlo.* Statistical Science, 7(4), 473–483. (Initial positive sequence ESS estimator)
- CS 4850 course notes, Cornell University — Definition 6.11, Lemma 6.12, Lemma 6.13, Equation 6.1.
