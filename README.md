# MCMC Simulator

A modular Metropolis-Hastings MCMC simulator built for Cornell CS 4850. Any probability distribution or sampling problem plugs in through a clean interface — the MH engine is shared across all of them.

The implementation follows **Equation 6.1** from the course notes exactly:

```
P_xy = K_xy · κ_x · min(w_x, w_y) / w_x     for x ≠ y
```

with full log-space arithmetic to prevent numerical underflow.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Quick Start](#quick-start)
4. [Usage Guide](#usage-guide)
   - [Mode 1 — Sample from Data (KDE)](#mode-1--sample-from-data-kde)
   - [Mode 2 — Sample from a Formula](#mode-2--sample-from-a-formula)
   - [Mode 3 — Discrete Graph Coloring](#mode-3--discrete-graph-coloring)
   - [Interactive (No-Config) Mode](#interactive-no-config-mode)
5. [Configuration Reference](#configuration-reference)
6. [Reading Your Results](#reading-your-results)
7. [Plotting](#plotting)
8. [How the Math Works](#how-the-math-works)
9. [Running Tests](#running-tests)

---

## Project Structure

```
mcmc-simulator/
├── main.py                  ← MCMCSampler: your single entry point
├── core/
│   ├── problem.py           ← SamplingProblem ABC (the plugin interface)
│   ├── mh.py                ← Metropolis-Hastings engine (Eq 6.1)
│   ├── classifier.py        ← Auto-selects proposal kernel from problem type
│   ├── configurator.py      ← Validates / completes config dicts; interactive prompts
│   └── diagnostics.py       ← TV distance, mixing time, autocorrelation (coming soon)
├── problems/
│   ├── continuous.py        ← DataDrivenProblem (KDE), FormulaProblem
│   └── discrete.py          ← DiscreteGraphProblem (Glauber dynamics)
├── tests/
│   └── test_mh.py           ← pytest suite (4 tests)
└── data/                    ← Drop CSV / NumPy files here
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

You should see **4 passed** in under 2 seconds.

---

## Quick Start

```python
from main import MCMCSampler
import numpy as np

sampler = MCMCSampler()

# Sample from a standard normal using a formula
result = sampler.run({
    "problem_type": "continuous_formula",
    "formula":      lambda x: -0.5 * x**2,   # log w(x) for N(0,1)
    "notes":        "Standard Normal",
})

sampler.plot(result)
print(f"Acceptance rate: {result.acceptance_rate:.3f}")
```

That's it. The sampler automatically picks the right proposal kernel, sets burn-in, and runs the chain. You confirm once at the terminal, then it runs.

---

## Usage Guide

All interaction goes through `MCMCSampler.run(config)`. Pass a Python dict describing your problem; the sampler fills in defaults for anything missing, prints a summary, asks for confirmation, then returns a `SamplerResult`.

### Mode 1 — Sample from Data (KDE)

Use this when you have observed data and want to draw new samples from the same distribution. The simulator fits a **Gaussian kernel density estimate (KDE)** to your data and uses that as the target.

```python
import numpy as np
from main import MCMCSampler

# Load your data — any 1-D NumPy array works
data = np.loadtxt("data/returns.csv")

result = MCMCSampler().run({
    "problem_type": "continuous_data",
    "data":         data,
    "notes":        "Daily stock returns",   # optional label
})
```

**What happens automatically:**
- Fits a KDE to `data`
- Detects dimension (1-D or multi-D)
- Sets step size to `2.38 × std(data)` — the theoretically optimal scale for ~44% acceptance (Roberts, Gelman & Gilks 1997)
- Runs 1 000 burn-in steps + 5 000 sampling steps

**Multi-dimensional data** works the same way — just pass a 2-D array with shape `(n_observations, d_dimensions)`:

```python
data_2d = np.column_stack([returns, volumes])   # shape (n, 2)
result = MCMCSampler().run({
    "problem_type": "continuous_data",
    "data":         data_2d,
})
```

---

### Mode 2 — Sample from a Formula

Use this when you know the (unnormalized) log-density analytically.

```python
from main import MCMCSampler
import math

result = MCMCSampler().run({
    "problem_type": "continuous_formula",
    "formula":      lambda x: -0.5 * x**2,     # log w(x) — does NOT need to be normalized
    "dimension":    1,
    "notes":        "N(0, 1)",
})
```

**Key rules for the formula:**

| Rule | Example |
|---|---|
| Must accept a single argument `x` | `lambda x: ...` |
| Must return the **log** of the (unnormalized) weight | `-0.5 * x**2` for N(0,1) |
| For 1-D: `x` is a `float` | `lambda x: -x**2 / 2` |
| For multi-D: `x` is a NumPy array | `lambda x: -0.5 * np.dot(x, x)` |
| Can use any Python/NumPy math | `lambda x: -abs(x)` for Laplace |

**Bounded domains** — if your distribution is only defined on an interval, pass `bounds`:

```python
result = MCMCSampler().run({
    "problem_type": "continuous_formula",
    "formula":      lambda x: math.log(x) - x,   # log of Gamma-like density
    "dimension":    1,
    "bounds":       [0.01, 20.0],                 # physical constraint
    "notes":        "Gamma-like on (0, 20)",
})
```

The sampler automatically switches to a **reflected Gaussian** kernel that bounces off the boundaries instead of stepping outside them.

**Multi-dimensional formula:**

```python
import numpy as np
from main import MCMCSampler

# 2-D standard normal
result = MCMCSampler().run({
    "problem_type": "continuous_formula",
    "formula":      lambda x: -0.5 * float(np.dot(x, x)),
    "dimension":    2,
    "notes":        "2-D N(0, I)",
})
```

---

### Mode 3 — Discrete Graph Coloring

Use this to sample uniformly from all **proper q-colorings** of an undirected graph using Glauber dynamics.

A proper coloring assigns one of `q` colors to each vertex so that no two adjacent vertices share the same color.

**Option A — Adjacency list (recommended):**

```python
from main import MCMCSampler

# Represent the graph as a dict: vertex → list of neighbors
# Every edge must appear in BOTH directions
triangle = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1],
}

result = MCMCSampler().run({
    "problem_type": "discrete_graph",
    "adjacency":    triangle,
    "q":            4,          # number of colors (must be >= max_degree + 1 to mix)
    "notes":        "Triangle graph, 4 colors",
})
```

**Option B — Edge list (compact format):**

```python
result = MCMCSampler().run({
    "problem_type": "discrete_graph",
    "data": {
        "n_vertices": 5,
        "n_colors":   4,
        "edges":      [(0,1), (1,2), (2,3), (3,4), (4,0)],   # 5-cycle
    },
    "notes": "5-cycle, 4 colors",
})
```

**Choosing `q`:**

| Rule | Meaning |
|---|---|
| `q >= max_degree + 1` | A valid initial coloring always exists |
| `q >= max_degree + 2` | Glauber dynamics mix in polynomial time (Jerrum 1995) |
| `q >= 2 × max_degree` | Mixing in O(n log n) (Vigoda 1999) |

---

### Interactive (No-Config) Mode

Run the sampler with no arguments and it will prompt you for everything in the terminal:

```python
from main import MCMCSampler

MCMCSampler().run()
```

Or from the command line:

```bash
conda activate mcmc
python -c "from main import MCMCSampler; MCMCSampler().run()"
```

You will be asked:
1. Problem type (data / formula / graph / path)
2. Data file path **or** log-density formula as a Python expression
3. Dimension, bounds, notes

For the formula prompt, type any valid Python expression in `x`, e.g.:

```
Enter log_w(x): -0.5*x**2
```

The sampler wraps it in a lambda automatically.

---

## Configuration Reference

Full list of keys accepted by `MCMCSampler.run(config)`:

| Key | Type | Required for | Description |
|---|---|---|---|
| `problem_type` | `str` | all | `"continuous_data"`, `"continuous_formula"`, `"discrete_graph"` |
| `data` | `np.ndarray` | `continuous_data` | Observed samples, shape `(n,)` or `(n, d)` |
| `formula` | `callable` | `continuous_formula` | `lambda x: log_w(x)` — returns **log** unnormalized weight |
| `dimension` | `int` | multi-D formula | State-space dimension (inferred from `data` if omitted) |
| `bounds` | `[float, float]` | optional | Physical bounds `[low, high]`; triggers reflected kernel |
| `adjacency` | `dict[int, list[int]]` | `discrete_graph` | Adjacency list, bidirectional |
| `q` | `int` | `discrete_graph` | Number of colors |
| `notes` | `str` | optional | Free-form label shown in plots and summaries |

### Proposal kernels chosen automatically

| Problem | Condition | Kernel |
|---|---|---|
| `continuous_data` / `continuous_formula` | 1-D, no bounds | Gaussian random walk |
| `continuous_data` / `continuous_formula` | 1-D, with bounds | Reflected Gaussian |
| `continuous_data` / `continuous_formula` | Multi-D, no bounds | Multivariate Gaussian |
| `discrete_graph` | any | Glauber dynamics |

---

## Reading Your Results

`MCMCSampler.run()` returns a `SamplerResult` dataclass:

```python
result = MCMCSampler().run(config)

# The actual samples (post-burn-in)
result.samples          # list of floats, arrays, or dicts depending on problem type

# Chain health
result.acceptance_rate  # float in [0, 1]; target is ~0.44 for continuous 1-D

# Full diagnostics dict
result.diagnostics["acceptance_rate"]       # same as above
result.diagnostics["effective_sample_size"] # ESS — accounts for autocorrelation
result.diagnostics["n_samples"]             # number of post-burn-in samples
result.diagnostics["autocorrelations"]      # list of ACF values used in ESS

# What was configured
result.problem_spec     # ProblemSpec dataclass — type, dimension, bounds, etc.
result.proposal_config  # dict — kernel type, step size, burn-in, n_samples
```

**Interpreting acceptance rate:**

| Range | Meaning |
|---|---|
| `< 0.1` | Step size too large — most proposals rejected; chain barely moves |
| `0.2 – 0.5` | Good for continuous distributions; near-optimal mixing |
| `0.5 – 0.8` | Acceptable; chain mixes but may be slow |
| `> 0.9` | Step size too small — proposals always accepted; chain moves slowly |

**Interpreting ESS (Effective Sample Size):**

ESS corrects for autocorrelation between consecutive samples. An ESS of 500 from 5 000 samples means you effectively have 500 independent draws. ESS >= n/10 is generally considered healthy.

---

## Plotting

```python
# Display in a window
sampler.plot(result)

# Save to a file instead
sampler.plot(result, save_path="output/my_plot.png")
```

What you get:

- **Continuous 1-D:** histogram with KDE overlay; title shows `n`, acceptance rate, ESS
- **Continuous multi-D:** one subplot per dimension (marginal histograms)
- **Discrete:** bar chart of color frequencies per vertex (up to 6 vertices shown)

---

## How the Math Works

The simulator implements the **Metropolis-Hastings algorithm** (Eq 6.1, CS 4850 notes).

At each step, given current state `x`:

1. **Propose** a candidate `y` from the proposal kernel `K(x, ·)`
2. **Compute** the log-acceptance probability:

   ```
   log alpha = min(log w(y), log w(x)) - log w(x) + log kappa(x) + log K(y->x) - log K(x->y)
   ```

3. **Accept** `y` with probability `min(1, alpha)`; otherwise stay at `x`

The stationary distribution is `pi proportional to w` (Lemma 6.12 in the notes).

Everything is computed in **log-space** to avoid floating-point underflow when `w(x)` is very small.

All three proposal kernels are **symmetric** (`K(x->y) = K(y->x)`), so the Hastings correction cancels and `log alpha` simplifies to `min(log w(y), log w(x)) - log w(x)`.

**Step size calibration:** for Gaussian random-walk proposals, the step size is set to `2.38 * std(data)`. This comes from Roberts, Gelman & Gilks (1997): for a Gaussian target N(0, sigma^2), the proposal std `h* = 2.38 * sigma` maximises the expected squared jump distance and achieves the theoretically optimal **44% acceptance rate** for 1-D MH.

---

## Running Tests

```bash
conda activate mcmc
pytest tests/ -v
```

The test suite covers:

| Test | What it checks |
|---|---|
| `test_mh_standard_normal` | Chain on N(0,1) formula: KS test, mean ~0, std ~1, acc in [0.2, 0.8] |
| `test_glauber_triangle` | Glauber on P3 path graph: all sampled colorings are proper; chain mixes |
| `test_proposal_classifier` | All 5 kernel-selection rules fire correctly |
| `test_problem_configurator` | All 4 problem types configure without interactive prompts |

---

## References

- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). *Weak convergence and optimal scaling of random walk Metropolis algorithms.* Annals of Applied Probability, 7(1), 110-120.
- Jerrum, M. (1995). *A very simple algorithm for estimating the number of k-colorings of a low-degree graph.* Random Structures & Algorithms, 7(2), 157-165.
- Vigoda, E. (1999). *Improved bounds for sampling colorings.* FOCS 1999.
- CS 4850 course notes, Cornell University — Eq 6.1, Lemma 6.12.
