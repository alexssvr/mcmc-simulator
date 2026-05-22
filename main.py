"""
main.py — MCMCSampler: the single user-facing entry point for the simulator.

Typical usage (programmatic)
-----------------------------
::

    from main import MCMCSampler
    import numpy as np

    sampler = MCMCSampler()

    # Data-driven continuous problem
    result = sampler.run({
        "problem_type": "continuous_data",
        "data": np.random.normal(0, 1, 1000),
        "notes": "Standard Normal via KDE",
    })
    sampler.plot(result)

    # Formula-driven continuous problem
    result = sampler.run({
        "problem_type": "continuous_formula",
        "formula": lambda x: -0.5 * x ** 2,
        "dimension": 1,
        "bounds": None,
        "notes": "N(0,1)",
    })

    # Discrete graph coloring
    result = sampler.run({
        "problem_type": "discrete_graph",
        "adjacency": {0: [1, 2], 1: [0, 2], 2: [0, 1]},
        "q": 3,
        "notes": "triangle q=3",
    })

Full interactive mode (no config — prompts for everything)::

    sampler.run()
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from core.classifier import ProposalClassifier
from core.configurator import ProblemConfigurator, ProblemSpec
from core.mh import MetropolisHastings
from core.problem import SamplingProblem
from problems.continuous import DataDrivenProblem, FormulaProblem
from problems.discrete import DiscreteGraphProblem


# ---------------------------------------------------------------------------
# SamplerResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SamplerResult:
    """Immutable record returned by :meth:`MCMCSampler.run`.

    Attributes
    ----------
    samples : list
        Post-burn-in chain states.  Element type matches the problem's
        state space (``float``, ``numpy.ndarray``, or ``dict``).
    acceptance_rate : float
        Fraction of proposed moves accepted over the full run
        (including burn-in).
    problem_spec : ProblemSpec
        Fully-validated problem description produced by the configurator.
    proposal_config : dict
        Proposal kernel configuration produced by the classifier.
        Contains ``"type"``, ``"step_size"``, ``"recommended_burn_in"``,
        ``"recommended_n_samples"``, and optionally ``"bounds"`` / ``"cov"``.
    diagnostics : dict
        Post-run diagnostics::

            {
                "acceptance_rate":      float,
                "effective_sample_size": float,   # ESS
                "n_samples":            int,
                "autocorrelations":     list[float],  # lags used in ESS
            }
    """

    samples: list
    acceptance_rate: float
    problem_spec: ProblemSpec
    proposal_config: dict
    diagnostics: dict


# ---------------------------------------------------------------------------
# MCMCSampler
# ---------------------------------------------------------------------------

class MCMCSampler:
    """Single entry point for configuring, running, and plotting MCMC chains.

    All state is contained in the returned :class:`SamplerResult`; this
    class itself is stateless and can be reused across multiple runs.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: dict[str, Any] | None = None) -> SamplerResult:
        """Configure and run the MH sampler end-to-end.

        Parameters
        ----------
        config : dict or None
            Optional problem specification.  Missing keys trigger
            interactive ``input()`` prompts.  Recognised keys:

            ``"problem_type"``
                ``'discrete_graph'`` | ``'continuous_data'`` |
                ``'continuous_formula'`` | ``'path'``
            ``"data"``        : numpy.ndarray   (continuous_data)
            ``"formula"``     : callable        (continuous_formula)
            ``"dimension"``   : int             (optional; inferred from data)
            ``"bounds"``      : [low, high]     (optional; inferred from data)
            ``"notes"``       : str             (optional)
            ``"adjacency"``   : dict[int, list] (discrete_graph only)
            ``"q"``           : int             (discrete_graph only)

        Returns
        -------
        SamplerResult
            Samples, diagnostics, and metadata.

        Raises
        ------
        SystemExit
            If the user answers ``'n'`` at the confirmation prompt.
        ValueError
            If a required key is missing for the chosen problem type.
        NotImplementedError
            If ``problem_type == 'path'`` (not yet implemented).
        """
        cfg: dict[str, Any] = dict(config) if config is not None else {}

        # ── Normalise discrete_graph edge-list data-dict format ──────────────
        # Accepts:  data={'edges': [(u,v), …], 'n_vertices': N, 'n_colors': q}
        # Converts to: adjacency=dict, q=int, plus sensible defaults so that
        # the configurator never needs to prompt for dimension/bounds/notes.
        if (cfg.get("problem_type") == "discrete_graph"
                and isinstance(cfg.get("data"), dict)):
            _gd  = cfg.pop("data")
            _n   = int(_gd.get("n_vertices", 0))
            _q   = int(_gd.get("n_colors", 3))
            _adj: dict[int, list[int]] = {i: [] for i in range(_n)}
            for _u, _v in _gd.get("edges", []):
                _adj[int(_u)].append(int(_v))
                _adj[int(_v)].append(int(_u))
            cfg.setdefault("adjacency", _adj)
            cfg.setdefault("q",         _q)
            cfg.setdefault("dimension", _n)
            cfg.setdefault("bounds",    [0, _q - 1])
            cfg.setdefault("notes",     f"graph {_n}V q={_q}")

        # ── Step 1: Configure problem ────────────────────────────────────────
        spec: ProblemSpec = ProblemConfigurator().configure(cfg)

        # ── Step 2: Classify proposal ────────────────────────────────────────
        # IMPORTANT: pass only *explicitly-provided* bounds to the classifier.
        # spec.bounds may have been inferred from data.min()/data.max() by the
        # configurator — that is a statistical property of the dataset, NOT a
        # physical constraint on the state space.  Passing inferred bounds to
        # the classifier would incorrectly trigger reflected_gaussian for any
        # data-driven problem regardless of whether the user intended it.
        # cfg.get("bounds", None) returns None unless the caller set "bounds"
        # explicitly, which is the correct signal for a truly bounded domain.
        explicit_bounds: list[float] | None = cfg.get("bounds", None)
        state_desc: dict[str, Any] = {
            "type":      spec.state_space_type,
            "dimension": spec.dimension,
            "bounds":    explicit_bounds,
            "notes":     spec.notes,
        }
        # Pass the data standard deviation so the classifier can tune the
        # gaussian_random_walk step size to the actual data scale.
        # Only meaningful for DataDrivenProblem (spec.has_data == True).
        data_std: float | None = (
            float(spec.data.std()) if spec.has_data and spec.data is not None
            else None
        )
        proposal_config: dict[str, Any] = ProposalClassifier().classify(
            state_desc, data_std=data_std
        )

        burn_in: int    = proposal_config["recommended_burn_in"]
        n_samples: int  = proposal_config["recommended_n_samples"]
        ptype: str      = proposal_config["type"]
        step_size       = proposal_config.get("step_size")
        step_str: str   = f"{step_size:.4g}" if step_size is not None else "N/A"

        # ── Step 3: Print confirmation summary ───────────────────────────────
        print()
        print("=== MCMC Configuration ===")
        print(f"Problem type : {spec.problem_type}")
        print(f"State space  : {spec.state_space_type}, dimension {spec.dimension}")
        print(f"Proposal K   : {ptype} (step size: {step_str})")
        print(f"Burn-in      : {burn_in}")
        print(f"Samples      : {n_samples}")

        # ── Step 4: User confirmation ────────────────────────────────────────
        answer: str = input("Proceed? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

        # ── Step 5: Instantiate SamplingProblem ──────────────────────────────
        problem: SamplingProblem = self._build_problem(spec, proposal_config, cfg)

        # ── Step 6: Run MetropolisHastings ───────────────────────────────────
        mh = MetropolisHastings(problem)
        x0 = problem.initial_state()
        print(f"\nRunning chain: {burn_in} burn-in + {n_samples} samples …")
        samples: list = mh.sample(
            x0=x0,
            n_steps=burn_in + n_samples,
            burn_in=burn_in,
        )
        print(f"Done.  Acceptance rate: {mh.acceptance_rate:.3f}")

        # ── Step 7: Compute diagnostics ──────────────────────────────────────
        diagnostics: dict[str, Any] = self._compute_diagnostics(
            samples, mh.acceptance_rate
        )

        # ── Step 8: Return result ────────────────────────────────────────────
        return SamplerResult(
            samples=samples,
            acceptance_rate=mh.acceptance_rate,
            problem_spec=spec,
            proposal_config=proposal_config,
            diagnostics=diagnostics,
        )

    def plot(
        self,
        result: SamplerResult,
        save_path: str | None = None,
    ) -> None:
        """Plot a histogram (with KDE overlay) or color-frequency bar chart.

        Continuous 1-D
            Single panel: histogram (density=True) with scipy KDE overlay.
        Continuous multi-D
            One subplot per dimension showing marginal histogram + KDE.
        Discrete (dict states)
            One subplot per vertex (up to 6) showing color-frequency bar chart.

        Parameters
        ----------
        result : SamplerResult
            Return value of :meth:`run`.
        save_path : str or None
            If given, save the figure to this path (PNG/PDF/SVG) instead of
            calling ``plt.show()``.  Useful for non-interactive / testing use.
        """
        stype: str = result.problem_spec.state_space_type
        dim: int   = result.problem_spec.dimension
        samples    = result.samples
        notes: str = result.problem_spec.notes or result.problem_spec.problem_type

        acc   = result.diagnostics["acceptance_rate"]
        ess   = result.diagnostics["effective_sample_size"]
        n     = result.diagnostics["n_samples"]
        title_suffix = f"n={n}  acc={acc:.3f}  ESS={ess:.0f}"

        if stype == "discrete":
            self._plot_discrete(samples, dim, notes, title_suffix)
        elif dim == 1:
            self._plot_continuous_1d(samples, notes, title_suffix)
        else:
            self._plot_continuous_nd(samples, dim, notes, title_suffix)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_problem(
        spec: ProblemSpec,
        proposal_config: dict[str, Any],
        config: dict[str, Any],
    ) -> SamplingProblem:
        """Instantiate the correct SamplingProblem subclass from a ProblemSpec.

        Parameters
        ----------
        spec : ProblemSpec
            Validated problem specification.
        proposal_config : dict
            Proposal kernel config from :class:`~core.classifier.ProposalClassifier`.
        config : dict
            The original raw config dict (may carry extra keys such as
            ``"adjacency"`` and ``"q"`` for discrete graph problems).

        Raises
        ------
        ValueError
            If required keys are missing for the chosen problem type.
        NotImplementedError
            For ``'path'`` problems (not yet implemented).
        """
        pt: str = spec.problem_type

        if pt == "continuous_data":
            if spec.data is None:
                raise ValueError(
                    "ProblemSpec.data is None for a 'continuous_data' problem. "
                    "Provide a numpy array under the 'data' key."
                )
            return DataDrivenProblem(
                data=spec.data,
                proposal_config=proposal_config,
                notes=spec.notes,
            )

        if pt == "continuous_formula":
            if spec.formula is None:
                raise ValueError(
                    "ProblemSpec.formula is None for a 'continuous_formula' problem. "
                    "Provide a callable log_w under the 'formula' key."
                )
            return FormulaProblem(
                log_w=spec.formula,
                proposal_config=proposal_config,
                dimension=spec.dimension,
                bounds=spec.bounds,
                notes=spec.notes,
            )

        if pt == "discrete_graph":
            adjacency = config.get("adjacency")
            q = config.get("q")
            if adjacency is None:
                raise ValueError(
                    "Key 'adjacency' (dict[int, list[int]]) is required in "
                    "config for 'discrete_graph' problems."
                )
            if q is None:
                raise ValueError(
                    "Key 'q' (int, number of colors) is required in config "
                    "for 'discrete_graph' problems."
                )
            return DiscreteGraphProblem(
                adjacency=adjacency,
                q=int(q),
                notes=spec.notes,
            )

        if pt == "path":
            raise NotImplementedError(
                "'path' problems are not yet implemented. "
                "Subclass SamplingProblem and register it here."
            )

        raise ValueError(f"Unknown problem type: {pt!r}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _compute_diagnostics(
        self, samples: list, acceptance_rate: float
    ) -> dict[str, Any]:
        """Compute acceptance rate, ESS, and the autocorrelations used.

        Parameters
        ----------
        samples : list
            Post-burn-in chain states.
        acceptance_rate : float
            From :attr:`~core.mh.MetropolisHastings.acceptance_rate`.

        Returns
        -------
        dict
            Keys: ``acceptance_rate``, ``effective_sample_size``,
            ``n_samples``, ``autocorrelations``.
        """
        n: int = len(samples)
        ess, acf = self._compute_ess(samples)
        return {
            "acceptance_rate":       acceptance_rate,
            "effective_sample_size": ess,
            "n_samples":             n,
            "autocorrelations":      acf,
        }

    # ------------------------------------------------------------------
    # ESS helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(samples: list) -> np.ndarray:
        """Convert a heterogeneous list of states to a float numpy array.

        * ``float`` states → shape ``(n,)``
        * ``numpy.ndarray`` states → shape ``(n, d)``
        * ``dict`` states → sorted values → shape ``(n, |V|)``
        """
        if not samples:
            return np.array([], dtype=float)
        first = samples[0]
        if isinstance(first, dict):
            keys = sorted(first.keys())
            return np.array(
                [[s[k] for k in keys] for s in samples], dtype=float
            )
        return np.asarray(samples, dtype=float)

    @staticmethod
    def _acf_truncated(x: np.ndarray, max_lag: int) -> tuple[list[float], float]:
        """Compute normalized ACF at lags 1..max_lag, stopping at first ≤ 0 lag.

        Uses the *initial positive sequence* estimator (Geyer 1992): sum only
        contiguous positive autocorrelations starting from lag 1.  This avoids
        over-estimating the sum when the ACF oscillates or is noisy.

        Returns
        -------
        acf : list[float]
            Autocorrelations at lags 1, 2, … up to the cutoff.
        rho_sum : float
            Sum of all returned autocorrelations (used directly in ESS).
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

    @classmethod
    def _compute_ess(
        cls, samples: list, max_lag: int = 100
    ) -> tuple[float, list[float]]:
        """Compute effective sample size using truncated positive-sequence ACF.

        For multi-dimensional states, ESS is computed per dimension; the
        minimum (most conservative) is returned together with the ACF that
        produced it.

        Formula::

            ESS = n / (1 + 2 · Σ ρ_k)

        Parameters
        ----------
        samples : list
            Post-burn-in states.
        max_lag : int
            Maximum lag to consider before forcing truncation.

        Returns
        -------
        ess : float
            Effective sample size, clamped to ``[1, n]``.
        acf : list[float]
            Autocorrelations used for the returned ESS.
        """
        arr = cls._to_array(samples)
        n = len(arr)
        if n == 0:
            return 0.0, []

        if arr.ndim == 1:
            acf, rho_sum = cls._acf_truncated(arr, max_lag)
            ess = n / (1.0 + 2.0 * rho_sum)
            return max(1.0, min(float(n), ess)), acf

        # Multi-D: minimum ESS over all dimensions
        best_ess = float("inf")
        best_acf: list[float] = []
        for d in range(arr.shape[1]):
            acf_d, rho_sum_d = cls._acf_truncated(arr[:, d], max_lag)
            ess_d = n / (1.0 + 2.0 * rho_sum_d)
            if ess_d < best_ess:
                best_ess = ess_d
                best_acf = acf_d

        return max(1.0, min(float(n), best_ess)), best_acf

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_continuous_1d(
        samples: list, notes: str, title_suffix: str
    ) -> None:
        """Histogram + KDE overlay for a 1-D continuous chain."""
        arr = np.asarray(samples, dtype=float)
        fig, ax = plt.subplots(figsize=(8, 4))

        # Histogram (density so KDE is on the same scale)
        ax.hist(arr, bins=60, density=True, alpha=0.45, color="steelblue",
                label="Samples")

        # KDE overlay
        kde = gaussian_kde(arr)
        x_grid = np.linspace(arr.min() - 0.5, arr.max() + 0.5, 400)
        ax.plot(x_grid, kde(x_grid), color="crimson", linewidth=2, label="KDE")

        ax.set_xlabel("State")
        ax.set_ylabel("Density")
        ax.set_title(f"{notes}\n{title_suffix}")
        ax.legend()

    @staticmethod
    def _plot_continuous_nd(
        samples: list, dim: int, notes: str, title_suffix: str
    ) -> None:
        """Marginal histogram + KDE for each dimension of a multi-D chain."""
        arr = np.asarray(samples, dtype=float)   # (n, d)
        n_plot = min(dim, 6)                      # cap at 6 subplots
        fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4), squeeze=False)

        for d in range(n_plot):
            ax = axes[0, d]
            col = arr[:, d]
            ax.hist(col, bins=50, density=True, alpha=0.45, color="steelblue",
                    label=f"dim {d}")
            kde = gaussian_kde(col)
            x_grid = np.linspace(col.min() - 0.5, col.max() + 0.5, 300)
            ax.plot(x_grid, kde(x_grid), color="crimson", linewidth=2)
            ax.set_xlabel(f"Dimension {d}")
            ax.set_ylabel("Density" if d == 0 else "")
            ax.set_title(f"Marginal dim {d}")
            ax.legend(fontsize=8)

        fig.suptitle(f"{notes} — {title_suffix}", fontsize=10)

    @staticmethod
    def _plot_discrete(
        samples: list, dim: int, notes: str, title_suffix: str
    ) -> None:
        """Color-frequency bar chart for each vertex (up to 6) of a discrete chain."""
        # Convert dict states → array of shape (n, n_vertices)
        keys = sorted(samples[0].keys())
        arr = np.array([[s[k] for k in keys] for s in samples], dtype=int)

        n_plot = min(len(keys), 6)
        fig, axes = plt.subplots(1, n_plot, figsize=(3 * n_plot, 4), squeeze=False)

        for i in range(n_plot):
            ax = axes[0, i]
            vertex = keys[i]
            colors, counts = np.unique(arr[:, i], return_counts=True)
            freq = counts / counts.sum()
            ax.bar(colors, freq, color="steelblue", alpha=0.8, width=0.6)
            ax.set_xlabel("Color")
            ax.set_ylabel("Frequency" if i == 0 else "")
            ax.set_title(f"Vertex {vertex}")
            ax.set_xticks(colors)

        fig.suptitle(f"{notes} — {title_suffix}", fontsize=10)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sampler = MCMCSampler()
    result = sampler.run()
    sampler.plot(result)
