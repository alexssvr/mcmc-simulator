"""
core/configurator.py — ProblemConfigurator and ProblemSpec.

Builds a fully-validated ProblemSpec from either a config dict (programmatic
use) or interactive terminal prompts (exploratory / CLI use).

Typical usage
-------------
Fully programmatic (no prompts)::

    from core.configurator import ProblemConfigurator

    cfg = ProblemConfigurator()
    spec = cfg.configure({
        "problem_type": "continuous_data",
        "data": np.array([1.2, 3.4, 5.6]),
        "notes": "sensor readings",
    })

Partially specified (missing keys trigger prompts for those fields only)::

    spec = cfg.configure({"problem_type": "continuous_formula", "formula": my_log_w})

Fully interactive (no config provided)::

    spec = cfg.configure()
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# ProblemSpec
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProblemSpec:
    """Fully-validated description of an MCMC sampling problem.

    Attributes
    ----------
    problem_type : str
        One of ``'discrete_graph'``, ``'continuous_data'``,
        ``'continuous_formula'``, or ``'path'``.
    state_space_type : str
        Inferred from ``problem_type``: ``'discrete'``, ``'continuous'``,
        or ``'path'``.
    dimension : int
        Number of scalar components in a single state.  Inferred from
        ``data.shape[1]`` (or ``data.ndim`` for 1-D arrays) when not
        given explicitly.
    bounds : None or list[float]
        ``None`` for an unbounded state space; otherwise ``[low, high]``
        giving coordinate-wise bounds (same bounds for every dimension).
        Inferred from ``[data.min(), data.max()]`` when not given explicitly
        and data is present.
    has_data : bool
        ``True`` iff a non-``None`` numpy array was provided.
    data : numpy.ndarray or None
        Raw observed data array, or ``None`` if the problem is
        formula-only or discrete.
    formula : callable or None
        A callable ``log_w(x) -> float`` defining the unnormalized log
        target density, or ``None`` if the problem is data-driven or
        discrete.
    notes : str
        Free-form human-readable description of the problem.
    """

    problem_type: str
    state_space_type: str
    dimension: int
    bounds: list[float] | None
    has_data: bool
    data: np.ndarray | None
    formula: Callable[..., float] | None
    notes: str


# ---------------------------------------------------------------------------
# _STATE_SPACE_MAP — single source of truth for the type inference rule
# ---------------------------------------------------------------------------

_STATE_SPACE_MAP: dict[str, str] = {
    "discrete_graph":      "discrete",
    "continuous_data":     "continuous",
    "continuous_formula":  "continuous",
    "path":                "path",
}

_VALID_PROBLEM_TYPES: frozenset[str] = frozenset(_STATE_SPACE_MAP)

_CONTINUOUS_TYPES: frozenset[str] = frozenset(
    k for k, v in _STATE_SPACE_MAP.items() if v == "continuous"
)


# ---------------------------------------------------------------------------
# ProblemConfigurator
# ---------------------------------------------------------------------------

class ProblemConfigurator:
    """Builds a :class:`ProblemSpec` from a config dict, interactive prompts,
    or a combination of both.

    When *config* is supplied to :meth:`configure`, its keys are used
    directly.  Any required field that is absent from *config* is
    collected via ``input()`` prompts so that partial configs are
    accepted gracefully.

    When *config* is ``None``, all required fields are collected
    interactively.

    Parameters
    ----------
    None — the class is stateless; create one instance and call
    :meth:`configure` as many times as needed.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, config: dict[str, Any] | None = None) -> ProblemSpec:
        """Build and return a validated :class:`ProblemSpec`.

        Parameters
        ----------
        config : dict or None
            Optional dictionary of problem parameters.  Recognised keys:

            ``"problem_type"`` : str
                One of ``'discrete_graph'``, ``'continuous_data'``,
                ``'continuous_formula'``, ``'path'``.
            ``"data"`` : numpy.ndarray, optional
                Observed data array.  For 1-D arrays the dimension is 1;
                for 2-D arrays the dimension is ``data.shape[1]``.
            ``"formula"`` : callable, optional
                ``log_w(x) -> float`` — unnormalized log target density.
            ``"dimension"`` : int, optional
                Overrides dimension inferred from *data*.
            ``"bounds"`` : list[float], optional
                ``[low, high]`` coordinate-wise bounds.  Overrides the
                bounds inferred from *data*.
            ``"notes"`` : str, optional
                Free-form description of the problem.

        Returns
        -------
        ProblemSpec
            A fully-populated and validated spec.

        Raises
        ------
        ValueError
            If *problem_type* is not one of the four recognised values;
            or if a continuous problem has both ``"data"`` and ``"formula"``
            explicitly present in *config* but both set to ``None``
            (indicating a deliberate omission rather than interactive use).
        TypeError
            If *formula* is provided but is not callable.

        Notes
        -----
        If a continuous problem is missing both ``"data"`` and ``"formula"``
        and at least one of those keys was absent from *config*, the
        configurator enters interactive mode for that field: the user is
        prompted to type a Python expression for ``log_w(x)`` which is
        compiled via :func:`eval` into a callable.  Standard ``math`` and
        ``numpy`` (as ``np``) are available in the expression namespace.
        """
        cfg: dict[str, Any] = dict(config) if config is not None else {}

        # ---- Required: problem_type ----------------------------------------
        problem_type: str = self._resolve_problem_type(cfg)

        # ---- Derived: state_space_type (no prompt needed) ------------------
        state_space_type: str = _STATE_SPACE_MAP[problem_type]

        # ---- Optional inputs: data and formula -----------------------------
        data: np.ndarray | None = cfg.get("data", None)
        formula: Callable[..., float] | None = cfg.get("formula", None)

        if formula is not None and not callable(formula):
            raise TypeError(
                f"'formula' must be callable, got {type(formula).__name__!r}"
            )

        # Validate / prompt that continuous problems have data or formula.
        #
        # Two cases when both are still None after reading cfg:
        #   a) Both keys were *explicitly present* in config (both set to None)
        #      → the caller deliberately omitted them; raise immediately.
        #   b) At least one key was *absent* from config
        #      → the user is in interactive / partial-config mode; prompt for
        #      a formula expression rather than raising.
        if problem_type in _CONTINUOUS_TYPES and data is None and formula is None:
            both_explicit_in_config = "data" in cfg and "formula" in cfg
            if both_explicit_in_config:
                raise ValueError(
                    f"problem_type={problem_type!r} requires either 'data' "
                    f"(a numpy array) or 'formula' (a callable log_w), "
                    f"but neither was provided."
                )
            # Interactive path: prompt the user for a log_w expression.
            formula = self._prompt_formula()

        # ---- Optional: dimension (infer from data, then prompt) ------------
        dimension: int = self._resolve_dimension(cfg, data)

        # ---- Optional: bounds (infer from data, then prompt) ---------------
        bounds: list[float] | None = self._resolve_bounds(cfg, data)

        # ---- Optional: notes (prompt if absent) ----------------------------
        notes: str = self._resolve_notes(cfg)

        return ProblemSpec(
            problem_type=problem_type,
            state_space_type=state_space_type,
            dimension=dimension,
            bounds=bounds,
            has_data=data is not None,
            data=data,
            formula=formula,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Private resolution helpers (each handles one field)
    # ------------------------------------------------------------------

    def _resolve_problem_type(self, cfg: dict[str, Any]) -> str:
        """Return problem_type from cfg or prompt; validate it."""
        if "problem_type" in cfg:
            value: str = cfg["problem_type"]
        else:
            value = input(
                "Problem type? "
                "(discrete_graph / continuous_data / continuous_formula / path): "
            ).strip()

        if value not in _VALID_PROBLEM_TYPES:
            raise ValueError(
                f"Invalid problem_type {value!r}. "
                f"Must be one of: {sorted(_VALID_PROBLEM_TYPES)}"
            )
        return value

    def _resolve_dimension(
        self, cfg: dict[str, Any], data: np.ndarray | None
    ) -> int:
        """Return dimension: explicit > inferred from data > prompted."""
        if "dimension" in cfg:
            value = cfg["dimension"]
            return int(value)

        # Infer from data shape
        if data is not None:
            if data.ndim == 1:
                return 1
            # 2-D: rows are observations, columns are dimensions
            return int(data.shape[1])

        # Fall back to prompt
        raw: str = input(
            "Dimension of state space? (press Enter for 1): "
        ).strip()
        if not raw:
            return 1
        value = int(raw)
        if value < 1:
            raise ValueError(f"dimension must be >= 1, got {value}")
        return value

    def _resolve_bounds(
        self, cfg: dict[str, Any], data: np.ndarray | None
    ) -> list[float] | None:
        """Return bounds: explicit > inferred from data > prompted."""
        if "bounds" in cfg:
            raw = cfg["bounds"]
            if raw is None:
                return None
            low, high = float(raw[0]), float(raw[1])
            if low >= high:
                raise ValueError(
                    f"bounds[0] must be strictly less than bounds[1], "
                    f"got [{low}, {high}]"
                )
            return [low, high]

        # Infer from data range
        if data is not None:
            return [float(data.min()), float(data.max())]

        # Fall back to prompt
        raw_str: str = input(
            "Bounds? Enter as low,high or press Enter for unbounded: "
        ).strip()
        if not raw_str:
            return None
        parts = raw_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Expected bounds in 'low,high' format, got {raw_str!r}"
            )
        low, high = float(parts[0].strip()), float(parts[1].strip())
        if low >= high:
            raise ValueError(
                f"bounds[0] must be strictly less than bounds[1], "
                f"got [{low}, {high}]"
            )
        return [low, high]

    def _prompt_formula(self) -> Callable[..., float]:
        """Prompt the user for a log_w expression and return it as a callable.

        The user enters a Python expression in *x* (e.g. ``-0.5*x**2``).
        The expression is wrapped in ``lambda x: <expr>`` and compiled via
        :func:`eval`.  Standard math functions (``math``, ``numpy`` as ``np``)
        are available in the evaluation namespace.

        Returns
        -------
        Callable[..., float]
            A one-argument callable ``log_w(x) -> float``.

        Raises
        ------
        ValueError
            If the entered expression cannot be compiled or is empty.
        """
        import math as _math

        raw: str = input(
            "No formula provided. Enter a Python expression for log_w(x) "
            "(e.g. -0.5*x**2 for standard Gaussian): "
        ).strip()

        if not raw:
            raise ValueError(
                "A log_w(x) expression is required for continuous problems "
                "when no data or formula is provided."
            )

        namespace: dict[str, Any] = {"math": _math, "np": np, "__builtins__": {}}
        try:
            formula: Callable[..., float] = eval(f"lambda x: {raw}", namespace)
        except SyntaxError as exc:
            raise ValueError(
                f"Could not parse log_w expression {raw!r}: {exc}"
            ) from exc

        return formula

    def _resolve_notes(self, cfg: dict[str, Any]) -> str:
        """Return notes from cfg or prompt; defaults to empty string."""
        if "notes" in cfg:
            return str(cfg["notes"])
        raw: str = input(
            "Any notes about the state space? (press Enter to skip): "
        ).strip()
        return raw
