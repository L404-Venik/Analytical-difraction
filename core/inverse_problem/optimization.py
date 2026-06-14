# optimization.py — Shared data types for the synthesis framework.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from ..parameters import BodyParameters, ObservationParameters


# ---------------------------------------------------------------------------
# OptimizationTask
# ---------------------------------------------------------------------------

class OptimizationTask:
    """
    Describes what scattering property to optimize.

    Parameters
    ----------
    wavelength : float or np.ndarray
        Single wavelength for single-frequency optimization, or
        an array of wavelengths for broadband optimization.
    angles : np.ndarray
        Angles (radians) at which S is evaluated, in [0, 2π).
        Typically ``np.linspace(0, 2*np.pi, M, endpoint=False)``.
        The array length M is passed to ``calculate_S`` as the sample count.
    functional : Callable[[np.ndarray, np.ndarray, np.ndarray], float]
        Objective function ``f(S_th, S_ph, angles) → float``.
        Solvers minimize this value.
        For broadband mode, it is called once per wavelength and then
        aggregated according to ``SolverConfig.aggregation``.
        To maximize, return ``-f(...)`` from the callable.
        Multi-objective problems can be handled by returning a weighted sum.

    Examples
    --------
    Single wavelength, minimize backscattering (theta=0):

    >>> task = OptimizationTask(
    ...     wavelength=0.03,
    ...     angles=np.linspace(0, 2*np.pi, 360, endpoint=False),
    ...     functional=lambda S_th, S_ph, angles: np.abs(S_th[0])**2,
    ... )

    Broadband over 20 frequencies:

    >>> task = OptimizationTask(
    ...     wavelength=np.linspace(0.01, 0.05, 20),
    ...     angles=np.linspace(0, 2*np.pi, 360, endpoint=False),
    ...     functional=lambda S_th, S_ph, angles: np.abs(S_th[0])**2,
    ... )
    """

    def __init__(
        self,
        wavelength: Union[float, np.ndarray],
        angles: np.ndarray,
        functional: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ):
        wl = np.asarray(wavelength, dtype=np.float64)
        if wl.ndim == 0:
            self._wavelength: Union[float, np.ndarray] = float(wl)
        elif wl.ndim == 1:
            if len(wl) == 0:
                raise ValueError("wavelength array must not be empty.")
            self._wavelength = float(wl[0]) if len(wl) == 1 else wl
        else:
            raise ValueError("wavelength must be a scalar or a 1-D array.")

        if np.any(np.asarray(self._wavelength) <= 0):
            raise ValueError("All wavelengths must be > 0.")

        angles = np.asarray(angles, dtype=np.float64)
        if angles.ndim != 1 or len(angles) == 0:
            raise ValueError("angles must be a non-empty 1-D array.")

        self.angles = angles
        self.functional = functional

    @property
    def is_broadband(self) -> bool:
        """True when multiple wavelengths are optimized simultaneously."""
        return isinstance(self._wavelength, np.ndarray)

    @property
    def wavelengths(self) -> np.ndarray:
        """All wavelengths as a 1-D array (length 1 for single-frequency)."""
        if self.is_broadband:
            return self._wavelength
        return np.array([self._wavelength])

    @property
    def wavelength(self) -> float:
        """The single wavelength. Raises AttributeError if broadband."""
        if self.is_broadband:
            raise AttributeError(
                "This task is broadband. Use .wavelengths to get the array."
            )
        return self._wavelength  # type: ignore[return-value]

    @property
    def M(self) -> int:
        """Number of angular samples evaluated."""
        return len(self.angles)

    def to_observation(self) -> ObservationParameters:
        """
        Build the ObservationParameters that drives calculate_S for this task.

        Carries all wavelengths (one per row of the returned S arrays) and the
        angle grid. The body is supplied separately by the solver.
        """
        return ObservationParameters(wavelengths=self.wavelengths, angles=self.angles)

    def __repr__(self) -> str:
        if self.is_broadband:
            wl_str = (
                f"broadband [{self._wavelength[0]:.4g} … {self._wavelength[-1]:.4g}] m "
                f"({len(self._wavelength)} points)"
            )
        else:
            wl_str = f"λ={self._wavelength:.4g} m"
        return f"OptimizationTask({wl_str}, M={self.M})"


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """
    Controls how a solver executes the search and what it returns.

    Parameters
    ----------
    n_best : int
        Number of top candidates to return, sorted by F ascending.
        Retaining near-optimal solutions allows engineering trade-offs
        (cost, simplicity, mass) to be applied after the search. Default 1.
    aggregation : str or Callable
        Broadband aggregation rule — how per-wavelength F values are
        combined into a single scalar for ranking.
        Built-in strings: ``'mean'`` (default), ``'max'``, ``'sum'``.
        Custom: any callable ``f(values: np.ndarray) → float``.
        Ignored in single-frequency mode.
    progress : bool
        Show a progress bar (tqdm if available, else plain percentage).
        Useful for large search spaces where the sweep can take minutes.
    """

    n_best: int = 1
    aggregation: Union[str, Callable[[np.ndarray], float]] = "mean"
    progress: bool = True

    def __post_init__(self):
        if self.n_best < 1:
            raise ValueError(f"n_best must be >= 1, got {self.n_best}")
        valid_strings = {"mean", "max", "sum"}
        if isinstance(self.aggregation, str) and self.aggregation not in valid_strings:
            raise ValueError(
                f"aggregation string must be one of {valid_strings}, "
                f"got '{self.aggregation}'"
            )

    def _aggregate(self, values: np.ndarray) -> float:
        """Apply the aggregation rule to a 1-D array of per-wavelength F values."""
        if callable(self.aggregation):
            return float(self.aggregation(values))
        if self.aggregation == "mean":
            return float(np.mean(values))
        if self.aggregation == "max":
            return float(np.max(values))
        if self.aggregation == "sum":
            return float(np.sum(values))
        raise RuntimeError(f"Unknown aggregation: {self.aggregation}")  # unreachable


# ---------------------------------------------------------------------------
# SolverResult
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """
    Output of a synthesis search.

    Attributes
    ----------
    best : list of (F, BodyParameters)
        Top ``n_best`` candidate bodies, sorted by F ascending (lowest = best).
        Bodies carry no wavelength; to recover S for a result, call
        ``calculate_S(body, task.to_observation())``.
    n_evaluated : int
        Number of candidates actually evaluated.
    n_skipped : int
        Number of candidates whose objective was non-finite (NaN/inf) and
        excluded from ranking.
    elapsed_seconds : float
        Wall-clock time for the entire search.
    """

    best: list[tuple[float, BodyParameters]]
    n_evaluated: int
    n_skipped: int
    elapsed_seconds: float

    def __repr__(self) -> str:
        if self.best:
            f_str = f"best F={self.best[0][0]:.6g}"
        else:
            f_str = "no results"
        return (
            f"SolverResult({f_str}, "
            f"n_evaluated={self.n_evaluated}, "
            f"n_skipped={self.n_skipped}, "
            f"elapsed={self.elapsed_seconds:.2f}s)"
        )
