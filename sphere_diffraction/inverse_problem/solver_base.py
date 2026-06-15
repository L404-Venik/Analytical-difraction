"""
solver_base.py — Abstract base for synthesis solvers.

Concrete solvers inherit from Solver and implement a single method: run().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .optimization import OptimizationTask, SolverResult
from .search_space import SearchSpace


class Solver(ABC):
    """
    Abstract synthesis solver.

    A solver takes a physical search space and an optimization task,
    and returns a ranked list of candidate structures.

    Subclasses implement ``run()`` and are free to carry their own
    configuration as constructor arguments.

    Example
    -------
    >>> solver = BruteForceSolver(SolverConfig(n_best=5))
    >>> result = solver.run(space, task)
    """

    @abstractmethod
    def run(self, space: SearchSpace, task: OptimizationTask) -> SolverResult:
        """
        Execute the search and return the best candidates found.

        Parameters
        ----------
        space : SearchSpace
            Physical parameter space (geometry + materials).
        task : OptimizationTask
            Wavelength(s), angles, and objective functional to minimize.

        Returns
        -------
        SolverResult
            Ranked candidates and search statistics.
        """
        ...
