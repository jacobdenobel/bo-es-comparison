"""
CMA-ES optimizer implementations.
"""

import ioh
from base_optimizer import BaseOptimizer

try:
    from modcma import c_maes

    MODCMA_AVAILABLE = True
except ImportError:
    MODCMA_AVAILABLE = False
    print("Warning: modcma not available. CMA-ES optimizers will not work.")


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES optimizer using modcma."""

    def __init__(self):
        super().__init__("CMA-ES")
        if not MODCMA_AVAILABLE:
            raise ImportError("modcma package is required for CMA-ES")

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        CMA-ES optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        settings = c_maes.settings_from_dict(
            problem.meta_data.n_variables,
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
            target=problem.optimum.y + 1e-8,
            active=True,
        )
        es = c_maes.ModularCMAES(settings)
        es.run(problem)


class OnePlusOneCMAESOptimizer(BaseOptimizer):
    """(1+1)-CMA-ES optimizer using modcma."""

    def __init__(self):
        super().__init__("(1+1)-CMA-ES")
        if not MODCMA_AVAILABLE:
            raise ImportError("modcma package is required for (1+1)-CMA-ES")

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        (1+1)-CMA-ES optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        settings = c_maes.settings_from_dict(
            problem.meta_data.n_variables,
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
            target=problem.optimum.y + 1e-8,
            lambda0=1,
            mu0=1,
        )
        es = c_maes.ModularCMAES(settings)
        es.run(problem)
