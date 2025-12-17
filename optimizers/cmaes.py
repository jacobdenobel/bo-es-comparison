"""
CMA-ES optimizer implementations.
"""

import ioh
import numpy as np

from .base_optimizer import BaseOptimizer
from modcma import c_maes


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES optimizer using modcma."""

    def __init__(self):
        super().__init__("CMA-ES")

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        CMA-ES optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        settings = c_maes.settings_from_dict(
            problem.meta_data.n_variables,
            x0=np.random.uniform(problem.bounds.lb, problem.bounds.ub),
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
            target=problem.optimum.y + 1e-8,
            sigma0=0.3 * (problem.bounds.ub[0] - problem.bounds.lb[0]),
            active=True,
        )
        es = c_maes.ModularCMAES(settings)
        es.run(problem)


class OnePlusOneCMAESOptimizer(BaseOptimizer):
    """(1+1)-CMA-ES optimizer using modcma."""

    def __init__(self):
        super().__init__("(1+1)-CMA-ES")

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        (1+1)-CMA-ES optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        settings = c_maes.settings_from_dict(
            problem.meta_data.n_variables,
            x0=np.random.uniform(problem.bounds.lb, problem.bounds.ub),
            budget=budget,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
            sigma0=0.3 * (problem.bounds.ub[0] - problem.bounds.lb[0]),
            target=problem.optimum.y + 1e-8,
            lambda0=1,
            mu0=1,
        )
        es = c_maes.ModularCMAES(settings)
        es.run(problem)
