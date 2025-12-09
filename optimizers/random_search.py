"""
Random Search optimizer implementation.
"""

import numpy as np
import ioh
from base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    """Random Search optimizer."""

    def __init__(self):
        super().__init__("RandomSearch")

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        Random search optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        for _ in range(budget):
            x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
            y = problem(x)
