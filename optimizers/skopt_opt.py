"""
Scikit-Optimize (skopt) optimizer implementation.
"""

import numpy as np
import ioh
from .base_optimizer import BaseOptimizer
from .config import DOE_FACTOR

from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei

class SkoptOptimizer(BaseOptimizer):
    """Scikit-Optimize Gaussian Process optimizer."""

    def __init__(self, acquisition="EI"):
        super().__init__(f"Skopt-{acquisition}")
        self.acquisition = acquisition

    def optimize(self, problem: ioh.ProblemType, budget: int, seed:int) -> None:
        """
        Scikit-Optimize optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        # Define search space
        dimensions = []
        for i in range(problem.meta_data.n_variables):
            dimensions.append(
                Real(problem.bounds.lb[i], problem.bounds.ub[i], name=f"x{i}")
            )
       
        result = gp_minimize(
            func=problem,
            dimensions=dimensions,
            n_calls=budget,
            acq_func=self.acquisition,
            random_state=seed,
            n_initial_points=int(DOE_FACTOR * problem.meta_data.n_variables),
            initial_point_generator="halton"
        )
 
