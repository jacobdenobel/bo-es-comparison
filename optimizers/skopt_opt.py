"""
Scikit-Optimize (skopt) optimizer implementation.
"""

import numpy as np
import ioh
from base_optimizer import BaseOptimizer

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.acquisition import gaussian_ei

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Skopt optimizer will not work.")


class SkoptOptimizer(BaseOptimizer):
    """Scikit-Optimize Gaussian Process optimizer."""

    def __init__(self, acquisition="EI"):
        super().__init__(f"Skopt-{acquisition}")
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize package is required for Skopt optimizer")
        self.acquisition = acquisition

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
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

        # Define objective function for skopt (note: skopt minimizes)
        def objective(x):
            x_array = np.array(x)
            return problem(x_array)

        # Set acquisition function
        acq_func = "EI"  # Expected Improvement
        if self.acquisition == "PI":
            acq_func = "PI"  # Probability of Improvement
        elif self.acquisition == "LCB":
            acq_func = "LCB"  # Lower Confidence Bound

        # Run optimization
        try:
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=budget,
                acquisition=acq_func,
                random_state=42,
                n_initial_points=min(10, budget // 4),
            )
        except Exception as e:
            # Fallback to simpler random sampling if GP fails
            for _ in range(budget):
                x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
                problem(x)
