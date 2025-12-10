"""
SMAC3 optimizer implementation.
"""

import numpy as np
import ioh
from .base_optimizer import BaseOptimizer

try:
    from smac import HyperparameterOptimizationFacade, Scenario
    from ConfigSpace import Configuration, ConfigurationSpace, Float

    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False
    print("Warning: smac3 not available. SMAC optimizer will not work.")


class SMACOptimizer(BaseOptimizer):
    """SMAC3 optimizer."""

    def __init__(self):
        super().__init__("SMAC3")
        if not SMAC_AVAILABLE:
            raise ImportError("smac3 package is required for SMAC optimizer")

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        SMAC3 optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        # Create configuration space
        cs = ConfigurationSpace()
        for i in range(problem.meta_data.n_variables):
            cs.add_hyperparameter(
                Float(f"x{i}", (problem.bounds.lb[i], problem.bounds.ub[i]))
            )

        def objective(config: Configuration, seed: int = 0) -> float:
            """Objective function for SMAC."""
            x = np.array(
                [config[f"x{i}"] for i in range(problem.meta_data.n_variables)]
            )
            return problem(x)

        # Create scenario
        scenario = Scenario(cs, deterministic=True, n_trials=budget, seed=42)

        # Create SMAC facade and optimize
        smac = HyperparameterOptimizationFacade(scenario, objective)
        incumbent = smac.optimize()
