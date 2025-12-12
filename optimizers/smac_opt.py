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

        # Track function evaluations
        evaluation_count = 0
        
        def objective(config: Configuration, seed: int = 0) -> float:
            """Objective function for SMAC."""
            nonlocal evaluation_count
            
            x = np.array(
                [config[f"x{i}"] for i in range(problem.meta_data.n_variables)]
            )
            
            # Call IOH problem to ensure logging
            result = problem(x)
            evaluation_count += 1
            
            return result

        # Create scenario with proper budget - ensure SMAC uses all trials
        scenario = Scenario(
            configspace=cs, 
            deterministic=True, 
            n_trials=budget, 
            seed=42
        )

        try:
            # Create SMAC facade and optimize
            smac = HyperparameterOptimizationFacade(scenario, objective, overwrite=True)
            incumbent = smac.optimize()
            
            # If SMAC finished early, fill remaining budget with random search
            while evaluation_count < budget and problem.state.evaluations < budget:
                x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
                problem(x)
                evaluation_count += 1
                
        except Exception as e:
            # If SMAC fails, fall back to random search to use remaining budget
            print(f"SMAC optimization failed: {e}. Falling back to random sampling.")
            while evaluation_count < budget and problem.state.evaluations < budget:
                x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
                problem(x)
                evaluation_count += 1
