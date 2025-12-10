"""
Meta's Ax optimizer implementation.
"""

import numpy as np
import ioh
from .base_optimizer import BaseOptimizer

try:
    from ax.service.ax_client import AxClient
    from ax.utils.measurement.synthetic_functions import hartmann6

    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False
    print("Warning: ax-platform not available. Ax optimizer will not work.")


class AxOptimizer(BaseOptimizer):
    """Meta's Ax (Adaptive Experimentation) optimizer."""

    def __init__(self, strategy="default"):
        super().__init__(f"Ax-{strategy}")
        if not AX_AVAILABLE:
            raise ImportError("ax-platform package is required for Ax optimizer")
        self.strategy = strategy

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        Ax optimization.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        # Create Ax client
        ax_client = AxClient(random_seed=42)

        # Create experiment
        parameters = []
        for i in range(problem.meta_data.n_variables):
            parameters.append(
                {
                    "name": f"x{i}",
                    "type": "range",
                    "bounds": [
                        float(problem.bounds.lb[i]),
                        float(problem.bounds.ub[i]),
                    ],
                }
            )

        ax_client.create_experiment(
            name="bbob_experiment",
            parameters=parameters,
            objective_name="objective",
            minimize=True,
        )

        # Run optimization loop
        for _ in range(budget):
            try:
                # Get next trial
                parameters, trial_index = ax_client.get_next_trial()

                # Convert to numpy array
                x = np.array(
                    [parameters[f"x{i}"] for i in range(problem.meta_data.n_variables)]
                )

                # Evaluate
                y = problem(x)

                # Complete trial
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data={"objective": (y, 0.0)}
                )

            except Exception as e:
                # Fallback to random sampling if Ax fails
                x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
                problem(x)
