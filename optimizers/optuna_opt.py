"""
Optuna optimizer implementation.
"""

import numpy as np
import ioh
from .base_optimizer import BaseOptimizer

import optuna


class OptunaOptimizer(BaseOptimizer):
    """Optuna TPE optimizer."""

    def __init__(self, sampler_name="TPE"):
        super().__init__(f"Optuna-{sampler_name}")
        self.sampler_name = sampler_name

    def optimize(self, problem: ioh.ProblemType, budget: int, seed: int) -> None:
        """
        Optuna optimization using TPE sampler.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        # Suppress optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            # Create suggestion for each dimension
            x = []
            for i in range(problem.meta_data.n_variables):
                x.append(
                    trial.suggest_float(
                        f"x{i}", problem.bounds.lb[i], problem.bounds.ub[i]
                    )
                )
            x = np.array(x)
            return problem(x)

        # Create study and optimize
        sampler = optuna.samplers.TPESampler(seed=seed)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=budget, show_progress_bar=False)
