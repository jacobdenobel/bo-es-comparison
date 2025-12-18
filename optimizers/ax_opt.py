import numpy as np
import ioh
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_node import GenerationStep  # older Ax versions use GenerationStep


from .base_optimizer import BaseOptimizer


class AxBoTorchOptimizer(BaseOptimizer):
    """Ax optimizer using BoTorch-based Bayesian Optimization."""

    def __init__(self, strategy="sobol+botorch", n_init=16):
        super().__init__(f"Ax-{strategy}")
        self.strategy = strategy
        self.n_init = n_init

    def _make_generation_strategy(self, budget: int) -> GenerationStrategy:
        # NOTE: API differs slightly across Ax versions.
        # This pattern is the most common: Sobol -> BoTorch.
        return GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=min(self.n_init, budget),
                    min_trials_observed=min(self.n_init, budget),
                    max_parallelism=1,  # set >1 if you can evaluate in parallel
                    model_kwargs={"seed": None},  # AxClient already seeded
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,  # BoTorch-powered Bayesian optimization
                    num_trials=max(0, budget - self.n_init),
                    max_parallelism=1,
                ),
            ]
        )

    def optimize(self, problem: ioh.ProblemType, budget: int, seed: int) -> None:
        gs = self._make_generation_strategy(budget)

        ax_client = AxClient(random_seed=seed, generation_strategy=gs)

        parameters = []
        for i in range(problem.meta_data.n_variables):
            parameters.append(
                {
                    "name": f"x{i}",
                    "type": "range",
                    "bounds": [float(problem.bounds.lb[i]), float(problem.bounds.ub[i])],
                    "value_type": "float",
                }
            )

        ax_client.create_experiment(
            name="bbob_experiment",
            parameters=parameters,
            objectives={"objective": ObjectiveProperties(minimize=True)},
        )

        for _ in range(budget):
            params, trial_index = ax_client.get_next_trial()
            x = np.array([params[f"x{i}"] for i in range(problem.meta_data.n_variables)], dtype=float)
            y = float(problem(x))

            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"objective": y},
            )
