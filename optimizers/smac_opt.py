import numpy as np
import ioh
from .base_optimizer import BaseOptimizer

try:
    from smac import Scenario

    # Depending on SMAC version, one of these will exist:
    try:
        from smac import BlackBoxOptimizationFacade as BlackBoxFacade
    except Exception:
        from smac import BlackBoxFacade  # older/newer naming in some releases

    from ConfigSpace import Configuration, ConfigurationSpace, Float

    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False
    print("Warning: smac3 not available. SMAC optimizer will not work.")


class SMACOptimizer(BaseOptimizer):
    """SMAC3 optimizer (BlackBoxFacade)."""

    def __init__(self):
        super().__init__("SMAC3")
        if not SMAC_AVAILABLE:
            raise ImportError("smac3 package is required for SMAC optimizer")

    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        # Create configuration space
        cs = ConfigurationSpace()
        for i in range(problem.meta_data.n_variables):
            cs.add_hyperparameter(
                Float(
                    f"x{i}", (float(problem.bounds.lb[i]), float(problem.bounds.ub[i]))
                )
            )

        evaluation_count = 0

        # SMAC blackbox objective: config -> cost (float)
        def objective(config: Configuration, seed: int = 0) -> float:
            nonlocal evaluation_count
            x = np.array(
                [config[f"x{i}"] for i in range(problem.meta_data.n_variables)],
                dtype=float,
            )

            # IOH call (keeps logging / state)
            y = float(problem(x))
            evaluation_count += 1

            # SMAC minimizes by default. If your IOH problem is a maximization,
            # flip the sign here.
            return y

        scenario = Scenario(
            configspace=cs,
            deterministic=True,
            n_trials=int(budget),
            seed=42,
        )

        try:
            smac = BlackBoxFacade(
                scenario=scenario,
                target_function=objective,
                overwrite=True,
            )
            smac.optimize()

            # If SMAC stops early, consume remaining budget with random search
            while evaluation_count < budget and problem.state.evaluations < budget:
                x = np.random.uniform(problem.bounds.lb, problem.bounds.ub).astype(
                    float
                )
                problem(x)
                evaluation_count += 1

        except Exception as e:
            print(
                f"SMAC (BlackBoxFacade) failed: {e}. Falling back to random sampling."
            )
            while evaluation_count < budget and problem.state.evaluations < budget:
                x = np.random.uniform(problem.bounds.lb, problem.bounds.ub).astype(
                    float
                )
                problem(x)
                evaluation_count += 1
