"""
Benchmark runner for comparing optimization algorithms on BBOB functions.
"""

import ioh
import numpy as np
from typing import List, Optional
from optimizers.base_optimizer import BaseOptimizer, set_seeds
from config import *


class BenchmarkRunner:
    """Runs benchmark experiments comparing multiple optimizers."""

    def __init__(self, log_root: str = LOG_ROOT):
        self.log_root = log_root
        self.results = {}

    def run_single_experiment(
        self,
        optimizer: BaseOptimizer,
        function_id: int,
        instance_id: int,
        dimension: int,
        budget: int,
        logger: Optional[ioh.logger.Analyzer] = None,
    ) -> None:
        """
        Run a single optimization experiment.

        Args:
            optimizer: Optimizer instance
            function_id: BBOB function ID (1-24)
            instance_id: Problem instance ID
            dimension: Problem dimension
            budget: Function evaluation budget
            logger: Optional IOH logger
        """
        problem = ioh.get_problem(function_id, instance_id, dimension)

        if logger is not None:
            problem.attach_logger(logger)

        configured_samples = get_initial_samples(function_id, dimension)
        initial_samples = [
            np.asarray(sample, dtype=float) for sample in configured_samples
        ]
        for idx, sample in enumerate(initial_samples):
            if sample.shape[0] != problem.meta_data.n_variables:
                raise ValueError(
                    f"Initial sample #{idx} for f{function_id} in dimension {dimension} "
                    f"has {sample.shape[0]} variables, "
                    f"expected {problem.meta_data.n_variables}."
                )

        target_initial = max(len(initial_samples), DOE_FACTOR * dimension)
        target_initial = min(budget, target_initial)

        while len(initial_samples) < target_initial:
            init_sample = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
            initial_samples.append(init_sample)

        for rep in range(N_REP):
            used_budget = 0

            if initial_samples:
                for sample in initial_samples:
                    if used_budget >= budget:
                        break
                    problem(sample)
                    used_budget += 1

            remaining_budget = max(0, budget - used_budget)
            if remaining_budget > 0:
                optimizer.optimize(problem, remaining_budget)

            problem.reset()

    def run_benchmark(
        self,
        optimizers: List[BaseOptimizer],
        functions: List[int] = FUNCTIONS,
        instances: List[int] = INSTANCES,
        dimensions: tuple = DIMENSIONS,
        budget_factor: int = BUDGET_FACTOR,
        log_results: bool = True,
    ) -> None:
        """
        Run complete benchmark comparing multiple optimizers.

        Args:
            optimizers: List of optimizer instances
            functions: List of BBOB function IDs to test
            instances: List of problem instance IDs
            dimensions: Tuple of dimensions to test
            budget_factor: Budget = budget_factor * dimension
            log_results: Whether to log results to files
        """
        print(f"Starting benchmark with {len(optimizers)} optimizers...")
        print(
            f"Testing {len(functions)} functions, {len(instances)} instances, "
            f"{len(dimensions)} dimensions"
        )

        for optimizer in optimizers:
            print(f"\nRunning optimizer: {optimizer.name}")

            # Set seeds for reproducibility
            set_seeds(RANDOM_SEED)

            total_experiments = len(dimensions) * len(functions) * len(instances)
            experiment_count = 0

            for dim in dimensions:
                print(f"  Running dimension {dim}...")

                # Set up logger for this dimension and optimizer
                logger = None
                if log_results:
                    # Structure: data/dimension/optimizer/
                    dim_folder = f"{self.log_root}/D{dim}"
                    logger = ioh.logger.Analyzer(
                        algorithm_name=optimizer.name,
                        folder_name=optimizer.name,
                        root=dim_folder,
                    )

                budget = budget_factor * dim
                for fid in functions:
                    for iid in instances:
                        experiment_count += 1
                        if experiment_count % 50 == 0:
                            progress = (experiment_count / total_experiments) * 100
                            print(
                                f"  Progress: {progress:.1f}% "
                                f"({experiment_count}/{total_experiments})"
                            )

                        try:
                            self.run_single_experiment(
                                optimizer, fid, iid, dim, budget, logger
                            )
                        except Exception as e:
                            print(f"  Error in experiment f{fid}_i{iid}_d{dim}: {e}")
                            continue

            print(f"  Completed {optimizer.name}")

        print(f"\nBenchmark completed! Results saved to: {self.log_root}")


def main():
    """Main function to run the benchmark."""
    from optimizers import create_optimizer, get_available_optimizers

    # Get available optimizers
    available_opts = get_available_optimizers()
    print("Available optimizers:", available_opts)

    if OPTIMIZER_NAMES:
        normalized_available = {name.lower(): name for name in available_opts}
        requested_opt_names = []
        for requested in OPTIMIZER_NAMES:
            key = requested.lower()
            if key not in normalized_available:
                print(f"✗ Requested optimizer '{requested}' is not available and will be skipped.")
                continue
            requested_opt_names.append(normalized_available[key])

        if not requested_opt_names:
            print("No requested optimizers are available. Exiting.")
            return
    else:
        requested_opt_names = available_opts

    # Create optimizer instances
    optimizers = []
    for opt_name in requested_opt_names:
        try:
            optimizer = create_optimizer(opt_name)
            optimizers.append(optimizer)
            print(f"✓ Created {optimizer.name}")
        except Exception as e:
            print(f"✗ Failed to create {opt_name}: {e}")

    if not optimizers:
        print("No optimizers available! Please install required packages.")
        return

    # Run benchmark
    runner = BenchmarkRunner()
    runner.run_benchmark(optimizers)


if __name__ == "__main__":
    main()
