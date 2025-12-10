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

        for rep in range(N_REP):
            optimizer.optimize(problem, budget)
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

    # Create optimizer instances
    optimizers = []
    for opt_name in available_opts:
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
