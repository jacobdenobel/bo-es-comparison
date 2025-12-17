"""
benchmark.py

Benchmark runner for comparing optimization algorithms on BBOB (IOH) functions.

Run:
    python benchmark.py
"""

from __future__ import annotations

import multiprocessing
from itertools import product

import ioh
from typing import List, Optional

from optimizers.config import (
    LOG_ROOT,
    FUNCTIONS,
    INSTANCES,
    DIMENSIONS,
    BUDGET_FACTOR,
    N_REP,
    RANDOM_SEED,
)


from optimizers import create_optimizer, get_available_optimizers
from optimizers.base_optimizer import BaseOptimizer, set_seeds


def run_single_experiment(
    optimizer: BaseOptimizer,
    function_id: int,
    instance_id: int,
    dimension: int,
    budget: int,
    logger: Optional[ioh.logger.Analyzer] = None,
) -> None:
    """
    Run N_REP repetitions for one (optimizer, function, instance, dimension).

    Notes:
      - We set a different (but deterministic) seed per repetition to avoid identical reps.
      - If your optimizer keeps state between runs, implement optimizer.reset() and it will
        be called automatically per repetition.
    """
    problem = ioh.get_problem(function_id, instance_id, dimension)

    if logger is not None:
        problem.attach_logger(logger)

    for rep in range(N_REP):
        seed = RANDOM_SEED + (function_id * instance_id * dimension * (1 + rep) * 7)
        set_seeds(seed)
        optimizer.reset()
        optimizer.optimize(problem, budget, seed)

        print(
            optimizer.__class__.__name__,
            problem.meta_data,
            problem.state.evaluations,
            problem.state.current_best,
        )
        problem.reset()


def run_benchmark(
    optimizers: List[BaseOptimizer],
    functions: List[int] = FUNCTIONS,
    instances: List[int] = INSTANCES,
    dimensions: tuple = DIMENSIONS,
    budget_factor: int = BUDGET_FACTOR,
    log_results: bool = True,
    log_root: str = LOG_ROOT,
) -> None:
    """
    Run complete benchmark comparing multiple optimizers.
    """
    print(f"Starting benchmark with {len(optimizers)} optimizers...")
    print(
        f"Testing {len(functions)} functions, {len(instances)} instances, "
        f"{len(dimensions)} dimensions, {N_REP} repetitions"
    )
    print(f"Logs root: {log_root}")

    for optimizer in optimizers:
        print(f"\nRunning optimizer: {optimizer.name}")

        total_experiments = len(dimensions) * len(functions) * len(instances)
        experiment_count = 0

        for dim in dimensions:
            logger = None
            if log_results:
                # Structure: LOG_ROOT/D{dim}/{optimizer.name}/...
                dim_folder = f"{log_root}/D{dim}"
                logger = ioh.logger.Analyzer(
                    algorithm_name=optimizer.name,
                    folder_name=optimizer.name,
                    root=dim_folder,
                )

            budget = budget_factor * dim

            for fid in functions:
                for iid in instances:
                    experiment_count += 1
                    if (
                        experiment_count % 50 == 0
                        or experiment_count == total_experiments
                    ):
                        progress = (experiment_count / total_experiments) * 100
                        print(
                            f"  Progress: {progress:.1f}% "
                            f"({experiment_count}/{total_experiments})"
                        )

                    try:
                        run_single_experiment(
                            optimizer=optimizer,
                            function_id=fid,
                            instance_id=iid,
                            dimension=dim,
                            budget=budget,
                            logger=logger,
                        )
                    except Exception as e:
                        print(f"  Error in experiment f{fid}_i{iid}_d{dim}: {e}")

            # Close logger cleanly per dimension (Analyzer writes files as it goes,
            # but closing helps flush resources)
            if logger is not None:
                try:
                    logger.close()
                except Exception:
                    pass


def main() -> None:
    """
    Main entry point: creates all available optimizers and runs the full benchmark.

    Run:
        python benchmark.py
    """

    available_opts = get_available_optimizers()
    print("Available optimizers:", available_opts)

    optimizers: List[BaseOptimizer] = []
    for opt_name in available_opts:
        try:
            opt = create_optimizer(opt_name)
            optimizers.append(opt)
            print(f"✓ Created {opt.name}")
        except Exception as e:
            print(f"✗ Failed to create {opt_name}: {e}")

    if not optimizers:
        print("No optimizers available! Please install required packages.")
        return

    print("This only runs one function now w/o logging")

    run_benchmark(
        optimizers=optimizers,
        functions=FUNCTIONS[:1],
        instances=INSTANCES[:1],
        dimensions=DIMENSIONS[:1],
        budget_factor=BUDGET_FACTOR,
        log_results=False,
        log_root=LOG_ROOT,
    )


if __name__ == "__main__":
    main()
