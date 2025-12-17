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

from config import (
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

        # Reset optimizer state per repetition if supported
        if hasattr(optimizer, "reset") and callable(getattr(optimizer, "reset")):
            optimizer.reset()

        optimizer.optimize(problem, budget)
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
    

    for optimizer in optimizers:
        logger = None
        if log_results:
            logger = ioh.logger.Analyzer(
                algorithm_name=optimizer.name,
                folder_name=optimizer.name,
                root=log_root,
            )
        breakpoint()
        for dim in dimensions:
            budget = budget_factor * dim
            for fid in functions:
                for iid in instances:
                    experiment_count += 1
                    run_single_experiment(
                        optimizer=optimizer,
                        function_id=fid,
                        instance_id=iid,
                        dimension=dim,
                        budget=budget,
                        logger=logger,
                    )


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
    
    run_benchmark(
        optimizers=optimizers,
        functions=FUNCTIONS,
        instances=INSTANCES,
        dimensions=DIMENSIONS,
        budget_factor=BUDGET_FACTOR,
        log_results=True,
        log_root=LOG_ROOT,
    )


if __name__ == "__main__":
    main()
