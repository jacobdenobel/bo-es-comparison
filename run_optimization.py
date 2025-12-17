import sys
import argparse
from benchmark import run_benchmark
from optimizers import create_optimizer, get_available_optimizers
from config import FUNCTIONS, INSTANCES, DIMENSIONS, BUDGET_FACTOR, LOG_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single optimizer on BBOB.")
    parser.add_argument("optimizer_name", type=str)
    parser.add_argument("--budget-factor", type=int, default=BUDGET_FACTOR)
    parser.add_argument("--log-root", type=str, default=LOG_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    requested = args.optimizer_name.lower()

    available = get_available_optimizers()
    if requested not in [n.lower() for n in available]:
        print(f"Requested optimizer '{args.optimizer_name}' is not available.")
        print(f"Available: {available}")
        sys.exit(1)

    optimizer = create_optimizer(requested)
    print(f"Running optimizer: {optimizer.name}")

    run_benchmark(
        optimizers=[optimizer],
        functions=FUNCTIONS,
        instances=INSTANCES,
        dimensions=DIMENSIONS,
        budget_factor=args.budget_factor,
        log_results=True,
        log_root=args.log_root,
    )


if __name__ == "__main__":
    main()
