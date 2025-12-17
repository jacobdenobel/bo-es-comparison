"""
Optimizer factory for creating optimizer instances.
"""

from .random_search import RandomSearchOptimizer
from .cmaes import CMAESOptimizer, OnePlusOneCMAESOptimizer
from .optuna_opt import OptunaOptimizer
from .smac_opt import SMACOptimizer

# Import new optimizers conditionally to avoid import errors
try:
    from .skopt_opt import SkoptOptimizer

    SKOPT_IMPORT_OK = True
except ImportError:
    SKOPT_IMPORT_OK = False

try:
    from .ax_opt import AxOptimizer

    AX_IMPORT_OK = True
except ImportError:
    AX_IMPORT_OK = False

try:
    from .botorch_opt import BoTorchOptimizer

    BOTORCH_IMPORT_OK = True
except ImportError:
    BOTORCH_IMPORT_OK = False


def create_optimizer(optimizer_name: str):
    """
    Factory function to create optimizer instances.

    Args:
        optimizer_name: Name of the optimizer to create

    Returns:
        Optimizer instance
    """
    optimizers = {
        "random": RandomSearchOptimizer,
        "cmaes": CMAESOptimizer,
        "one_plus_one_cmaes": OnePlusOneCMAESOptimizer,
        "optuna_tpe": lambda: OptunaOptimizer("TPE"),
        "optuna_random": lambda: OptunaOptimizer("Random"),
        "optuna_cmaes": lambda: OptunaOptimizer("CmaEs"),
        "smac3": SMACOptimizer,
    }

    # Add new optimizers if imports succeeded
    if SKOPT_IMPORT_OK:
        optimizers.update(
            {
                "skopt_ei": lambda: SkoptOptimizer("EI"),
                "skopt_pi": lambda: SkoptOptimizer("PI"),
                "skopt_lcb": lambda: SkoptOptimizer("LCB"),
            }
        )

    if AX_IMPORT_OK:
        optimizers["ax"] = AxOptimizer

    if BOTORCH_IMPORT_OK:
        optimizers["botorch_ei"] = lambda: BoTorchOptimizer("EI")

    if optimizer_name.lower() not in optimizers:
        available = list(optimizers.keys())
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {available}")

    return optimizers[optimizer_name.lower()]()


def get_available_optimizers():
    """Get list of available optimizers."""
    available = ["random"]

    # Check for modcma
    try:
        import modcma

        available.extend(["cmaes", "one_plus_one_cmaes"])
    except ImportError:
        pass
    
    # Check for scikit-optimize
    try:
        import skopt

        available.extend(["skopt_ei", "skopt_pi", "skopt_lcb"])
    except ImportError:
        pass
    print("remove this return")
    return available

    # Check for optuna
    try:
        import optuna

        available.extend(["optuna_tpe"])
    except ImportError:
        pass

    # Check for smac
    try:
        import smac

        available.extend(["smac3"])
    except ImportError:
        pass


    # Check for ax-platform
    try:
        import ax

        available.append("ax")
    except ImportError:
        pass

    # Check for botorch
    try:
        import botorch

        available.append("botorch_ei")
    except ImportError:
        pass

    return available
