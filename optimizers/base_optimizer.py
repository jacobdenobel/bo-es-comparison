"""
Base optimizer interface and utilities.
"""

import abc
import numpy as np
import ioh
from typing import Any


class BaseOptimizer(abc.ABC):
    """Base class for all optimizers."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def optimize(self, problem: ioh.ProblemType, budget: int) -> None:
        """
        Optimize the given problem within the budget.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        pass


def set_seeds(seed: int = 12):
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)

    # Set modcma seed if available
    try:
        from modcma import c_maes

        c_maes.utils.set_seed(seed)
    except ImportError:
        pass


def make_experiment_seed(
    base_seed: int,
    optimizer_name: str,
    function_id: int,
    instance_id: int,
    dimension: int,
    rep: int,
) -> int:
    """
    Create a deterministic seed per (optimizer, function, instance, dimension, repetition).
    Uses a stable hash (not Python's built-in hash, which changes between runs).
    """
    import hashlib

    key = f"{base_seed}|{optimizer_name}|f{function_id}|i{instance_id}|d{dimension}|r{rep}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    # Convert 8 bytes to an int in [0, 2^32-1] to keep RNGs happy
    return int.from_bytes(digest, "little") % (2**32)
