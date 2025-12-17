"""
Base optimizer interface and utilities.
"""

import abc
import numpy as np
import ioh

import random
from modcma import c_maes


class BaseOptimizer(abc.ABC):
    """Base class for all optimizers."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def optimize(self, problem: ioh.ProblemType, budget: int, seed: int) -> None:
        """
        Optimize the given problem within the budget.

        Args:
            problem: IOH problem instance
            budget: Maximum number of function evaluations
        """
        pass
    
    def reset(self) -> None:
        pass


def set_seeds(seed: int = 12):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    c_maes.utils.set_seed(seed)
