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