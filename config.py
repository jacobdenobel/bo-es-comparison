"""
Configuration settings for the BBOB benchmark experiments.
"""

# Problem settings
FUNCTIONS = list(range(1, 25))  # BBOB functions 1-24
INSTANCES = list(range(1, 15))  # Problem instances 1-14
DIMENSIONS = (2, 3, 5, 10)     # Problem dimensions to test
N_REP = 5                      # Number of repetitions per experiment
BUDGET_FACTOR = 30             # Budget = BUDGET_FACTOR * dimension
DOE_FACTOR = 3                 # Design of Experiments factor (evaluations = DOE_FACTOR * dimension when no custom samples)
OPTIMIZER_NAMES = []           # Optional list of optimizer identifiers to run (e.g., ["smac"])


# Leave empty to fall back to DOE-based random initialization.
INITIAL_SAMPLES = {}

# Random seed for reproducibility
RANDOM_SEED = 12

# Logging settings
LOG_ROOT = 'data'


def get_initial_samples(function_id: int, dimension: int):
    """
    Fetch configured initial samples for a given function and dimension.

    Returns:
        List of coordinate lists (may be empty if not configured).
    """
    dim_entry = INITIAL_SAMPLES.get(dimension)
    if not dim_entry:
        return []

    if isinstance(dim_entry, dict):
        return dim_entry.get(function_id, dim_entry.get("default", [])) or []

    return dim_entry or []
