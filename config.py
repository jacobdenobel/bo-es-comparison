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


# Random seed for reproducibility
RANDOM_SEED = 12

# Logging settings
LOG_ROOT = 'data'


