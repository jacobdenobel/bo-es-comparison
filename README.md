# BBOB Optimizer Comparison

This project compares different optimization algorithms (CMA-ES, SMAC3, Random Search, Optuna, Skopt, Ax, BoTorch) on BBOB (Black-Box Optimization Benchmarking) functions across different dimensions (2, 3, 5, 10).

## Project Structure

```
bo-es-comparison/
├── config.py              # Configuration settings
├── base_optimizer.py       # Base optimizer interface
├── optimizers/            # Optimizer implementations
│   ├── __init__.py        # Optimizer factory
│   ├── random_search.py   # Random Search
│   ├── cmaes.py          # CMA-ES variants
│   ├── optuna_opt.py     # Optuna optimizers
│   ├── smac_opt.py       # SMAC3 optimizer
│   ├── skopt_opt.py      # Scikit-Optimize optimizers
│   ├── ax_opt.py         # Meta's Ax optimizer
│   └── botorch_opt.py    # BoTorch optimizer
├── benchmark.py           # Benchmark runner
├── analysis.py           # Results analysis and visualization
├── run_custom_benchmark.py # Custom benchmark with CLI arguments
├── setup.sh              # Automated setup script (bash)
├── setup.py              # Automated setup script (Python)
├── requirements.txt      # Package dependencies
└── README.md            # This file
```

## Installation

### Quick Setup (Automated)

Run the setup script to automatically create the environment and install packages:

```bash
# Using bash script (Linux/macOS)
./setup.sh

# OR using Python script (cross-platform)
python setup.py
```

### Manual Setup

### 1. Create conda environment with Python 3.12
```bash
conda create -n bo-es python=3.12
```

### 2. Activate the environment
```bash
conda activate bo-es
```

### 3. Install modcma first (requires Python 3.12)
```bash
pip install modcma
```

### 4. Install all other requirements
```bash
pip install -r requirements.txt
```

**Note**: Some packages may not install successfully due to dependencies. The system will automatically detect which optimizers are available and only use those.

## Usage

### Running the Benchmark

To run the full benchmark with all available optimizers:

```bash
python benchmark.py
```

This will:
- Automatically detect which optimizers are available (based on installed packages)
- Run each optimizer on BBOB functions 1-24, instances 1-14, dimensions 2,3,5,10
- Save results to the `data/` directory

### Analyzing Results

To analyze and visualize results:

```bash
python analysis.py
```

Or use the analyzer programmatically:

```python
from analysis import ResultAnalyzer

analyzer = ResultAnalyzer()
analyzer.get_summary_statistics()
analyzer.plot_convergence(function_ids=[1], dimensions=[2])
analyzer.plot_performance_comparison(function_ids=[1, 2, 3], dimension=2)
```

### Using Individual Optimizers

```python
from optimizers import create_optimizer
import ioh

# Create optimizer
optimizer = create_optimizer('optuna_tpe')

# Create problem
problem = ioh.get_problem(1, 1, 2)  # Function 1, instance 1, 2D

# Optimize
optimizer.optimize(problem, budget=100)
```

## Troubleshooting

### Package Installation Issues

If some packages fail to install:
1. The system will automatically detect available optimizers
2. You can run with just the successfully installed optimizers
3. For package-specific issues:
   - **modcma**: Requires Python 3.12, install separately first
   - **ax-platform**: May have SQLAlchemy version conflicts (warnings are harmless)
   - **botorch/torch**: Large packages, may take time to install
   - **scikit-optimize**: May have dependency conflicts with newer scipy versions

### Checking Available Optimizers

```python
from optimizers import get_available_optimizers
print("Available optimizers:", get_available_optimizers())
```

## Available Optimizers

- **random**: Random Search
- **cmaes**: CMA-ES (requires `modcma`)
- **one_plus_one_cmaes**: (1+1)-CMA-ES (requires `modcma`)
- **optuna_tpe**: Optuna with TPE sampler (requires `optuna`)
- **optuna_random**: Optuna with Random sampler (requires `optuna`)
- **optuna_cmaes**: Optuna with CMA-ES sampler (requires `optuna`)
- **smac**: SMAC3 (requires `smac3`)
- **skopt_ei**: Scikit-Optimize with Expected Improvement (requires `scikit-optimize`)
- **skopt_pi**: Scikit-Optimize with Probability of Improvement (requires `scikit-optimize`)
- **skopt_lcb**: Scikit-Optimize with Lower Confidence Bound (requires `scikit-optimize`)
- **ax**: Meta's Ax optimizer (requires `ax-platform`)
- **botorch_ei**: BoTorch with Expected Improvement (requires `botorch`, `torch`)

## Configuration

Edit `config.py` to modify:
- Functions to test (`FUNCTIONS`)
- Problem instances (`INSTANCES`) 
- Dimensions (`DIMENSIONS`)
- Number of repetitions (`N_REP`)
- Budget factor (`BUDGET_FACTOR`)

## Results

Results are saved in IOH format in the `data/` directory, organized by optimizer name. Each optimizer gets its own subdirectory with detailed logging files that can be analyzed using IOHInspector.