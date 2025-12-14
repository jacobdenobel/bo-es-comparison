import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark


class DummyProblem:
    """Simple stand-in for an IOH problem to track evaluation counts."""

    def __init__(self, dimension):
        self.meta_data = types.SimpleNamespace(n_variables=dimension)
        bounds = types.SimpleNamespace(
            lb=np.full(dimension, -5.0),
            ub=np.full(dimension, 5.0),
        )
        self.bounds = bounds
        self._current_calls = 0
        self.call_history = []

    def __call__(self, x):
        self._current_calls += 1
        return float(np.sum(x))

    def reset(self):
        self.call_history.append(self._current_calls)
        self._current_calls = 0

    def attach_logger(self, logger):
        # Logger attachment is irrelevant for these unit tests.
        pass


class DummyOptimizer:
    """Tracks optimize calls and budgets."""

    name = "dummy"

    def __init__(self):
        self.calls = []

    def optimize(self, problem, budget):
        self.calls.append({"problem": problem, "budget": budget})


@pytest.fixture
def patch_ioh(monkeypatch):
    """Return helper to patch ioh.get_problem with a provided dummy problem."""

    def _patch(problem):
        monkeypatch.setattr(
            benchmark,
            "ioh",
            types.SimpleNamespace(get_problem=lambda *args, **kwargs: problem),
        )

    return _patch


def test_run_single_experiment_uses_configured_initial_samples(monkeypatch, patch_ioh):
    dimension = 2
    budget = 20
    dummy_problem = DummyProblem(dimension)
    patch_ioh(dummy_problem)
    monkeypatch.setattr(benchmark, "N_REP", 1)
    monkeypatch.setattr(benchmark, "DOE_FACTOR", 3)

    target_initial = benchmark.DOE_FACTOR * dimension
    configured = [[float(i) for i in range(dimension)] for _ in range(target_initial)]
    monkeypatch.setattr(
        benchmark, "get_initial_samples", lambda fid, dim: list(configured)
    )

    uniform_calls = {"count": 0}

    def fake_uniform(*args, **kwargs):
        uniform_calls["count"] += 1
        return np.zeros(dimension)

    monkeypatch.setattr(benchmark.np.random, "uniform", fake_uniform)

    optimizer = DummyOptimizer()
    runner = benchmark.BenchmarkRunner(log_root=".")
    runner.run_single_experiment(
        optimizer, function_id=1, instance_id=1, dimension=dimension, budget=budget
    )

    assert uniform_calls["count"] == 0
    assert dummy_problem.call_history == [target_initial]
    assert optimizer.calls == [{"problem": dummy_problem, "budget": budget - target_initial}]


def test_run_single_experiment_fills_random_samples(monkeypatch, patch_ioh):
    dimension = 3
    budget = 25
    dummy_problem = DummyProblem(dimension)
    patch_ioh(dummy_problem)
    monkeypatch.setattr(benchmark, "N_REP", 1)
    monkeypatch.setattr(benchmark, "DOE_FACTOR", 3)

    configured = [[float(i) for i in range(dimension)] for _ in range(2)]
    monkeypatch.setattr(
        benchmark, "get_initial_samples", lambda fid, dim: list(configured)
    )

    random_draws = []

    def fake_uniform(low, high):
        random_draws.append((low.copy(), high.copy()))
        return np.full(dimension, 0.42)

    monkeypatch.setattr(benchmark.np.random, "uniform", fake_uniform)

    optimizer = DummyOptimizer()
    runner = benchmark.BenchmarkRunner(log_root=".")
    runner.run_single_experiment(
        optimizer, function_id=1, instance_id=1, dimension=dimension, budget=budget
    )

    target_initial = benchmark.DOE_FACTOR * dimension
    additional_needed = target_initial - len(configured)

    assert len(random_draws) == additional_needed
    assert dummy_problem.call_history == [target_initial]
    assert optimizer.calls == [{"problem": dummy_problem, "budget": budget - target_initial}]


def test_run_single_experiment_respects_budget_cap(monkeypatch, patch_ioh):
    dimension = 4
    budget = 5  # smaller than DOE_FACTOR * dimension
    dummy_problem = DummyProblem(dimension)
    patch_ioh(dummy_problem)
    monkeypatch.setattr(benchmark, "N_REP", 1)
    monkeypatch.setattr(benchmark, "DOE_FACTOR", 3)
    monkeypatch.setattr(benchmark, "get_initial_samples", lambda fid, dim: [])

    draw_count = {"count": 0}

    def fake_uniform(low, high):
        draw_count["count"] += 1
        return np.zeros(dimension)

    monkeypatch.setattr(benchmark.np.random, "uniform", fake_uniform)

    optimizer = DummyOptimizer()
    runner = benchmark.BenchmarkRunner(log_root=".")
    runner.run_single_experiment(
        optimizer, function_id=1, instance_id=1, dimension=dimension, budget=budget
    )

    assert draw_count["count"] == budget
    assert dummy_problem.call_history == [budget]
    assert optimizer.calls == []
