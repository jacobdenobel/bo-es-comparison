# analysis/rank_plot.py
"""
Rank-over-time across functions.

x-axis: evaluations (log scale)
y-axis: aggregated rank across functions (lower is better)

Robust design for BBOB:
- Do NOT compare raw objective across different functions.
- Instead:
  1) per function: aggregate performance across runs for each optimizer at each eval
  2) per function: convert performance to ranks (rank 1 = best)
  3) across functions: aggregate ranks (mean/median)

Outputs:
  plots/rank_over_time_D{dim}.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append(".")
from analysis.performance_over_time import ResultAnalyzer, IOH_INSPECTOR_AVAILABLE


@dataclass(frozen=True)
class RankPlotConfig:
    optimizer_col: str = "algorithm_name"
    function_col: str = "function_id"
    instance_col: str = "instance"
    run_col: str = "data_id"
    eval_col: str = "evaluations"   # auto-infer if missing
    value_col: str = "best_y"

    n_eval_points: int = 60
    aggregate_runs: str = "median"        # "median" or "mean"
    aggregate_functions: str = "mean"     # "mean" or "median"

    exclude_optimizers: Set[str] = frozenset({"geometric_mean", "variable", "None"})


def _to_pandas(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    for attr in ("to_pandas", "to_dataframe"):
        fn = getattr(df, attr, None)
        if callable(fn):
            out = fn()
            if isinstance(out, pd.DataFrame):
                return out
    raise TypeError("Loaded data is not a pandas DataFrame and cannot be converted.")


def _infer_eval_col(d: pd.DataFrame, preferred: str) -> str:
    if preferred in d.columns:
        return preferred
    for c in ("evaluation", "eval", "evals", "evaluations", "n_evaluations", "budget", "fevals"):
        if c in d.columns:
            return c
    raise KeyError(f"Cannot find evaluations column. Columns: {list(d.columns)}")


def _safe_logspace_ints(e_min: int, e_max: int, n: int) -> np.ndarray:
    e_min = max(1, int(e_min))
    e_max = max(e_min, int(e_max))
    if e_min == e_max:
        return np.array([e_min], dtype=np.int64)
    grid = np.logspace(np.log10(e_min), np.log10(e_max), n)
    return np.unique(np.round(grid).astype(np.int64))


def _sample_incumbent_per_run(
    d_keyed: pd.DataFrame,
    by_cols: List[str],
    eval_col: str,
    value_col: str,
    eval_points: np.ndarray,
) -> pd.DataFrame:
    """
    merge_asof per run group (robust; avoids 'left keys must be sorted').
    """
    base = pd.DataFrame({eval_col: np.asarray(eval_points, dtype=np.int64)}).sort_values(eval_col)
    pieces: List[pd.DataFrame] = []

    for keys, g in d_keyed.groupby(by_cols, sort=False):
        g = g.sort_values(eval_col)
        m = pd.merge_asof(
            base,
            g[[eval_col, value_col]],
            on=eval_col,
            direction="backward",
            allow_exact_matches=True,
        )
        m = m.dropna(subset=[value_col])
        if m.empty:
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(by_cols, keys):
            m[col] = val

        pieces.append(m)

    if not pieces:
        return pd.DataFrame(columns=[*by_cols, eval_col, value_col])

    out = pd.concat(pieces, ignore_index=True)
    out[eval_col] = out[eval_col].astype(np.int64)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])
    return out


def compute_rank_over_time_across_functions(
    df_in,
    cfg: RankPlotConfig,
    function_ids: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      - evaluations
      - optimizer
      - rank  (aggregated across functions)
    """
    d = _to_pandas(df_in).copy()
    eval_col = _infer_eval_col(d, cfg.eval_col)

    # Filter functions and optimizers
    if function_ids is not None:
        d = d[d[cfg.function_col].isin(list(function_ids))]
    d = d[~d[cfg.optimizer_col].isin(set(cfg.exclude_optimizers))]

    # Run identity columns
    run_cols = [cfg.run_col, cfg.function_col]
    if cfg.instance_col in d.columns:
        run_cols.append(cfg.instance_col)

    # Drop NA essentials
    need = [cfg.optimizer_col, cfg.function_col, eval_col, cfg.value_col, *run_cols]
    d = d.dropna(subset=need)
    if d.empty:
        return pd.DataFrame(columns=["evaluations", "optimizer", "rank"])

    # Dtypes
    d[eval_col] = pd.to_numeric(d[eval_col], errors="coerce")
    d[cfg.value_col] = pd.to_numeric(d[cfg.value_col], errors="coerce")
    d = d.dropna(subset=[eval_col, cfg.value_col])
    if d.empty:
        return pd.DataFrame(columns=["evaluations", "optimizer", "rank"])

    d[eval_col] = d[eval_col].astype(np.int64)

    # Make key cols consistent (string safest)
    key_cols = [cfg.optimizer_col, *run_cols]
    for c in key_cols:
        d[c] = d[c].astype(str)
    d[cfg.function_col] = d[cfg.function_col].astype(str)

    # Eval grid for this dimension
    eval_points = _safe_logspace_ints(int(d[eval_col].min()), int(d[eval_col].max()), cfg.n_eval_points)

    # Best-so-far incumbent per run
    d = d.sort_values([cfg.optimizer_col, *run_cols, eval_col])
    d[cfg.value_col] = d.groupby([cfg.optimizer_col, *run_cols], sort=False)[cfg.value_col].cummin()

    # Sample incumbent values for each run on the grid
    d_keyed = d[[cfg.optimizer_col, *run_cols, eval_col, cfg.value_col]].copy()
    sampled = _sample_incumbent_per_run(
        d_keyed=d_keyed,
        by_cols=[cfg.optimizer_col, *run_cols],
        eval_col=eval_col,
        value_col=cfg.value_col,
        eval_points=eval_points,
    )
    if sampled.empty:
        return pd.DataFrame(columns=["evaluations", "optimizer", "rank"])

    # Aggregate across runs to get per-function performance per optimizer per eval
    agg_run = np.median if cfg.aggregate_runs.lower() == "median" else np.mean
    per_fn_perf = (
        sampled
        .groupby([cfg.function_col, cfg.optimizer_col, eval_col], as_index=False)
        .agg(perf=(cfg.value_col, lambda s: float(agg_run(s.to_numpy(dtype=float)))))
    )

    # Rank within each function at each evaluation
    per_fn_perf["rank_fn"] = per_fn_perf.groupby([cfg.function_col, eval_col])["perf"].rank(
        method="average",
        ascending=True,
    )

    # Aggregate ranks across functions
    agg_fn = np.mean if cfg.aggregate_functions.lower() == "mean" else np.median
    out = (
        per_fn_perf
        .groupby([cfg.optimizer_col, eval_col], as_index=False)
        .agg(rank=("rank_fn", lambda s: float(agg_fn(s.to_numpy(dtype=float)))))
        .rename(columns={cfg.optimizer_col: "optimizer", eval_col: "evaluations"})
        .sort_values(["optimizer", "evaluations"])
    )

    out["evaluations"] = out["evaluations"].astype(np.int64)
    return out


def plot_rank_over_time(rank_df: pd.DataFrame, title: str, outpath: str, show: bool = True) -> None:
    if rank_df.empty:
        print("No rank data to plot.")
        return

    plt.figure(figsize=(10, 6))
    for opt, g in rank_df.groupby("optimizer", sort=False):
        plt.plot(g["evaluations"], g["rank"], label=str(opt))

    plt.xscale("log")
    plt.gca().invert_yaxis()  # rank 1 on top
    plt.xlabel("Evaluations (log scale)")
    plt.ylabel("Mean rank across functions (lower is better)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    n_opts = rank_df["optimizer"].nunique()
    plt.legend(ncol=2 if n_opts <= 12 else 3, fontsize=9, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")

    if show:
        plt.show()
    else:
        plt.close()


class RankPlotAnalyzer:
    def __init__(self, data_path: str = "data"):
        self.analyzer = ResultAnalyzer(data_path)

    def run(
        self,
        dimensions: Optional[List[int]] = None,
        function_ids: Optional[List[int]] = None,
        cfg: Optional[RankPlotConfig] = None,
        show: bool = True,
    ) -> None:
        if not IOH_INSPECTOR_AVAILABLE:
            print("iohinspector not available for loading data")
            return

        if cfg is None:
            cfg = RankPlotConfig()

        if dimensions is None:
            dimensions = self.analyzer.discover_dimensions()
            if not dimensions:
                print("No dimensions found.")
                return

        if function_ids is None:
            function_ids = list(range(1, 25))

        os.makedirs("plots", exist_ok=True)

        for dim in dimensions:
            print(f"Computing rank-over-time across functions for D={dim} ...")
            df = self.analyzer.load_data(function_ids=list(function_ids), dimensions=[dim])
            if df is None or len(df) == 0:
                print(f"  No data for D={dim}")
                continue

            rank_df = compute_rank_over_time_across_functions(df, cfg=cfg, function_ids=function_ids)
            outpath = f"plots/rank_over_time_D{dim}.png"
            plot_rank_over_time(
                rank_df,
                title=f"Rank over time across functions (D={dim})",
                outpath=outpath,
                show=show,
            )


def main():
    print("=== Rank Over Time Across Functions ===")
    rpa = RankPlotAnalyzer(data_path="data")
    dims = rpa.analyzer.discover_dimensions()
    if not dims:
        print("No dimensions available.")
        return

    cfg = RankPlotConfig(
        n_eval_points=60,
        aggregate_runs="median",
        aggregate_functions="mean",
    )

    rpa.run(dimensions=dims, function_ids=list(range(1, 25)), cfg=cfg, show=True)
    print("Done.")


if __name__ == "__main__":
    main()
