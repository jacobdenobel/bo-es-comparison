"""
Analysis and visualization tools for benchmark results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional

try:
    import iohinspector

    IOH_INSPECTOR_AVAILABLE = True
except ImportError:
    IOH_INSPECTOR_AVAILABLE = False
    print("Warning: iohinspector not available. Analysis tools will be limited.")


class ResultAnalyzer:
    """Analyzes and visualizes benchmark results."""

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.manager = None

        if IOH_INSPECTOR_AVAILABLE:
            self.manager = iohinspector.manager.DataManager()
            if self.data_path.exists():
                self.manager.add_folder(str(self.data_path))

    def discover_dimensions(self) -> List[int]:
        """
        Automatically discover available dimensions from the data folder structure.

        Returns:
            List of dimensions found in data/D{X}/ folders
        """
        dimensions = []
        if not self.data_path.exists():
            print(f"Data path {self.data_path} does not exist")
            return dimensions

        # Find all D{number} folders
        for dim_folder in self.data_path.glob("D*"):
            if dim_folder.is_dir():
                # Extract dimension number from folder name D2, D3, etc.
                dim_name = dim_folder.name
                if dim_name.startswith("D") and dim_name[1:].isdigit():
                    dimension = int(dim_name[1:])
                    dimensions.append(dimension)

        dimensions.sort()  # Sort for consistent order
        print(f"Discovered dimensions from data folder: {dimensions}")
        return dimensions

    def load_data(
        self,
        function_ids: Optional[List[int]] = None,
        dimensions: Optional[List[int]] = None,
        instances: Optional[List[int]] = None,
    ):
        """
        Load benchmark data from the new structure: data/dimension/optimizer/

        Args:
            function_ids: List of function IDs to load
            dimensions: List of dimensions to load
            instances: List of instance IDs to load

        Returns:
            Loaded data or None if iohinspector not available
        """
        if not IOH_INSPECTOR_AVAILABLE:
            print("iohinspector not available for data loading")
            return None

        # Create a fresh DataManager for each load to handle the new structure
        manager = iohinspector.manager.DataManager()

        # Add folders for each dimension
        if dimensions:
            for dim in dimensions:
                dim_folder = self.data_path / f"D{dim}"
                if dim_folder.exists():
                    manager.add_folder(str(dim_folder))
        else:
            # If no specific dimensions, add all D* folders
            for dim_folder in self.data_path.glob("D*"):
                if dim_folder.is_dir():
                    manager.add_folder(str(dim_folder))

        try:
            selection = manager.select(
                function_ids=function_ids, dimensions=dimensions, instances=instances
            )
            return selection.load(monotonic=True, include_meta_data=True)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def plot_all_functions(self, dimension: int = 2, figsize: tuple = (20, 20)) -> None:
        """
        Plot convergence curves for all 24 BBOB functions in a 6x4 grid.

        Args:
            dimension: Dimension to analyze
            figsize: Figure size
        """
        if not IOH_INSPECTOR_AVAILABLE:
            print("iohinspector required for plotting")
            return

        # Define unified font sizes for consistency
        FONT_SIZES = {
            'function_id': 11,  # Function ID in upper right
            'axis_label': 11,   # X and Y axis labels
            'tick_label': 10,   # Tick labels
            'text': 10,         # Error and "No data" text
            'suptitle': 18,     # Overall figure title
            'legend': 11        # Legend text
        }

        # Create 6x4 subplot grid for 24 functions with more square subplots
        fig, axes = plt.subplots(6, 4, figsize=figsize, squeeze=False)

        # Adjust spacing to make subplots more square
        fig.subplots_adjust(right=0.85, hspace=0.4, wspace=0.4)

        function_ids = list(range(1, 25))  # Functions 1-24

        for idx, fid in enumerate(function_ids):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            # Load data for this function
            df = self.load_data(function_ids=[fid], dimensions=[dimension])

            if df is not None and len(df) > 0:
                try:
                    iohinspector.plot.single_function_fixedbudget(df, ax=ax)
                    ax.grid(True, alpha=0.3)

                    # Add function ID in upper right corner
                    ax.text(0.95, 0.95, f"F{fid}", transform=ax.transAxes, 
                           fontsize=FONT_SIZES['function_id'], ha='right', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                    # Remove all individual legends
                    ax.legend().set_visible(False)

                    # Adjust label sizes consistently
                    ax.tick_params(labelsize=FONT_SIZES['tick_label'])
                    ax.set_xlabel("evaluations", fontsize=FONT_SIZES['axis_label'])
                    ax.set_ylabel("best value", fontsize=FONT_SIZES['axis_label'])

                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error: {str(e)[:50]}...",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=FONT_SIZES['text'],
                    )
                    # Add function ID in upper right corner for error case
                    ax.text(0.95, 0.95, f"F{fid}", transform=ax.transAxes, 
                           fontsize=FONT_SIZES['function_id'], ha='right', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=FONT_SIZES['text'],
                )
                # Add function ID in upper right corner for no data case
                ax.text(0.95, 0.95, f"F{fid}", transform=ax.transAxes, 
                       fontsize=FONT_SIZES['function_id'], ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add overall title
        fig.suptitle(
            f"BBOB Functions 1-24 Comparison (Dimension {dimension})",
            fontsize=FONT_SIZES['suptitle'],
            y=0.98,
        )

        # Create a single legend at the bottom using first subplot's data
        if len(function_ids) > 0:
            # Get legend info from the first subplot that has data
            legend_ax = None
            for idx in range(24):
                row = idx // 4
                col = idx % 4
                if axes[row, col].get_legend_handles_labels()[0]:  # Has legend data
                    legend_ax = axes[row, col]
                    break

            if legend_ax is not None:
                handles, labels = legend_ax.get_legend_handles_labels()
                # Create single legend at bottom center
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=min(len(labels), 6),
                    fontsize=FONT_SIZES['legend'],
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                )

        # Use tight_layout with padding for title and legend
        try:
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        except:
            pass  # If tight_layout fails, continue without it

        # Save the plot
        import os

        os.makedirs("plots", exist_ok=True)
        filename = f"plots/BBOB_{dimension}d.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved as: {filename}")

        plt.show()

    def save_all_dimension_plots(self, dimensions: Optional[List[int]] = None) -> None:
        """
        Generate and save plots for all specified dimensions.
        If no dimensions specified, auto-discover from data folder.

        Args:
            dimensions: List of dimensions to plot. If None, auto-discover from data folder.
        """
        if dimensions is None:
            dimensions = self.discover_dimensions()
            if not dimensions:
                print(
                    "No dimensions found in data folder. Please check your data structure."
                )
                return

        print(f"Generating plots for dimensions: {dimensions}")

        for dim in dimensions:
            print(f"\nGenerating plots for dimension {dim}...")

            # Plot all 24 functions for this dimension
            self.plot_all_functions(dimension=dim)

        print(f"\nAll plots saved in 'plots/' directory:")
        import os

        if os.path.exists("plots"):
            files = [f for f in os.listdir("plots") if f.endswith(".png")]

    def get_summary_statistics(self):
        """Get summary statistics of the benchmark results."""
        if not self.data_path.exists():
            print(f"Data directory {self.data_path} does not exist")
            return

        print(f"Benchmark Results Summary")
        print(f"========================")
        print(f"Data directory: {self.data_path}")

        # List dimension folders (D2, D3, D5, D10, etc.)
        dimension_folders = [
            d for d in self.data_path.iterdir() if d.is_dir() and d.name.startswith("D")
        ]
        dimension_folders.sort(
            key=lambda x: int(x.name[1:])
        )  # Sort by dimension number

        print(f"Number of dimensions: {len(dimension_folders)}")

        total_optimizers = set()
        for dim_folder in dimension_folders:
            print(f"\n{dim_folder.name} (Dimension {dim_folder.name[1:]}):")

            # List optimizer folders within this dimension
            optimizer_folders = [d for d in dim_folder.iterdir() if d.is_dir()]

            for opt_folder in optimizer_folders:
                total_optimizers.add(opt_folder.name)
                # Count data files
                data_files = list(opt_folder.glob("**/*.dat"))
                print(f"  - {opt_folder.name}: {len(data_files)} data files")

        print(f"\nTotal unique optimizers: {len(total_optimizers)}")
        print(f"Optimizers: {', '.join(sorted(total_optimizers))}")


def main():
    """Example usage of the analyzer."""
    analyzer = ResultAnalyzer()

    # Print summary
    analyzer.get_summary_statistics()

    # Plot some results if data is available
    try:

        # All 24 functions in 6x4 grid for first available dimension
        print("\nPlotting all 24 BBOB functions...")
        available_dims = analyzer.discover_dimensions()
        if available_dims:
            analyzer.plot_all_functions(dimension=available_dims[0])
        else:
            print("No dimensions found for plotting")

        # Generate plots for multiple dimensions
        print("\nGenerating comprehensive plots...")
        analyzer.save_all_dimension_plots()  # Auto-discover dimensions from data folder

    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
