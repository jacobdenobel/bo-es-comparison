"""
Violin plot analysis for optimizer performance comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List, Optional

# Import the ResultAnalyzer from our analysis module
import sys
sys.path.append('.')
from analysis.performance_over_time import ResultAnalyzer, IOH_INSPECTOR_AVAILABLE


class ViolinPlotAnalyzer:
    """Creates violin plots to show distribution of optimizer performance."""

    def __init__(self, data_path: str = "data"):
        self.analyzer = ResultAnalyzer(data_path)

    def plot_optimizer_violin(self, dimensions: Optional[List[int]] = None, 
                            function_ids: Optional[List[int]] = None,
                            figsize: tuple = (16, 10)) -> None:
        """
        Create violin plots showing the distribution of final objective values for each optimizer.

        Args:
            dimensions: List of dimensions to include. If None, auto-discover.
            function_ids: List of function IDs to include. If None, use all functions 1-24.
            figsize: Figure size
        """
        if not IOH_INSPECTOR_AVAILABLE:
            print("iohinspector required for plotting")
            return

        # Auto-discover dimensions if not provided
        if dimensions is None:
            dimensions = self.analyzer.discover_dimensions()
            if not dimensions:
                print("No dimensions found in data folder.")
                return

        # Use all BBOB functions if not specified
        if function_ids is None:
            function_ids = list(range(1, 25))

        # Define consistent font sizes
        FONT_SIZES = {
            'title': 16,
            'axis_label': 14,
            'tick_label': 12,
            'legend': 12
        }

        # Create subplots for each dimension
        n_dims = len(dimensions)
        if n_dims == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        else:
            cols = min(n_dims, 3)
            rows = (n_dims + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 2))
            if n_dims == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else axes

        for idx, dim in enumerate(dimensions):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                continue

            print(f"Processing dimension {dim}...")
            
            # Load data for this dimension
            df = self.analyzer.load_data(function_ids=function_ids, dimensions=[dim])
            
            if df is None or len(df) == 0:
                ax.text(0.5, 0.5, f"No data for D={dim}", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=FONT_SIZES['axis_label'])
                ax.set_title(f"Dimension {dim}", fontsize=FONT_SIZES['title'])
                continue

            # Get final objective values (best_y) for each optimizer
            try:
                # Group by algorithm and get final best values
                final_values = []
                algorithm_names = []
                
                # Get unique algorithms
                algorithms = df['algorithm_name'].unique()
                
                for alg in algorithms:
                    if alg in ["geometric_mean", "None", "variable"]:
                        continue
                    
                    alg_data = df.filter(df['algorithm_name'] == alg)
                    
                    # Get final best values for each run
                    if len(alg_data) > 0:
                        # Group by data_id to get final value for each run
                        run_groups = alg_data.group_by(['data_id', 'function_id', 'instance'])
                        
                        for group_key, group_data in run_groups:
                            # Get the best value achieved in this run
                            best_val = group_data['best_y'].min()
                            final_values.append(best_val)
                            algorithm_names.append(alg)

                if not final_values:
                    ax.text(0.5, 0.5, f"No valid data for D={dim}", 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=FONT_SIZES['axis_label'])
                    ax.set_title(f"Dimension {dim}", fontsize=FONT_SIZES['title'])
                    continue

                # Create DataFrame for seaborn
                plot_data = pd.DataFrame({
                    'Objective Value': final_values,
                    'Optimizer': algorithm_names
                })

                # Create violin plot
                sns.violinplot(data=plot_data, x='Optimizer', y='Objective Value', ax=ax, 
                              inner='box', palette='Set2')
                
                # Customize the plot
                ax.set_title(f"Objective Value Distribution - Dimension {dim}", 
                           fontsize=FONT_SIZES['title'])
                ax.set_xlabel("Optimizer", fontsize=FONT_SIZES['axis_label'])
                ax.set_ylabel("Final Objective Value (log scale)", fontsize=FONT_SIZES['axis_label'])
                
                # Use log scale for better visualization
                ax.set_yscale('log')
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45, labelsize=FONT_SIZES['tick_label'])
                ax.tick_params(axis='y', labelsize=FONT_SIZES['tick_label'])
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                n_runs = len(plot_data)
                n_optimizers = len(plot_data['Optimizer'].unique())
                ax.text(0.02, 0.98, f"Runs: {n_runs}\nOptimizers: {n_optimizers}", 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=FONT_SIZES['tick_label'])

            except Exception as e:
                ax.text(0.5, 0.5, f"Error plotting D={dim}:\n{str(e)[:50]}...", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=FONT_SIZES['tick_label'])
                ax.set_title(f"Dimension {dim} (Error)", fontsize=FONT_SIZES['title'])

        # Remove any unused subplots
        if n_dims < len(axes):
            for idx in range(n_dims, len(axes)):
                fig.delaxes(axes[idx])

        # Overall title
        fig.suptitle("Optimizer Performance Distribution Comparison", 
                    fontsize=FONT_SIZES['title'] + 2, y=0.98)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the plot
        import os
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/violin_optimizer_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Violin plot saved as: {filename}")

        plt.show()

    def plot_single_dimension_violin(self, dimension: int, 
                                   function_ids: Optional[List[int]] = None,
                                   figsize: tuple = (12, 8)) -> None:
        """
        Create a violin plot for a single dimension with better formatting.
        
        Args:
            dimension: Dimension to plot
            function_ids: List of function IDs to include. If None, use all functions 1-24.
            figsize: Figure size
        """
        if not IOH_INSPECTOR_AVAILABLE:
            print("iohinspector required for plotting")
            return

        # Use all BBOB functions if not specified
        if function_ids is None:
            function_ids = list(range(1, 25))

        print(f"Creating violin plot for dimension {dimension}...")
        
        # Load data for this dimension
        df = self.analyzer.load_data(function_ids=function_ids, dimensions=[dimension])
        
        if df is None or len(df) == 0:
            print(f"No data found for dimension {dimension}")
            return

        try:
            # Prepare data for plotting
            final_values = []
            algorithm_names = []
            
            # Get unique algorithms
            algorithms = df['algorithm_name'].unique()
            
            for alg in algorithms:
                if alg in ["geometric_mean", "None", "variable"]:
                    continue
                
                alg_data = df.filter(df['algorithm_name'] == alg)
                
                # Get final best values for each run
                if len(alg_data) > 0:
                    # Group by data_id to get final value for each run
                    run_groups = alg_data.group_by(['data_id', 'function_id', 'instance'])
                    
                    for group_key, group_data in run_groups:
                        # Get the best value achieved in this run
                        best_val = group_data['best_y'].min()
                        final_values.append(best_val)
                        algorithm_names.append(alg)

            if not final_values:
                print(f"No valid data found for dimension {dimension}")
                return

            # Create DataFrame for seaborn
            plot_data = pd.DataFrame({
                'Objective Value': final_values,
                'Optimizer': algorithm_names
            })

            # Create the plot
            plt.figure(figsize=figsize)
            
            # Create violin plot with better styling
            ax = sns.violinplot(data=plot_data, x='Optimizer', y='Objective Value', 
                               inner='box', palette='Set2')
            
            # Customize the plot
            plt.title(f"Optimizer Performance Distribution - Dimension {dimension}", 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel("Optimizer", fontsize=14, fontweight='bold')
            plt.ylabel("Final Objective Value (log scale)", fontsize=14, fontweight='bold')
            
            # Use log scale for better visualization
            plt.yscale('log')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            n_runs = len(plot_data)
            n_optimizers = len(plot_data['Optimizer'].unique())
            n_functions = len(function_ids)
            
            stats_text = f"Functions: {n_functions}\nRuns: {n_runs}\nOptimizers: {n_optimizers}"
            plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                    fontsize=11, fontweight='bold')

            # Adjust layout
            plt.tight_layout()

            # Save the plot
            import os
            os.makedirs("plots", exist_ok=True)
            filename = f"plots/violin_D{dimension}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Violin plot saved as: {filename}")

            plt.show()

        except Exception as e:
            print(f"Error creating violin plot for dimension {dimension}: {e}")

def main():
    """Main function to demonstrate violin plot usage."""
    print("=== Violin Plot Analysis ===")
    
    # Create analyzer
    violin_analyzer = ViolinPlotAnalyzer()
    
    # Check available data
    violin_analyzer.analyzer.get_summary_statistics()
    
    # Generate individual plots for specific dimensions
    available_dims = violin_analyzer.analyzer.discover_dimensions()
    
    if available_dims:
        print(f"\nGenerating individual plots for each dimension...")
        for dim in available_dims:
            violin_analyzer.plot_single_dimension_violin(dimension=dim)
    
    print("\nViolin plot analysis complete!")


if __name__ == "__main__":
    main()
