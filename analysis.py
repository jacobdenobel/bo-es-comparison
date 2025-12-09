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
    
    def load_data(self, 
                  function_ids: Optional[List[int]] = None,
                  dimensions: Optional[List[int]] = None,
                  instances: Optional[List[int]] = None):
        """
        Load benchmark data.
        
        Args:
            function_ids: List of function IDs to load
            dimensions: List of dimensions to load
            instances: List of instance IDs to load
        
        Returns:
            Loaded data or None if iohinspector not available
        """
        if not IOH_INSPECTOR_AVAILABLE or self.manager is None:
            print("iohinspector not available for data loading")
            return None
        
        selection = self.manager.select(
            function_ids=function_ids,
            dimensions=dimensions,
            instances=instances
        )
        return selection.load(monotonic=True, include_meta_data=True)
    
    def plot_convergence(self, 
                        function_ids: List[int] = [1], 
                        dimensions: List[int] = [2],
                        figsize: tuple = (20, 8)) -> None:
        """
        Plot convergence curves for specified functions and dimensions.
        
        Args:
            function_ids: List of function IDs to plot
            dimensions: List of dimensions to plot
            figsize: Figure size
        """
        if not IOH_INSPECTOR_AVAILABLE:
            print("iohinspector required for plotting")
            return
        
        df = self.load_data(function_ids=function_ids, dimensions=dimensions)
        if df is None or len(df) == 0:
            print("No data available for plotting")
            return
        
        fig, axes = plt.subplots(len(function_ids), len(dimensions), 
                               figsize=figsize, squeeze=False)
        
        # Adjust figure size to accommodate legends
        fig.subplots_adjust(right=0.85)
        
        for i, fid in enumerate(function_ids):
            for j, dim in enumerate(dimensions):
                ax = axes[i, j]
                
                # Filter data for this function and dimension
                try:
                    subset = df[(df['function_id'] == fid) & (df['dimension'] == dim)]
                    
                    if len(subset) > 0:
                        iohinspector.plot.single_function_fixedbudget(subset, ax=ax)
                        ax.set_title(f'Function {fid}, Dimension {dim}')
                        ax.grid(True)
                        # Move legend outside the plot area
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax.text(0.5, 0.5, 'No data', 
                               transform=ax.transAxes, ha='center')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {e}', 
                           transform=ax.transAxes, ha='center')
        
        # Use tight_layout with padding for legends
        try:
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        except:
            pass  # If tight_layout fails, continue without it
        plt.show()
    
    def plot_performance_comparison(self, 
                                   function_ids: List[int] = [1, 2, 3],
                                   dimension: int = 2,
                                   figsize: tuple = (35, 5)) -> None:
        """
        Plot performance comparison across multiple functions.
        
        Args:
            function_ids: List of function IDs to compare
            dimension: Dimension to analyze
            figsize: Figure size
        """
        if not IOH_INSPECTOR_AVAILABLE:
            print("iohinspector required for plotting")
            return
        
        fig, axes = plt.subplots(1, len(function_ids), figsize=figsize)
        if len(function_ids) == 1:
            axes = [axes]
        
        # Adjust figure size to accommodate legends
        fig.subplots_adjust(right=0.8)
        
        for i, fid in enumerate(function_ids):
            df = self.load_data(function_ids=[fid], dimensions=[dimension])
            if df is not None and len(df) > 0:
                try:
                    iohinspector.plot.single_function_fixedbudget(df, ax=axes[i])
                    axes[i].set_title(f'Function {fid} (D={dimension})')
                    axes[i].grid(True)
                    # Move legend outside the plot area
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error: {e}', 
                               transform=axes[i].transAxes, ha='center')
            else:
                axes[i].text(0.5, 0.5, 'No data', 
                           transform=axes[i].transAxes, ha='center')
        
        # Use tight_layout with padding for legends
        try:
            plt.tight_layout(rect=[0, 0, 0.8, 1])
        except:
            pass  # If tight_layout fails, continue without it
        plt.show()
    
    def get_summary_statistics(self):
        """Get summary statistics of the benchmark results."""
        if not self.data_path.exists():
            print(f"Data directory {self.data_path} does not exist")
            return
        
        print(f"Benchmark Results Summary")
        print(f"========================")
        print(f"Data directory: {self.data_path}")
        
        # List optimizer folders
        optimizer_folders = [d for d in self.data_path.iterdir() if d.is_dir()]
        print(f"Number of optimizers: {len(optimizer_folders)}")
        
        for folder in optimizer_folders:
            print(f"  - {folder.name}")
            # Count data files
            data_files = list(folder.glob("**/*.dat"))
            print(f"    Data files: {len(data_files)}")


def main():
    """Example usage of the analyzer."""
    analyzer = ResultAnalyzer()
    
    # Print summary
    analyzer.get_summary_statistics()
    
    # Plot some results if data is available
    try:
        analyzer.plot_convergence(function_ids=[1], dimensions=[2])
        analyzer.plot_performance_comparison(function_ids=[1, 2, 3], dimension=2)
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()