#!/usr/bin/env python3
"""
Interactive demo script for analyzing benchmark results.
"""

from analysis import ResultAnalyzer
import matplotlib.pyplot as plt

def main():
    print("BBOB Benchmark Analysis Demo")
    print("===========================")
    
    # Initialize analyzer
    analyzer = ResultAnalyzer()
    
    # Show summary
    analyzer.get_summary_statistics()
    
    print("\nTrying to create some plots...")
    
    try:
        print("\n1. Plotting convergence for Function 1, Dimension 2...")
        analyzer.plot_convergence(function_ids=[1], dimensions=[2])
        
        print("\n2. Plotting performance comparison for Functions 1-3, Dimension 2...")
        analyzer.plot_performance_comparison(function_ids=[1, 2, 3], dimension=2)
        
        print("\n3. Plotting convergence across dimensions for Function 1...")
        analyzer.plot_convergence(function_ids=[1], dimensions=[2, 3, 5])
        
    except Exception as e:
        print(f"Plotting error: {e}")
        print("This might be due to missing display or plotting backend issues.")
        print("Try running in a Jupyter notebook or with a display available.")
    
    print("\nAnalysis complete!")
    print("\nYou can also use the analyzer programmatically:")
    print("  from analysis import ResultAnalyzer")
    print("  analyzer = ResultAnalyzer()")
    print("  analyzer.plot_convergence(function_ids=[1, 2], dimensions=[2, 3])")

if __name__ == "__main__":
    main()