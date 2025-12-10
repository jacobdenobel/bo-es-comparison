#!/usr/bin/env python3
"""
Setup script for BBOB Optimizer Comparison project
"""
import subprocess
import sys
import os


def run_command(cmd, description, critical=True):
    """Run a command and handle errors."""
    print(f"{description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} successful")
        return True
    except subprocess.CalledProcessError as e:
        if critical:
            print(f"❌ {description} failed: {e.stderr.strip()}")
            return False
        else:
            print(f"⚠️  {description} had issues: {e.stderr.strip()}")
            print("Continuing anyway...")
            return True


def check_conda():
    """Check if conda is available."""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_optimizers():
    """Check available optimizers."""
    try:
        # Import and check optimizers
        sys.path.insert(0, ".")
        from optimizers import get_available_optimizers

        optimizers = get_available_optimizers()
        print(f"✅ Found {len(optimizers)} available optimizers:")
        for opt in optimizers:
            print(f"   - {opt}")
        return True
    except Exception as e:
        print(f"❌ Error checking optimizers: {e}")
        return False


def main():
    """Main setup function."""
    print("=== BBOB Optimizer Comparison Setup ===")
    print()

    # Check conda
    if not check_conda():
        print("❌ Error: conda is not installed or not in PATH")
        print(
            "Please install conda/miniconda first: https://docs.conda.io/en/latest/miniconda.html"
        )
        sys.exit(1)

    env_name = "bo-es"

    # Create environment
    if not run_command(
        f"conda create -n {env_name} python=3.12 -y",
        "Creating conda environment with Python 3.12",
    ):
        sys.exit(1)

    print()

    # Note about activation
    print("⚠️  Note: You need to manually activate the environment after this script:")
    print(f"    conda activate {env_name}")
    print()

    # Install modcma (using conda run to execute in the environment)
    if not run_command(
        f"conda run -n {env_name} pip install modcma",
        "Installing modcma",
        critical=False,
    ):
        print("modcma installation failed - CMA-ES optimizers may not work")

    print()

    # Install requirements
    if not run_command(
        f"conda run -n {env_name} pip install -r requirements.txt",
        "Installing requirements",
        critical=False,
    ):
        print("Some packages may have failed to install")
        print("The system will automatically detect available optimizers")

    print()

    # Check optimizers (in the new environment)
    print("Checking available optimizers...")
    cmd = f"conda run -n {env_name} python -c \"from optimizers import get_available_optimizers; opts=get_available_optimizers(); print(f'Available: {{opts}}')\""

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print("✅ Optimizer check successful:")
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Could not check optimizers: {e.stderr.strip()}")

    print()
    print("=== Setup Complete ===")
    print()
    print("Next steps:")
    print(f"1. Activate environment:  conda activate {env_name}")
    print("2. Run benchmark:         python benchmark.py")
    print("3. Analyze results:       python analysis.py")
    print()


if __name__ == "__main__":
    main()
