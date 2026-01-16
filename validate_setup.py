#!/usr/bin/env python3
"""Quick validation script to check if the setup is correct."""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_imports():
    """Check if imports work (without actually importing, just syntax check)."""
    try:
        # Check syntax
        import py_compile
        files = [
            "src/mlops_project/data.py",
            "src/mlops_project/model.py",
            "src/mlops_project/train.py"
        ]
        for f in files:
            try:
                py_compile.compile(f, doraise=True)
                print(f"✓ Syntax check passed: {f}")
            except py_compile.PyCompileError as e:
                print(f"✗ Syntax error in {f}: {e}")
                return False
        return True
    except Exception as e:
        print(f"✗ Import check failed: {e}")
        return False

def check_configs():
    """Check if config files exist."""
    configs = [
        "configs/config.yaml",
        "configs/data.yaml",
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/wandb.yaml"
    ]
    all_exist = True
    for config in configs:
        if not check_file_exists(config, "Config file"):
            all_exist = False
    return all_exist

def check_dockerfiles():
    """Check if Dockerfiles exist."""
    dockerfiles = [
        "dockerfiles/train.dockerfile",
        "dockerfiles/api.dockerfile"
    ]
    all_exist = True
    for df in dockerfiles:
        if not check_file_exists(df, "Dockerfile"):
            all_exist = False
    return all_exist

def check_structure():
    """Check project structure."""
    print("\n=== Checking Project Structure ===")
    
    structure_ok = True
    structure_ok &= check_file_exists("pyproject.toml", "Project config")
    structure_ok &= check_file_exists("tasks.py", "Tasks file")
    structure_ok &= check_file_exists("src/mlops_project/__init__.py", "Package init")
    structure_ok &= check_configs()
    structure_ok &= check_dockerfiles()
    
    return structure_ok

def main():
    """Run all validation checks."""
    print("=" * 50)
    print("MLOps Project Setup Validation")
    print("=" * 50)
    
    # Check structure
    structure_ok = check_structure()
    
    # Check syntax
    print("\n=== Checking Code Syntax ===")
    syntax_ok = check_imports()
    
    # Summary
    print("\n" + "=" * 50)
    print("Validation Summary")
    print("=" * 50)
    
    if structure_ok and syntax_ok:
        print("✓ All basic checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: uv sync")
        print("2. Download data: invoke preprocess_data")
        print("3. Run tests: invoke test")
        print("4. Train model: invoke train")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

