#!/usr/bin/env python3
"""
Environment verification script for EC349 project.
Run this script to verify that all dependencies and configurations are correct.
"""

import sys
import os

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking package imports...")
    try:
        import torch
        import transformers
        import pandas
        import numpy
        import sklearn
        import pytorch_lightning
        import pyarrow
        import dotenv
        import tqdm
        import pytest
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_pytorch():
    """Check PyTorch installation and GPU support."""
    print("\nChecking PyTorch configuration...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")
        if torch.backends.mps.is_available():
            print("✓ Apple GPU (MPS) support is available")
        else:
            print("⚠ Apple GPU (MPS) support is not available (will use CPU)")
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False

def check_project_structure():
    """Check if required directories and files exist."""
    print("\nChecking project structure...")
    required_dirs = ['data', 'src', 'models', 'outputs']
    required_files = [
        'requirements.txt',
        'setup.py',
        'src/config.py',
        'data/yelp_business_data.csv',
        'data/yelp_review.csv',
        'data/yelp_user.csv'
    ]
    
    all_good = True
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ Directory '{dir_name}' exists")
        else:
            print(f"  ✗ Directory '{dir_name}' is missing")
            all_good = False
    
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"  ✓ File '{file_name}' exists")
        else:
            print(f"  ⚠ File '{file_name}' is missing (may be optional)")
    
    return all_good

def check_source_modules():
    """Check if source modules can be imported."""
    print("\nChecking source modules...")
    try:
        sys.path.insert(0, '.')
        from src import config, data_loading, preprocessing, features
        from src import sentiment, feature_selection, model, train, utils
        print("✓ All source modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Source module import failed: {e}")
        return False

def check_config():
    """Check configuration settings."""
    print("\nChecking configuration...")
    try:
        from src import config
        print(f"  Data directory: {config.DATA_DIR}")
        print(f"  Learning rate: {config.LEARNING_RATE}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Device: {config.get_device()}")
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration check failed: {e}")
        return False

def main():
    """Run all environment checks."""
    print("=" * 60)
    print("EC349 Project Environment Verification")
    print("=" * 60)
    
    checks = [
        check_imports(),
        check_pytorch(),
        check_project_structure(),
        check_source_modules(),
        check_config()
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✓ Environment is ready!")
        print("\nYou can now run the project using:")
        print("  python examples/quick_start.py")
        print("  python examples/inference.py")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
