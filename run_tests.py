# run_tests.py
# Test runner script for entropy calculator module

import subprocess
import sys
import os

def run_tests():
    """Run comprehensive tests for entropy calculator module."""
    
    print("=" * 60)
    print("ENTROPY CALCULATOR MODULE TESTS")
    print("=" * 60)
    
    # Check if required packages are installed
    try:
        import torch
        import pytest
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Install requirements: pip install -r requirements.txt")
        return False
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    print("✓ Logs directory ready")
    
    # Run tests
    print("\nRunning tests...")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_entropy_calculator.py",
            "-v",                    # Verbose output
            "--tb=short",           # Short traceback format
            "--durations=10",       # Show 10 slowest tests
            "--cov=src/core",       # Coverage for core module
            "--cov-report=term-missing"  # Show missing lines
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✓ ALL TESTS PASSED")
            print("✓ Entropy calculator module is ready for use")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("✗ SOME TESTS FAILED")
            print("✗ Module needs fixes before proceeding")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False

def main():
    """Main test runner."""
    success = run_tests()
    
    if success:
        print("\nNext steps:")
        print("1. Module is ready for Figure 1 reproduction")
        print("2. Proceed to implement synthetic data generator")
        print("3. Create GPT model for experiments")
        sys.exit(0)
    else:
        print("\nAction required:")
        print("1. Fix failing tests")
        print("2. Re-run: python run_tests.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
