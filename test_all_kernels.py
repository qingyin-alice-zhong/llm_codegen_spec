#!/usr/bin/env python3
"""
Test all kernels systematically and identify which ones need fixes
"""

import os
import subprocess
import sys
from pathlib import Path
import time

def test_kernel(kernel_dir):
    """Test a single kernel and return result"""
    test_py = kernel_dir / "test.py"
    if not test_py.exists():
        return "NO_TEST", "Missing test.py file"
    
    try:
        # Run the test with a timeout
        cmd = ["python", "test.py", "--kernel_path", "canonical_scalar_allo.cc"]
        result = subprocess.run(
            cmd,
            cwd=kernel_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            if "PASS!" in result.stdout:
                return "PASS", "Success"
            else:
                return "FAIL", "Test failed without error code"
        else:
            # Extract key error information
            error_lines = result.stderr.split('\n') if result.stderr else []
            stdout_lines = result.stdout.split('\n') if result.stdout else []
            
            # Look for specific error patterns
            for line in error_lines + stdout_lines:
                if "uint8" in line and "ValueError" in line:
                    return "UINT8_BUG", "Allo uint8 parsing bug"
                elif "invalid literal for int()" in line and "t8" in line:
                    return "UINT8_BUG", "Allo uint8 parsing bug"
                elif "couldn't parse function signature" in line:
                    return "SIGNATURE_ERROR", "Function signature parsing error"
                elif "FileNotFoundError" in line:
                    return "MISSING_FILE", "Missing required file"
                elif "ImportError" in line or "ModuleNotFoundError" in line:
                    return "IMPORT_ERROR", "Missing dependencies"
            
            # Return first few lines of error for debugging
            error_summary = "\n".join((result.stderr or result.stdout or "Unknown error").split('\n')[:3])
            return "ERROR", error_summary
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "Test timed out after 5 minutes"
    except Exception as e:
        return "EXCEPTION", str(e)

def main():
    """Test all kernels and categorize results"""
    dataset_dir = Path("npueval_dataset")
    
    if not dataset_dir.exists():
        print("npueval_dataset directory not found")
        return
    
    # Get all kernel directories (exclude tbd)
    kernel_dirs = []
    for item in dataset_dir.iterdir():
        if item.is_dir() and item.name not in {"__pycache__", "tbd", "utils.py"}:
            kernel_dirs.append(item)
    
    kernel_dirs.sort()
    
    print(f"Testing {len(kernel_dirs)} kernels...")
    print("=" * 80)
    
    results = {
        "PASS": [],
        "FAIL": [],
        "ERROR": [],
        "UINT8_BUG": [],
        "SIGNATURE_ERROR": [],
        "MISSING_FILE": [],
        "IMPORT_ERROR": [],
        "TIMEOUT": [],
        "NO_TEST": [],
        "EXCEPTION": []
    }
    
    for i, kernel_dir in enumerate(kernel_dirs, 1):
        kernel_name = kernel_dir.name
        print(f"[{i:3d}/{len(kernel_dirs)}] Testing {kernel_name}...", end="", flush=True)
        
        start_time = time.time()
        status, message = test_kernel(kernel_dir)
        duration = time.time() - start_time
        
        results[status].append((kernel_name, message))
        
        print(f" {status} ({duration:.1f}s)")
        if status not in ["PASS"] and message:
            print(f"    → {message[:100]}...")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    
    for status, kernels in results.items():
        if kernels:
            print(f"\n{status}: {len(kernels)} kernels")
            for kernel_name, message in kernels:
                print(f"  - {kernel_name}: {message[:80]}...")
    
    print(f"\nTOTAL: {len(kernel_dirs)} kernels tested")
    print(f"PASS: {len(results['PASS'])}")
    print(f"ISSUES: {len(kernel_dirs) - len(results['PASS'])}")
    
    # Write detailed results to file
    with open("test_results.txt", "w") as f:
        f.write("KERNEL TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for status, kernels in results.items():
            if kernels:
                f.write(f"{status}: {len(kernels)} kernels\n")
                f.write("-" * 30 + "\n")
                for kernel_name, message in kernels:
                    f.write(f"{kernel_name}: {message}\n")
                f.write("\n")
    
    print(f"\nDetailed results written to test_results.txt")

if __name__ == "__main__":
    main()


