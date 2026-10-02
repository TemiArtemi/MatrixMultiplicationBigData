#!/usr/bin/env python3
"""
Matrix multiplication benchmark - Python implementation.

Implements basic O(n³) triple-loop matrix multiplication.
Reads matrices from CSV, measures kernel time and memory.
"""

import os
import sys
import csv
import time
import random
import psutil
import argparse
from pathlib import Path
from typing import List, Tuple

# Try to import numpy for reference validation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available. Reference validation will be skipped.")


# Configuration
SEED = 42
VALUE_MIN = -10.0
VALUE_MAX = 10.0
SIZES = [10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000]
MATRICES_DIR = Path(__file__).parent.parent.parent / "data" / "matrices"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"

# Benchmark settings
WARMUP_RUNS = 2
MIN_REPETITIONS = 5
TIME_BUDGET_SECONDS = 180  # 3 minutes per configuration
BATCH_SIZES = {10: 1000, 50: 100, 100: 10}  # sizes -> batches for very small matrices

# Tolerances for correctness
ABS_TOL = 1e-9
REL_TOL = 1e-9


def load_matrix_csv(filepath: Path) -> List[List[float]]:
    """Load matrix from CSV file."""
    matrix = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            matrix.append([float(x) for x in row])
    return matrix


def multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Basic triple-loop matrix multiplication: C = A × B."""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def multiply_batch(A: List[List[float]], B: List[List[float]], batches: int) -> List[List[float]]:
    """Run multiplication multiple times (for very small matrices) and return last result."""
    C = None
    for _ in range(batches):
        C = multiply(A, B)
    return C


def matrices_equal(A: List[List[float]], B: List[List[float]], abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    """Check if two matrices are equal within tolerances."""
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            return False
        for j in range(len(A[i])):
            a, b = A[i][j], B[i][j]
            diff = abs(a - b)
            if diff > abs_tol and diff > rel_tol * max(abs(a), abs(b)):
                return False
    return True


def identity_matrix(n: int) -> List[List[float]]:
    """Create n×n identity matrix."""
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I


def zero_matrix(n: int) -> List[List[float]]:
    """Create n×n zero matrix."""
    return [[0.0] * n for _ in range(n)]


def validate_correctness() -> bool:
    """Run correctness tests."""
    print("=" * 60)
    print("CORRECTNESS VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Identity matrix (A × I = A)
    print("\n1. Testing with identity matrix...")
    rng = random.Random(SEED)
    n = 5
    A = [[rng.uniform(VALUE_MIN, VALUE_MAX) for _ in range(n)] for _ in range(n)]
    I = identity_matrix(n)
    C = multiply(A, I)
    if matrices_equal(C, A):
        print("   PASS: A × I = A")
    else:
        print("   FAIL: A × I != A")
        all_passed = False
    
    # Test 2: Zero matrix (A × 0 = 0)
    print("\n2. Testing with zero matrix...")
    Z = zero_matrix(n)
    C = multiply(A, Z)
    expected = zero_matrix(n)
    if matrices_equal(C, expected):
        print("   PASS: A × 0 = 0")
    else:
        print("   FAIL: A × 0 != 0")
        all_passed = False
    
    # Test 3: Small known matrices
    print("\n3. Testing with known small matrices...")
    A_test = [[1.0, 2.0], [3.0, 4.0]]
    B_test = [[5.0, 6.0], [7.0, 8.0]]
    expected = [[19.0, 22.0], [43.0, 50.0]]
    C = multiply(A_test, B_test)
    if matrices_equal(C, expected):
        print("   PASS: 2×2 known result")
    else:
        print(f"   FAIL: Expected {expected}, got {C}")
        all_passed = False
    
    # Test 4: Reference comparison with NumPy (if available)
    if HAS_NUMPY:
        print("\n4. Testing against NumPy reference...")
        n = 20
        A = [[rng.uniform(VALUE_MIN, VALUE_MAX) for _ in range(n)] for _ in range(n)]
        B = [[rng.uniform(VALUE_MIN, VALUE_MAX) for _ in range(n)] for _ in range(n)]
        C = multiply(A, B)
        
        A_np = np.array(A, dtype=np.float64)
        B_np = np.array(B, dtype=np.float64)
        C_np = np.matmul(A_np, B_np)
        
        if matrices_equal(C, C_np.tolist()):
            print("   PASS: Matches NumPy result")
        else:
            print("   FAIL: Differs from NumPy result")
            all_passed = False
    else:
        print("\n4. Skipping NumPy reference test (not installed)")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CORRECTNESS TESTS PASSED")
    else:
        print("SOME CORRECTNESS TESTS FAILED")
    print("=" * 60)
    
    return all_passed


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_size(n: int) -> List[dict]:
    """Run benchmark for a specific matrix size."""
    print(f"\n  Benchmarking n={n}...")
    
    # Load matrices
    a_path = MATRICES_DIR / f"A_{n}.csv"
    b_path = MATRICES_DIR / f"B_{n}.csv"
    
    if not a_path.exists() or not b_path.exists():
        print(f"    Matrices not found for n={n}, skipping")
        return []
    
    A = load_matrix_csv(a_path)
    B = load_matrix_csv(b_path)
    
    # Pre-allocate result matrix (outside timing)
    C = [[0.0] * n for _ in range(n)]
    
    # Warm-up runs
    print(f"    Warm-up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        multiply(A, B)
    
    # Determine if we need batching
    batches = BATCH_SIZES.get(n, 1)
    
    # Timed runs
    results = []
    for rep in range(MIN_REPETITIONS):
        # Measure memory before
        mem_before = get_memory_mb()
        
        # Time the kernel
        start = time.perf_counter()
        if batches > 1:
            multiply_batch(A, B, batches)
        else:
            multiply(A, B)
        end = time.perf_counter()
        
        # Measure memory after
        mem_after = get_memory_mb()
        
        elapsed_ms = (end - start) * 1000
        if batches > 1:
            elapsed_ms /= batches
        
        memory_mb = max(mem_after - mem_before, 0)
        
        results.append({
            'language': 'Python',
            'size': n,
            'numeric_type': 'float64',
            'repetition': rep + 1,
            'time_ms': elapsed_ms,
            'memory_mb': memory_mb
        })
        
        print(f"    Rep {rep + 1}: {elapsed_ms:.2f} ms, memory delta: {memory_mb:.2f} MB")
        
        # Check time budget
        if elapsed_ms > TIME_BUDGET_SECONDS * 1000:
            print(f"    Time budget exceeded ({TIME_BUDGET_SECONDS}s), stopping")
            break
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Python matrix multiplication benchmark')
    parser.add_argument('--validate-only', action='store_true', help='Only run correctness validation')
    parser.add_argument('--sizes', nargs='+', type=int, help='Specific sizes to test')
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Python Matrix Multiplication Benchmark")
    print(f"Matrices directory: {MATRICES_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Sizes: {SIZES}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Min repetitions: {MIN_REPETITIONS}")
    print(f"Time budget: {TIME_BUDGET_SECONDS}s per size")
    
    # Run correctness validation first
    if not validate_correctness():
        print("Correctness validation failed. Aborting benchmark.")
        sys.exit(1)
    
    if args.validate_only:
        return
    
    # Run benchmarks
    sizes_to_test = args.sizes if args.sizes else SIZES
    all_results = []
    
    for n in sizes_to_test:
        results = benchmark_size(n)
        all_results.extend(results)
    
    # Save raw results to CSV
    if all_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"results_python_{timestamp}.csv"
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'language', 'size', 'numeric_type', 'repetition', 'time_ms', 'memory_mb'
            ])
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        print("\nSUMMARY (median time per size):")
        print("-" * 50)
        for n in sizes_to_test:
            size_results = [r for r in all_results if r['size'] == n]
            if size_results:
                times = sorted([r['time_ms'] for r in size_results])
                median = times[len(times) // 2]
                q1 = times[len(times) // 4]
                q3 = times[3 * len(times) // 4]
                iqr = q3 - q1
                print(f"  n={n:5d}: {median:10.2f} ms (IQR: {iqr:.2f} ms)")
    
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()