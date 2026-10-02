#!/usr/bin/env python3
"""
Generate random matrices for matrix multiplication benchmarking.

Generates pairs of matrices (A, B) for each size in the test sequence.
Saves them as CSV files that can be read by Python, Java, and C.

Matrices: float64, values in range [-10, 10], fixed seed for reproducibility.
"""

import os
import csv
import random
from pathlib import Path


# Configuration
SEED = 42
VALUE_MIN = -10.0
VALUE_MAX = 10.0
SIZES = [10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000]
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "matrices"


def generate_matrix(n: int, rng: random.Random) -> list[list[float]]:
    """Generate an n x n matrix with random float64 values."""
    return [[rng.uniform(VALUE_MIN, VALUE_MAX) for _ in range(n)] for _ in range(n)]


def save_matrix_csv(matrix: list[list[float]], filepath: Path) -> None:
    """Save matrix as CSV (one row per line, comma-separated)."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    print(f"Generating matrices with seed={SEED}, range=[{VALUE_MIN}, {VALUE_MAX}]")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sizes: {SIZES}")
    print()

    for n in SIZES:
        print(f"  n={n:5d} ... ", end="", flush=True)

        # Generate matrices A and B
        A = generate_matrix(n, rng)
        B = generate_matrix(n, rng)

        # Save as CSV
        a_path = OUTPUT_DIR / f"A_{n}.csv"
        b_path = OUTPUT_DIR / f"B_{n}.csv"

        save_matrix_csv(A, a_path)
        save_matrix_csv(B, b_path)

        print(f"done (A: {a_path.name}, B: {b_path.name})")

    print()
    print("All matrices generated successfully.")


if __name__ == "__main__":
    main()