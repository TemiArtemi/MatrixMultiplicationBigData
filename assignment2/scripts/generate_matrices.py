#!/usr/bin/env python3
"""
Generate the dense input matrices used by Assignment 2, Part A.

This generator is intentionally IDENTICAL to the Assignment 1 generator
(assignment1/src/python/generate_matrices.py): same Mersenne Twister stream
(random.Random with seed 42), same value range, same size order. Running it
regenerates bit-identical CSV files, so the Assignment 2 submission is
self-contained without shipping the ~190 MB of matrices: the harness first
looks for A_n.csv / B_n.csv in assignment2/data/matrices and falls back to
the reused Assignment 1 copies in assignment1/data/matrices.

Output: data/matrices/A_<n>.csv and data/matrices/B_<n>.csv
"""

import csv
import random
from pathlib import Path

# Configuration (must match the Assignment 1 generator exactly)
SEED = 42
VALUE_MIN = -10.0
VALUE_MAX = 10.0
SIZES = [10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "matrices"


def generate_matrix(n: int, rng: random.Random) -> list[list[float]]:
    """Generate an n x n matrix with random float64 values."""
    return [[rng.uniform(VALUE_MIN, VALUE_MAX) for _ in range(n)] for _ in range(n)]


def save_matrix_csv(matrix: list[list[float]], filepath: Path) -> None:
    """Save matrix as CSV (one row per line, comma-separated)."""
    with open(filepath, "w", newline="") as f:
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
        a = generate_matrix(n, rng)
        b = generate_matrix(n, rng)
        save_matrix_csv(a, OUTPUT_DIR / f"A_{n}.csv")
        save_matrix_csv(b, OUTPUT_DIR / f"B_{n}.csv")
        print("done")

    print()
    print("All matrices generated successfully (bit-identical to Assignment 1).")


if __name__ == "__main__":
    main()
