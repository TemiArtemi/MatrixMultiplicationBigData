#!/usr/bin/env python3
"""
External (cross-language) correctness cross-check with NumPy.

Workflow:
  1. assignment2\\src\\c\\a2_bench.exe --dump-result <n> <dump.csv>
     (runs the C baseline kernel on the shared A_n x B_n inputs and writes C)
  2. python assignment2/scripts/verify_numpy.py --n <n> --dump <dump.csv>

The script recomputes A x B with NumPy (an optimised BLAS implementation is
used ONLY as a trusted reference for validation, never as a benchmarked
method) and compares it element-wise with the C result.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
ABS_TOL = 1e-9
REL_TOL = 1e-9


def load_matrix(path: Path) -> np.ndarray:
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append([float(x) for x in row])
    return np.array(rows, dtype=np.float64)


def find_matrix(which: str, n: int) -> Path:
    for d in (BASE / "assignment2" / "data" / "matrices",
              BASE / "assignment1" / "data" / "matrices"):
        p = d / f"{which}_{n}.csv"
        if p.exists():
            return p
    sys.exit(f"Matrix {which}_{n}.csv not found; run scripts/generate_matrices.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dump", required=True, help="C result CSV from --dump-result")
    args = ap.parse_args()

    a = load_matrix(find_matrix("A", args.n))
    b = load_matrix(find_matrix("B", args.n))
    c_dump = load_matrix(Path(args.dump))

    if a.shape != (args.n, args.n) or b.shape != (args.n, args.n):
        sys.exit(f"Input shape mismatch: {a.shape}, {b.shape}")
    if c_dump.shape != (args.n, args.n):
        sys.exit(f"Dumped C shape mismatch: {c_dump.shape}")

    ref = a @ b
    diff = np.abs(c_dump - ref)
    max_abs = float(diff.max())
    ok = bool(np.all(diff <= ABS_TOL + REL_TOL * np.abs(ref)))

    print(f"n = {args.n}")
    print(f"NumPy trace(C)  = {np.trace(ref):.6e}")
    print(f"C    trace(C)   = {np.trace(c_dump):.6e}")
    print(f"max |C - NumPy| = {max_abs:.3e}  (tol: {ABS_TOL} + {REL_TOL}*|ref|)")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
