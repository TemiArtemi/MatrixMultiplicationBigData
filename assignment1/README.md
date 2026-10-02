# Matrix Multiplication Big Data - Assignment 1

## Basic Matrix Multiplication in Different Languages

This project implements and benchmarks a basic O(n³) matrix multiplication algorithm in Python, Java, and C.

## Project Structure

```
├── src/
│   ├── python/     # Python implementation
│   ├── java/       # Java implementation
│   └── c/          # C implementation
├── data/
│   ├── matrices/   # Generated input matrices
│   └── results/    # Benchmark results (CSV)
├── scripts/        # Validation, benchmarking, and analysis scripts
├── report/         # Research report
└── Instructions/   # Assignment instructions
```

## Requirements

- **Python 3.14.7** (CPython) with `psutil`
- **Java 21** (OpenJDK)
- **GCC 16.2.0** (MSYS2)

## Build and Run

### Generate input matrices (run once)
```powershell
cd src/python
python generate_matrices.py
```
This creates CSV files in `data/matrices/` for sizes: 10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000.

### Python
```powershell
cd src/python

# Correctness validation only
python matrix_mult.py --validate-only

# Full benchmark
python matrix_mult.py
```
Results saved to `data/results/results_python_<timestamp>.csv`

### Java
```powershell
cd src/java

# Compile
javac MatrixMult.java

# Correctness validation only
java MatrixMult --validate-only

# Full benchmark
java MatrixMult
```
Results saved to `data/results/results_java_<timestamp>.csv`

### C
```powershell
cd src/c

# Compile (MSYS2 GCC)
gcc -O2 -march=native -o matrix_mult matrix_mult.c

# Correctness validation only
./matrix_mult --validate-only

# Full benchmark
./matrix_mult
```
Results saved to `data/results/results_c_<timestamp>.csv`

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Matrix type | float64 (double) |
| Value range | [-10, 10] uniform |
| Random seed | 42 (fixed, reproducible) |
| Loop order | i, j, k (all languages) |
| Warm-up runs | Python: 2, Java: 5, C: 2 |
| Min repetitions | 5 per configuration |
| Time budget | 3 minutes per size |
| Batching | n=10:1000x, n=50:100x, n=100:10x |
| Timers | Python: `time.perf_counter()`, Java: `System.nanoTime()`, C: `clock_gettime(CLOCK_MONOTONIC)` |
| Memory | Python: `psutil` RSS, Java: heap used, C: `GetProcessMemoryInfo` WorkingSet |

## Correctness Validation

All implementations validated against:
- Identity matrix: A × I = A
- Zero matrix: A × 0 = 0
- Known 2×2 multiplication
- Reference implementation (NumPy / different loop order)

Tolerances: absolute 1e-9, relative 1e-9.

## Reproducibility

All raw measurements, analysis scripts, and build instructions are included to reproduce the experiments.