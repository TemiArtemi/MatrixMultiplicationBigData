# Matrix Multiplication Big Data - Assignment 1

## Basic Matrix Multiplication in Different Languages

This project implements and benchmarks a basic O(n³) matrix multiplication algorithm in Python, Java, and C.

## Project Structure

```text
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

- **Python 3.11.9 (CPython)** with psutil, pandas, matplotlib, seaborn
- **Java 21 (OpenJDK)**
- **GCC 16.2.0 (MSYS2)**

## Build and Run Instructions

⚠ **IMPORTANT EXECUTION NOTE:**

All scripts and commands rely on relative paths to locate data and configuration files. To avoid FileNotFoundError or unexpected behavior, you must execute all commands from the exact working directories specified below.

### 1. Generate input matrices (run once)

```powershell
cd src/python
python generate_matrices.py
cd ../..
```

This creates CSV files in `data/matrices/` for sizes: 10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000.

### 2. Python Benchmark

```powershell
cd src/python

# Correctness validation only
python matrix_mult.py --validate-only

# Full benchmark
python matrix_mult.py
cd ../..
```

Results saved to `data/results/results_python_<timestamp>.csv`

### 3. Java Benchmark

```powershell
cd src/java

# Compile
javac MatrixMult.java

# Correctness validation only
java MatrixMult --validate-only

# Full benchmark
java MatrixMult
cd ../..
```

Results saved to `data/results/results_java_<timestamp>.csv`

### 4. C Benchmark

```powershell
cd src/c

# Compile (MSYS2 GCC)
gcc -O2 -march=native -o matrix_mult matrix_mult.c

# Correctness validation only
./matrix_mult --validate-only

# Full benchmark
./matrix_mult
cd ../..
```

Results saved to `data/results/results_c_<timestamp>.csv`

### 5. Generate Analysis and Plots (Phase 6)

To reproduce the summary statistics and the execution time plots found in the final report, run the analysis script from the root of the project:

```powershell
# Ensure you are at the project root folder (MatrixMultiplicationBigData)
python assignment1/scripts/analyze_results.py
```

This will read the raw CSVs from `data/results/` and output `summary_statistics.csv` and the `.png` plots directly into the `report/` folder.

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Matrix type | float64 (double) |
| Value range | [-10, 10] uniform |
| Random seed | 42 (fixed, reproducible) |
| Loop order | i, j, k (all languages) |
| Warm-up runs | Python: 2, Java: 5, C: 2 |
| Min repetitions | 5 or 2 per configuration |
| Time budget | 10 minutes per size |
| Batching | n=10:1000x, n=50:100x, n=100:10x |
| Timers | Python: time.perf_counter(), Java: System.nanoTime(), C: clock_gettime(CLOCK_MONOTONIC) |
| Memory | Python: psutil RSS, Java: heap used, C: GetProcessMemoryInfo WorkingSet |

## Correctness Validation

All implementations validated against:

- Identity matrix: A × I = A
- Zero matrix: A × 0 = 0
- Known 2×2 multiplication
- Reference implementation (NumPy / different loop order)
- Tolerances: absolute 1e-9, relative 1e-9.

## Reproducibility

All raw measurements, analysis scripts, and build instructions are included to reproduce the experiments.