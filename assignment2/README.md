# Matrix Multiplication Big Data — Assignment 2

## Optimised Matrix Multiplication Approaches and Sparse Matrices

This package contains the complete empirical study for Assignment 2 of the
Big Data course (ULPGC, 2026–2027):

- **Part A — dense optimisation:** the validated Assignment 1 triple-loop
  baseline (`i-j-k`) is compared against two distinct optimised kernels
  (`i-k-j` loop reordering and cache blocking) under an identical protocol.
- **Part B — sparse matrices:** a custom CSR implementation (Compressed
  Sparse Row) with Gustavson's sparse-by-sparse product is compared against
  the best dense method on *identical logical inputs* at several declared
  density levels.

Every kernel is implemented in **C** (single-threaded) so that the comparison
is intra-language and free of runtime confounders. The analysis pipeline
(statistics, tables, figures) is written in Python.

## Project structure

```text
assignment2/
├── src/c/                  # C kernels, sparse product, harness, validation
│   ├── a2_config.h         # experiment constants (sizes, budgets, tolerances)
│   ├── a2_timer.[ch]       # monotonic timer + peak-RSS sampler thread
│   ├── a2_matrix.[ch]      # dense matrices, CSV I/O, deterministic RNG
│   ├── a2_csr.[ch]         # CSR representation and conversions (student code)
│   ├── a2_kernels.[ch]     # dense kernels + method registry
│   ├── a2_sparse.c         # Gustavson SpGEMM (student code)
│   ├── a2_validate.[ch]    # correctness gate (runs before any benchmark)
│   └── a2_bench.c          # benchmark harness / CLI
├── scripts/
│   ├── generate_matrices.py    # regenerates the reused Assignment 1 matrices
│   ├── collect_system_info.py  # system fingerprint -> report/environment.md
│   └── analyze_results.py      # raw CSVs -> statistics, LaTeX tables, figures
├── data/
│   ├── matrices/           # optional local copy of A_n.csv / B_n.csv
│   └── results/            # raw measurements (CSV) + cross-check logs
├── report/                 # LaTeX report, generated figures and tables
└── README.md               # this file
```

## Requirements

- Windows 10/11 with **MSYS2 GCC** (developed with GCC 16.2.0) or any
  GCC/Clang with Windows headers (`psapi`).
- **CPython 3.11+** with `pandas`, `matplotlib`, `numpy` (analysis only).
- No external C libraries: the CSR structures and the sparse product are
  custom student code.

## Build (from the repository root)

```powershell
gcc -O2 -march=native -Wall -Wextra -o assignment2\src\c\a2_bench.exe `
    assignment2\src\c\a2_timer.c assignment2\src\c\a2_matrix.c `
    assignment2\src\c\a2_csr.c assignment2\src\c\a2_kernels.c `
    assignment2\src\c\a2_sparse.c assignment2\src\c\a2_validate.c `
    assignment2\src\c\a2_bench.c -lpsapi
```

All benchmark paths are relative to the repository root, so **every command
must be launched from the repository root** (`MatrixMultiplicationBigData/`).

## Inputs: reused Assignment 1 matrices + self-contained regeneration

Part A reuses the dense CSV matrices validated in Assignment 1
(`assignment1/data/matrices/A_n.csv`, `B_n.csv`; float64, uniform in
$[-10, 10]$, seed 42). The harness searches, in order:

1. `--matrices-dir <dir>` (explicit override),
2. `assignment2/data/matrices/` (optional local copy),
3. `assignment1/data/matrices/` (reused dataset).

To make this submission self-contained, `scripts/generate_matrices.py` is
the *identical* generator used in Assignment 1 (same Mersenne Twister stream,
seed, range and size order); running it recreates bit-identical CSV files in
`assignment2/data/matrices/`:

```powershell
python assignment2\scripts\generate_matrices.py
```

Part B matrices never touch the disk: they are generated deterministically
inside the harness (see “Experimental configuration”).

## Run

```powershell
# 1. Correctness gate (always runs before any measurement; also standalone):
assignment2\src\c\a2_bench.exe --validate-only

# 2. Full experiment (Part A + Part B), ~20-25 min on the reference machine:
assignment2\src\c\a2_bench.exe --part all

# 2a. Parts separately (useful while iterating):
assignment2\src\c\a2_bench.exe --part a
assignment2\src\c\a2_bench.exe --part b

# 3. Budget-boundary attempt (single baseline run at n = 4000, expected TIMEOUT):
assignment2\src\c\a2_bench.exe --part a --sizes 4000 --methods dense_ijk_baseline

# 4. Analysis: statistics, LaTeX tables and figures:
python assignment2\scripts\analyze_results.py
```

Useful flags: `--sizes`, `--sparse-sizes`, `--densities` (fractions),
`--methods dense_ijk_baseline,dense_ikj_reorder,dense_blocked`, `--reps`,
`--warmups`, `--budget <sec>`, `--block <b>`, `--out <csv>`,
`--dump-result <n> <path>` (writes the baseline product `C` to CSV for the
external NumPy cross-check), `--list-methods`, `--help`.

## Experimental configuration

| Parameter | Value |
|-----------|-------|
| Numeric type | `float64` (`double`), row-major contiguous storage |
| Dense inputs (Part A) | Reused Assignment 1 matrices, seed 42, uniform $[-10, 10]$ |
| Sparse inputs (Part B) | Generated in-harness: each entry is exactly `0.0` with probability $1-d$, otherwise uniform $[-10, 10]$; 32-bit LCG ($s \leftarrow 1103515245s + 12345$), stream restarted from seed 42 per configuration (zero patterns are nested across densities at fixed $n$) |
| Part A sizes | $n \in \{10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000\}$ |
| Part B sizes | $n \in \{200, 500, 1000, 2000\}$, plus $n = 3000$ for $d \in \{10\%, 1\%, 0.1\%\}$ (budget control) |
| Part B densities | $d \in \{50\%, 20\%, 10\%, 5\%, 1\%, 0.1\%\}$ (sparsity $= 100\% - d$) |
| Dense methods | `dense_ijk_baseline` (validated Assignment 1 baseline), `dense_ikj_reorder`, `dense_blocked` (block $b = 512$, tuned over $\{16..1024\}$) |
| Sparse method | `sparse_csr_gustavson` (custom CSR, Gustavson row-wise SpGEMM, sorted output rows) |
| Warm-up runs | 1 per configuration (CSV `repetition_id = 0`, excluded from statistics) |
| Measured repetitions | 5 per configuration |
| Per-run budget | 120 s watchdog; aborted runs are recorded as `TIMEOUT` **without** an invented time |
| Batching | factor 100 for $n \leq 50$ (time divided by the batch factor) |
| Timing boundary | Kernel only: inputs loaded/generated outside the timer; output allocated outside, initialised inside by the kernel (documented); dense→CSR conversion timed **separately** (`dense_to_csr_conversion`) |
| Timer | `clock_gettime(CLOCK_MONOTONIC)` |
| Memory metric | Peak process working set (RSS) sampled every 5 ms (`GetProcessMemoryInfo`); plus the analytic storage of each representation (`theoretical_storage_mb`) |
| Anti-DCE | `trace(C)` computed after every run and stored in the `checksum` column |
| Compiler flags | `-O2 -march=native -Wall -Wextra`, no OpenMP, 1 thread |
| Tolerances | $\epsilon_{\text{abs}} = \epsilon_{\text{rel}} = 10^{-9}$ (small tests); cross-method checks at benchmark sizes use $\epsilon_{\text{abs}} = 10^{-6}$ (worst-case $n\,\epsilon\,|C|_{\max} < 10^{-6}$ for every tested $n$) |

## Raw data schema

Every measurement row in `data/results/results_a2_*.csv`:

```text
assignment,method,language,n,numeric_type,seed,warmup_runs,repetition_id,
threads_or_workers,density,execution_time_ms,memory_metric_name,
memory_value_mb,checksum,status,nnz_A,nnz_B,nnz_C,block_size,
theoretical_storage_mb
```

- `repetition_id = 0` marks warm-up runs (kept for transparency, excluded
  from medians/IQR); `1..5` are the measured repetitions.
- `density` is the percentage of non-zero entries of the logical inputs
  (100 for Part A; the target $d$ for Part B — the realised counts are in
  `nnz_A` / `nnz_B`).
- `status` is `OK` or `TIMEOUT` (a run aborted by the watchdog keeps an
  empty `execution_time_ms`).
- `data/results/crosscheck_log_*.csv` records the element-wise agreement
  between methods on the real inputs (maximum absolute difference vs.
  tolerance), which is the correctness evidence at benchmark scale.

## Correctness

`--validate-only` runs before every benchmark and aborts it on failure:
identity and zero matrices, a known $2\times2$ product, every dense kernel
against an independent $k$-$i$-$j$ reference at odd and block-multiple sizes,
CSR round-trip losslessness, the Gustavson product against the dense baseline
at four densities, and sparse identity/zero products. At benchmark scale the
harness additionally compares the full result matrices of all executed
methods (`crosscheck_log_*.csv`), and `--dump-result` enables an external
NumPy cross-check (`scripts/verify_numpy.py`).

## Declared budgets and limits

- **Memory budget:** 8 GiB (the largest tested configuration, $n = 3000$
  dense, needs ≈ 216 MiB of matrix storage; no OOM was observed).
- **Time budget:** 120 s per measured run (user-declared cap). The largest
  *tested* size is $n = 3000$ for every Part A method; the largest
  *attempted* size is $n = 4000$ with the baseline kernel, which exceeds the
  cap (≈ 294 s in Assignment 1) and is recorded as a `TIMEOUT` outcome.

## Reproducing the report

```powershell
python assignment2\scripts\collect_system_info.py   # system fingerprint
python assignment2\scripts\analyze_results.py       # stats + tables + figures
```

`report/Assignment02_Santana_Report.tex` consumes the generated
`report/tables/*.tex` and `report/figures/*.png`; compile it with any LaTeX
distribution (tested on Overleaf).

## Reused materials (Assignment 1)

- **Dataset:** `assignment1/data/matrices/A_n.csv`, `B_n.csv` (seed 42,
  float64, uniform $[-10, 10]$); regenerable with the bundled generator.
- **Baseline kernel:** `dense_ijk_baseline` reproduces the validated
  Assignment 1 C triple loop (`assignment1/src/c/matrix_mult.c`) with the
  same storage layout, compiler flags and timer.
