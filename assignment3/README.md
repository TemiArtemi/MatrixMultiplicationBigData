# Matrix Multiplication Big Data — Assignment 3

## Parallel Matrix Multiplication and Optional Vectorisation

This package contains the empirical study for Assignment 3 of the Big Data
course (ULPGC, 2026–2027):

- **Research question:** how does the thread count $p$ affect the kernel time,
  speedup and efficiency of a shared-memory dense matrix multiplication, and
  when does parallel execution help?
- **Parallel implementation:** the validated Assignment 2 cache-blocked
  (tiled i-k-j) kernel, parallelised with **OpenMP** over row tiles.
- **Equivalent sequential baseline:** the *same source file* compiled without
  `-fopenmp` (the pragmas are ignored), so algorithm, data layout, numeric
  type, inputs and accumulation order are identical by construction.
- **Optional SIMD extension (included):** the same source compiled with and
  without the GCC tree vectoriser, measured at $p=1$ and at one fixed $p$.

## Project structure

```text
assignment3/
├── src/c/
│   ├── a3_config.h        # experiment constants (budgets, tolerances, tiles)
│   ├── a3_timer.[ch]      # monotonic timer + peak-RSS sampler (reused from A2)
│   ├── a3_matrix.[ch]     # row-major dense matrices, CSV I/O (reused from A2)
│   ├── a3_kernels.[ch]    # tiled kernel, compiled 4x (seq/par x vec/novec)
│   ├── a3_registry.c      # registry of the compiled variants
│   ├── a3_validate.[ch]   # correctness gate (runs before every benchmark)
│   ├── a3_sysinfo.[ch]    # P/E-core topology, OpenMP settings, SIMD macros
│   ├── a3_affinity.[ch]   # EcoQoS opt-out and P-core affinity (Windows API)
│   └── a3_bench.c         # benchmark harness / CLI
├── scripts/
│   ├── build.ps1               # builds the harness + SIMD evidence files
│   ├── generate_matrices.py    # regenerates the reused Assignment 1 matrices
│   ├── verify_numpy.py         # external NumPy cross-check of a parallel result
│   ├── collect_system_info.py  # system fingerprint -> report/environment.md
│   └── analyze_results.py      # raw CSVs -> statistics, LaTeX tables, figures
├── data/
│   ├── matrices/          # optional local copy of A_n.csv / B_n.csv
│   └── results/           # raw CSVs, logs, vectorisation reports
├── report/                # LaTeX report, generated tables and figures
└── README.md
```

## Requirements

- Windows 10/11 with **MSYS2 GCC** (developed with GCC 16.2.0, libgomp,
  OpenMP 5.2) and `objdump` (binutils) on the `PATH`.
- **CPython 3.11+** with `pandas`, `matplotlib`, `numpy` (analysis and
  cross-check only).
- No numerical library is linked by the C code, so no hidden library threads
  exist; the harness records `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` anyway.

## Build (from the repository root)

```powershell
powershell -ExecutionPolicy Bypass -File assignment3\scripts\build.ps1
```

The script compiles `a3_kernels.c` four times:

| Object | Flags |
| --- | --- |
| `tiled_seq_vec` (sequential baseline) | `-O3 -march=native` |
| `tiled_par_vec` (parallel) | `-O3 -march=native -fopenmp` |
| `tiled_seq_novec` | `-O3 -march=native -fno-tree-vectorize` |
| `tiled_par_novec` | `-O3 -march=native -fno-tree-vectorize -fopenmp` |

and writes the SIMD evidence: `data/results/vec_report_*.txt`
(`-fopt-info-vec-optimized`), `data/results/simd_evidence.csv` (packed `ymm`
and scalar FMA instruction counts from `objdump`) and
`data/results/objdump_excerpt_par_vec.txt`.

## Inputs

The study reuses the Assignment 1 dense matrices
(`assignment1/data/matrices/A_n.csv`, `B_n.csv`; float64, uniform in
$[-10, 10]$, Mersenne Twister seed 42). The harness searches
`assignment3/data/matrices/` first and then the Assignment 1 folder.
`scripts/generate_matrices.py` is the identical generator and recreates
bit-identical files:

```powershell
python assignment3\scripts\generate_matrices.py
```

## Run (from the repository root)

```powershell
# 1. Correctness gate only (all variants, p in {1, 2, 4, 28, 56})
assignment3\src\c\a3_bench.exe --validate-only

# 2. External NumPy cross-check of a parallel result (n = 1000, p = 28)
assignment3\src\c\a3_bench.exe --dump-result 1000 28 C_1000.csv
python assignment3\scripts\verify_numpy.py --n 1000 --dump C_1000.csv

# 3. Row-tile tuning sweep (run once; fixed the default row tile to 16)
assignment3\src\c\a3_bench.exe --experiment tune --sizes 2000 --threads 1 28 `
    --row-tiles 8 16 32 64 --reps 3 --out assignment3\data\results\results_a3_tune.csv

# 4. Main thread-scaling experiment
assignment3\src\c\a3_bench.exe --experiment threads `
    --out assignment3\data\results\results_a3_threads.csv

# 5. SIMD extension (parallel runs at the p chosen from step 4)
assignment3\src\c\a3_bench.exe --experiment simd --simd-threads <p> `
    --out assignment3\data\results\results_a3_simd.csv

# 6. Fingerprint, statistics, tables and figures
python assignment3\scripts\collect_system_info.py
python assignment3\scripts\analyze_results.py
```

Default configuration: $n \in \{400, 1000, 2000, 3000, 4000\}$,
$p \in \{1, 2, 4, 8, 12, 16, 20, 24, 28, 56\}$, row tile 16, k/j tile 512,
1 warm-up + 5 measured repetitions, 120 s per-run watchdog.

## Measurement protocol

- **Timer:** `clock_gettime(CLOCK_MONOTONIC)` around the kernel call only.
  Inputs are loaded and $C$ is allocated outside the timer; zeroing $C$
  (done per row tile by the owning thread) is inside it.
- **Warm-up:** 1 run per configuration (`repetition_id = 0`, kept in the CSV,
  excluded from statistics). It also absorbs OpenMP worker-thread creation.
- **Anti-dead-code-elimination:** the trace of $C$ is logged per repetition.
- **Memory:** peak process working set (RSS) sampled every 5 ms by a
  background thread (`GetProcessMemoryInfo`).
- **Budget:** 120 s per run; overruns are logged as `TIMEOUT` with an empty
  time. Memory: $3 \cdot 8n^2$ bytes ≈ 366 MiB at $n = 4000$.
- **Correctness:** identity / zero / known 2x2 / independent k-i-j reference
  (tolerances $10^{-9}$ absolute and relative), plus **bitwise equality** of
  every parallel result with the sequential result; logged per
  configuration in `data/results/crosscheck_log_*.csv`.
- **Threads:** `omp_set_dynamic(0)` and `omp_set_num_threads(p)` per
  configuration. Thread affinity (`OMP_PROC_BIND`/`OMP_PLACES`) is **not
  supported by libgomp on this platform** (it prints *"Affinity not supported
  on this configuration"*), so thread placement on P-cores and E-cores is left
  to the Windows scheduler; this is a declared limitation.
- **Placement policy (Windows API, `a3_affinity.c`):** the harness opts the
  process out of EcoQoS power throttling (`SetProcessInformation`,
  `ProcessPowerThrottling`) and restricts every **1-thread** run (sequential
  baseline and parallel $p=1$) to the 16 logical processors of the P-cores
  (`SetProcessAffinityMask`); runs with $p \ge 2$ may use all 28 logical
  processors. Without this, Windows placed some 1-thread runs on E-cores and
  inflated the speedups: the evidence is in `data/results/placement_diagnostic/`
  and the biased first run is kept in `data/results/superseded/`. Use
  `--no-ecoqos-optout` / `--single-thread-any-core` to reproduce it.

## Output CSV columns

`assignment, method, language, n, numeric_type, seed, warmup_runs,
repetition_id, threads_or_workers, density, execution_time_ms,
memory_metric_name, memory_value_mb, checksum, status, variant, simd,
schedule, row_tile, block_size, experiment, theoretical_storage_mb, placement`

(The archived files in `data/results/superseded/` predate the `placement`
column.)
