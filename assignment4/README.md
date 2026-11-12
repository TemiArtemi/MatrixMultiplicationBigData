# Matrix Multiplication Big Data — Assignment 4

## Distributed Matrix Multiplication (C + MPI, local execution)

This package contains the empirical study for Assignment 4 of the Big Data
course (ULPGC, 2026–2027):

- **Research question:** how does a distributed dense matrix multiplication
  scale with the number of workers (strong scaling), and how much of the
  end-to-end time is computation, data movement and framework overhead?
- **Framework:** **MPI** (Microsoft MPI, `mpiexec`), C. One single-threaded
  process per worker.
- **Partitioning:** 1D **row-block** with the whole of $B$ broadcast. Rank 0
  owns $A$, $B$ and $C$; `MPI_Scatterv` sends each rank a block of rows of
  $A$, `MPI_Bcast` sends $B$ to every rank, each rank multiplies its rows
  with the Assignment 3 tiled kernel (16-row tiles x 512, single thread,
  `-O3 -march=native`), and `MPI_Gatherv` collects the rows of $C$ on rank 0.
  The $n \bmod p$ remainder rows go to the first ranks; ranks that own no row
  (p > n) take part in every collective with empty blocks.
- **Baselines (same session):** the Assignment 3 sequential kernel and the
  Assignment 3 OpenMP kernel at the same sizes and worker counts, measured by
  `assignment3\src\c\a3_bench.exe` right before the MPI runs.

> **Local Execution Limitation.** The instructor allowed running the
> assignment locally. All ranks therefore run on **one laptop** (28 logical
> processors, one RAM, one L3 cache) and exchange data through MS-MPI's
> shared-memory transport. The study measures process, copy and framework
> overhead; it does **not** measure inter-machine network bandwidth or
> multi-machine memory capacity. Workers are processes, not machines.

## Project structure

```text
assignment4/
├── src/c/
│   ├── a4_config.h        # experiment constants (budgets, tolerances, tiles)
│   ├── a4_timer.[ch]      # monotonic timer + peak-RSS sampler (from A3)
│   ├── a4_matrix.[ch]     # row-major dense matrices, CSV I/O (from A3)
│   ├── a4_affinity.[ch]   # EcoQoS opt-out and P-core affinity (from A3)
│   ├── a4_kernels.[ch]    # MPI-free tiled kernel on a row block
│   ├── a4_partition.[ch]  # Partitioner interface + row-block strategy
│   ├── a4_validate.[ch]   # correctness gate (runs on every launch)
│   ├── a4_sysinfo.[ch]    # P/E-core topology, MPI version, SIMD macros
│   └── a4_bench.c         # MPI benchmark harness / CLI
├── scripts/
│   ├── build.ps1               # builds libmsmpi.a (once) and the harness
│   ├── run_experiments.ps1     # baselines + MPI start-up + scaling runs
│   ├── generate_matrices.py    # regenerates the reused Assignment 1 matrices
│   ├── verify_numpy.py         # external NumPy cross-check of an MPI result
│   ├── collect_system_info.py  # system fingerprint -> report/environment.md
│   └── analyze_results.py      # raw CSVs -> statistics, LaTeX tables, figures
├── data/
│   ├── matrices/          # optional local copy of A_n.csv / B_n.csv
│   └── results/           # raw CSVs, logs, vectorisation report
├── report/                # LaTeX report, generated tables and figures
└── README.md
```

## Requirements

- Windows 10/11 with **MSYS2 GCC** (developed with GCC 16.2.0) including
  `gendef` and `dlltool` (package `mingw-w64-ucrt-x86_64-tools`).
- **Microsoft MPI runtime and SDK** (10.1):

  ```powershell
  winget install --id Microsoft.msmpi --exact
  winget install --id Microsoft.msmpisdk --exact
  ```

- **CPython 3.11+** with `pandas`, `matplotlib`, `numpy` (analysis only).
- The Assignment 3 harness built (`assignment3\scripts\build.ps1`), used for
  the same-session baselines.
- No numerical library is linked and OpenMP is not used by the distributed
  code, so the worker count is exactly the number of MPI ranks.

## Build (from the repository root)

```powershell
powershell -ExecutionPolicy Bypass -File assignment4\scripts\build.ps1
```

GCC cannot link the Visual-Studio import library `msmpi.lib`, so the script
generates a MinGW import library `assignment4\build\libmsmpi.a` once from
`msmpi.dll` (`gendef` + `dlltool`) and compiles with the SDK include paths.
The kernel is compiled with `-O3 -march=native` (the vectorisation report is
written to `data/results/vec_report_kernel.txt`), everything else with `-O2`.

## Inputs

The Assignment 1 dense matrices are reused (`assignment1/data/matrices/A_n.csv`,
`B_n.csv`; float64, uniform in $[-10, 10]$, Mersenne Twister seed 42). The
harness searches `assignment4/data/matrices/` first and then the Assignment 1
folder. `scripts/generate_matrices.py` is the identical generator and
recreates bit-identical files:

```powershell
python assignment4\scripts\generate_matrices.py
```

## Run (from the repository root)

```powershell
$env:Path += ";C:\Program Files\Microsoft MPI\Bin;C:\msys64\ucrt64\bin"

# 1. Correctness gate only, for several rank counts (incl. p > n for small n)
foreach ($p in 1,2,3,4,7,8,16,28) { mpiexec -n $p assignment4\src\c\a4_bench.exe --validate-only }

# 2. External NumPy cross-check of a 28-rank result (n = 1000)
mpiexec -n 28 assignment4\src\c\a4_bench.exe --dump-result 1000 C_1000.csv
python assignment4\scripts\verify_numpy.py --n 1000 --dump C_1000.csv

# 3. Full measurement session (baselines, MPI start-up, scaling runs)
.\assignment4\scripts\run_experiments.ps1

# 4. Fingerprint, statistics, tables and figures
python assignment4\scripts\collect_system_info.py
python assignment4\scripts\analyze_results.py
```

Call `run_experiments.ps1` directly (not with `powershell -File`), because
`-File` mangles comma-separated array parameters such as `-Workers 1,2,4`.

Default configuration: $n \in \{1000, 2000, 3000, 4000\}$,
$p \in \{1, 2, 4, 8, 16, 28\}$, 1 warm-up + 5 measured repetitions,
120 s per-run watchdog.

## Measurement protocol

- **End-to-end boundary:** from *$A$ and $B$ in rank-0 memory* to *$C$ in
  rank-0 memory* (scatter + broadcast + compute + gather), measured with
  `clock_gettime(CLOCK_MONOTONIC)` on rank 0 between barriers. CSV loading
  (`t_load_ms` column) and MPI start-up (`results_a4_startup.csv`, launcher
  wall clock of `mpiexec` for a program that only initialises and finalises
  MPI) are outside the boundary and reported separately.
- **Phases:** `t_scatter`, `t_bcast`, `t_compute` (slowest rank, plus a
  4-byte completion agreement), `t_gather`. They are consecutive intervals of
  one clock, so they add up to the total; the barriers remove overlap between
  phases, which is a property of the instrumentation, not of the framework.
  The fastest and slowest rank's kernel time (`t_kernel_min/max`) are
  gathered after the boundary and expose load imbalance.
- **Buffers:** per-rank buffers are allocated and touched outside the timer.
- **Warm-up:** 1 run per configuration (`repetition_id = 0`, kept in the CSV,
  excluded from statistics).
- **Anti-dead-code-elimination:** the trace of $C$ is logged per repetition.
- **Memory:** peak process working set (resident memory) of every rank,
  sampled every 5 ms (`GetProcessMemoryInfo`); `memory_value_mb` is the sum
  over ranks, `mem_rank0_mb` and `mem_max_nonroot_mb` are given separately.
  Rank 0 additionally holds the harness' reference result.
- **Budget:** 120 s per run; overruns are logged as `TIMEOUT` with empty
  times (collective decision, no invented value).
- **Correctness:** identity / zero / known 2x2 / independent k-i-j reference
  (tolerances $10^{-9}$ absolute and relative) and **bitwise equality** with
  the single-block result of the same kernel, for every rank count of the
  gate; plus a per-configuration bitwise check during the benchmark
  (`*_crosscheck.csv`) and an external NumPy check.
- **Placement (Windows API, `a4_affinity.c`):** every rank opts out of EcoQoS
  power throttling; the single-rank run is restricted to the P-cores; runs
  with $p \ge 2$ are scheduled by Windows over all 28 logical processors.
  Assignment 3 showed that without this the OS can place single-thread runs
  on E-cores and distort speedups.

## Output CSV columns

`assignment, method, language, n, numeric_type, seed, warmup_runs,
repetition_id, threads_or_workers, density, execution_time_ms,
memory_metric_name, memory_value_mb, checksum, status, partitioner, row_tile,
block_size, t_scatter_ms, t_bcast_ms, t_compute_ms, t_gather_ms,
t_kernel_min_ms, t_kernel_max_ms, mem_rank0_mb, mem_max_nonroot_mb,
placement, experiment, theoretical_storage_mb, t_load_ms`

`execution_time_ms` is the end-to-end time; `threads_or_workers` is the
number of MPI ranks.
