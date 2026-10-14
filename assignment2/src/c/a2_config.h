/**
 * a2_config.h - Experiment-wide configuration constants for Assignment 2.
 *
 * All tunable parameters of the controlled experiment live here so that the
 * kernels, the harness and the validation suite share a single source of truth.
 * The rationale of every value is documented in assignment2/README.md.
 */

#ifndef A2_CONFIG_H
#define A2_CONFIG_H

/* ---- Input generation (dense matrices reused from Assignment 1) ---- */
#define A2_SEED         42u     /* same seed as the Assignment 1 generator */
#define A2_VALUE_MIN    (-10.0)
#define A2_VALUE_MAX    10.0

/* ---- Measurement protocol ---- */
#define A2_WARMUP_RUNS      1   /* warm-up runs, excluded from statistics   */
#define A2_MEASURED_REPS    5   /* measured repetitions per configuration   */
#define A2_RUN_BUDGET_SEC   120.0 /* per-run watchdog (seconds)             */
#define A2_TINY_BATCH       100 /* batch factor for very small matrices     */
#define A2_TINY_MAX_N       50  /* n <= this value uses batched execution   */

/* ---- Floating-point comparison tolerances ------------------------------
 * Small validation tests use the tight tolerances below. Cross-method
 * checks at benchmark sizes use A2_CROSS_ABS_TOL because the worst-case
 * rounding error of an O(n) dot-product accumulation grows linearly with n
 * (n * eps * max|C| is below 1e-6 for every tested size; see report). */
#define A2_ABS_TOL          1e-9
#define A2_REL_TOL          1e-9
#define A2_CROSS_ABS_TOL    1e-6

/* ---- Dense cache blocking ------------------------------------------------
 * g_block_size can be overridden from the command line (--block). The
 * default of 512 was chosen with a tuning sweep over {16, 32, 64, 128, 256,
 * 512, 1024} at n in {1500, 2000, 3000} (median of 3 repetitions): 512 was
 * the fastest or near-fastest at every large size (see the tuning table in
 * the report). For n below the block size the kernel degenerates gracefully
 * to plain i-k-j, so the same value is harmless on small matrices. */
#define A2_DEFAULT_BLOCK    512

/* ---- Default data locations (relative to the repository root) ----------
 * Part A reuses the Assignment 1 CSV matrices when present; otherwise the
 * bundled generator (scripts/generate_matrices.py) can regenerate identical
 * files into the Assignment 2 data directory. */
#define A2_MATRICES_DIR_A1  "assignment1\\data\\matrices"
#define A2_MATRICES_DIR_A2  "assignment2\\data\\matrices"
#define A2_RESULTS_DIR      "assignment2\\data\\results"

/* Externally settable block size (see a2_kernels.c) */
extern int g_block_size;

#endif /* A2_CONFIG_H */
