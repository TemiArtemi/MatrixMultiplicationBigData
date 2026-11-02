/**
 * a3_config.h - Experiment-wide configuration constants for Assignment 3.
 *
 * All tunable parameters of the controlled experiment live here so that the
 * kernels, the harness and the validation suite share a single source of truth.
 * The rationale of every value is documented in assignment3/README.md.
 */

#ifndef A3_CONFIG_H
#define A3_CONFIG_H

/* ---- Input generation (dense matrices reused from Assignment 1) ---- */
#define A3_SEED         42u     /* same seed as the Assignment 1 generator */
#define A3_VALUE_MIN    (-10.0)
#define A3_VALUE_MAX    10.0

/* ---- Measurement protocol ---- */
#define A3_WARMUP_RUNS      1   /* warm-up runs, excluded from statistics;
                                   also absorbs OpenMP thread-pool creation */
#define A3_MEASURED_REPS    5   /* measured repetitions per configuration   */
#define A3_RUN_BUDGET_SEC   120.0 /* per-run watchdog (seconds)             */

/* ---- Floating-point comparison tolerances (same values as Assignment 2) */
#define A3_ABS_TOL          1e-9
#define A3_REL_TOL          1e-9
#define A3_CROSS_ABS_TOL    1e-6

/* ---- Tiling ---------------------------------------------------------------
 * Rectangular tiles: BI rows (the unit of parallel work) x BKJ columns of
 * the k and j dimensions. BKJ = 512 is the value tuned in Assignment 2; BI
 * is chosen with a short tuning sweep (see README and report). */
#define A3_DEFAULT_ROW_TILE 16
#define A3_DEFAULT_BLOCK    512

/* ---- Default data locations (relative to the repository root) -------- */
#define A3_MATRICES_DIR_A1  "assignment1\\data\\matrices"
#define A3_MATRICES_DIR_A3  "assignment3\\data\\matrices"
#define A3_RESULTS_DIR      "assignment3\\data\\results"

#endif /* A3_CONFIG_H */
