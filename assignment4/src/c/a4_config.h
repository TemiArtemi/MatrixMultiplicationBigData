/**
 * a4_config.h - Experiment-wide configuration constants for Assignment 4.
 *
 * All tunable parameters of the controlled experiment live here so that the
 * kernel, the partitioner, the harness and the validation suite share a
 * single source of truth. The rationale of every value is documented in
 * assignment4/README.md.
 */

#ifndef A4_CONFIG_H
#define A4_CONFIG_H

/* ---- Input generation (dense matrices reused from Assignment 1) ---- */
#define A4_SEED         42u     /* same seed as the Assignment 1 generator */
#define A4_VALUE_MIN    (-10.0)
#define A4_VALUE_MAX    10.0

/* ---- Measurement protocol (identical to Assignments 2 and 3) ---- */
#define A4_WARMUP_RUNS      1   /* warm-up runs, excluded from statistics  */
#define A4_MEASURED_REPS    5   /* measured repetitions per configuration  */
#define A4_RUN_BUDGET_SEC   120.0 /* per-run watchdog (seconds)            */

/* ---- Floating-point comparison tolerances (same values as Assignment 3) */
#define A4_ABS_TOL          1e-9
#define A4_REL_TOL          1e-9
#define A4_CROSS_ABS_TOL    1e-6

/* ---- Local kernel tiling --------------------------------------------------
 * Same tile shape as the Assignment 3 parallel runs (16 rows x 512), so the
 * per-rank compute phase is directly comparable with the OpenMP kernel. */
#define A4_DEFAULT_ROW_TILE 16
#define A4_DEFAULT_BLOCK    512

/* ---- Default data locations (relative to the repository root) -------- */
#define A4_MATRICES_DIR_A1  "assignment1\\data\\matrices"
#define A4_MATRICES_DIR_A4  "assignment4\\data\\matrices"
#define A4_RESULTS_DIR      "assignment4\\data\\results"

#endif /* A4_CONFIG_H */
