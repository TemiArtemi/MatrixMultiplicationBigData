/**
 * a3_bench.c - Benchmark harness for Assignment 3.
 *
 * Experiments:
 *   threads : sequential baseline (tiled_seq_vec) versus the parallel
 *             implementation (tiled_par_vec) for every thread count p,
 *             including p = 1, on the reused Assignment 1 matrices.
 *   simd    : optional extension; vectorised versus non-vectorised builds
 *             of the same source, at p = 1 (sequential) and at one fixed
 *             thread count, so vectorisation is isolated from threading.
 *   tune    : short sweep of the row-tile height used as the parallel
 *             work unit (run once, before the main experiment).
 *
 * Protocol enforced here (see assignment3/README.md for the rationale):
 *   - correctness gate runs first and aborts the harness on failure;
 *   - inputs are loaded OUTSIDE the timed region; the output matrix is
 *     allocated outside the timer and zeroed by the kernel inside it;
 *   - g_warmups warm-up runs (repetition_id = 0, kept in the CSV but
 *     excluded from statistics) plus g_reps measured repetitions; the
 *     warm-up also absorbs the creation of new OpenMP worker threads;
 *   - a monotonic timer (CLOCK_MONOTONIC) and an anti-dead-code-elimination
 *     checksum (trace of C) per repetition;
 *   - a per-run watchdog of --budget seconds aborts a run and records a
 *     TIMEOUT row with no invented time;
 *   - after each configuration the result is compared with the sequential
 *     result of the same build (bitwise) and logged.
 */

#include "a3_config.h"
#include "a3_timer.h"
#include "a3_matrix.h"
#include "a3_kernels.h"
#include "a3_validate.h"
#include "a3_sysinfo.h"

#include <windows.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define A3_MAX_LIST 32

/* ---- Experiment defaults (overridable from the command line) ----------- */
static int g_sizes[A3_MAX_LIST] = {400, 1000, 2000, 3000, 4000};
static int g_sizes_count = 5;

static int g_threads[A3_MAX_LIST] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 56};
static int g_threads_count = 10;

/* Thread counts exercised by the correctness gate: 1, small counts, the
 * logical-processor count and an oversubscribed count. */
static int g_validate_threads[A3_MAX_LIST] = {1, 2, 4, 28, 56};
static int g_validate_threads_count = 5;

static int g_simd_sizes[A3_MAX_LIST] = {1000, 2000, 3000};
static int g_simd_sizes_count = 3;
static int g_simd_threads = 28;

static int g_row_tiles[A3_MAX_LIST] = {8, 16, 32, 64};
static int g_row_tiles_count = 4;

static int g_row_tile = A3_DEFAULT_ROW_TILE;
static int g_block = A3_DEFAULT_BLOCK;
static int g_reps = A3_MEASURED_REPS;
static int g_warmups = A3_WARMUP_RUNS;
static double g_budget = A3_RUN_BUDGET_SEC;

/* ---- CSV helpers -------------------------------------------------------- */

static const char *MEM_METRIC = "peak_rss_working_set_mb";

static void csv_header(FILE *f) {
    fprintf(f, "assignment,method,language,n,numeric_type,seed,warmup_runs,"
               "repetition_id,threads_or_workers,density,execution_time_ms,"
               "memory_metric_name,memory_value_mb,checksum,status,variant,"
               "simd,schedule,row_tile,block_size,experiment,"
               "theoretical_storage_mb\n");
}

static void csv_row(FILE *f, const char *experiment, const KernelVariant *v,
                    int n, int rep_id, int p, double time_ms, double peak_mb,
                    double checksum, const char *status, int row_tile,
                    int block) {
    double storage_mb = 3.0 * 8.0 * (double)n * (double)n / (1024.0 * 1024.0);
    fprintf(f, "3,%s,C,%d,float64,%u,%d,%d,%d,100.0000,", v->name, n, A3_SEED,
            g_warmups, rep_id, p);
    if (strcmp(status, "OK") == 0) {
        fprintf(f, "%.4f", time_ms);
    }
    /* TIMEOUT: execution_time_ms stays empty (no invented time). */
    fprintf(f, ",%s,%.3f,%.6e,%s,%s,%s,%s,%d,%d,%s,%.4f\n", MEM_METRIC,
            peak_mb, checksum, status, v->variant, v->simd,
            v->threaded ? "dynamic_1" : "none", row_tile, block, experiment,
            storage_mb);
    fflush(f);
}

/* ---- Cross-check log ----------------------------------------------------- */

static FILE *g_cross_file = NULL;

static void cross_log(const char *experiment, const char *a, int pa,
                      const char *b, int n, const DenseMatrix *Ca,
                      const DenseMatrix *Cb) {
    int ok = 0;
    double diff = dm_max_abs_diff(Ca, Cb, &ok, A3_CROSS_ABS_TOL, A3_REL_TOL);
    int bitwise = memcmp(Ca->data, Cb->data,
                         (size_t)n * (size_t)n * sizeof(double)) == 0;
    if (g_cross_file) {
        fprintf(g_cross_file, "3,%s,%s,%d,%s,%d,%.6e,%d,%s\n", experiment, a,
                pa, b, n, diff, bitwise, ok ? "OK" : "MISMATCH");
        fflush(g_cross_file);
    }
    printf("   cross-check %s (p=%d) vs %s: max |diff| = %.3e, bitwise %s [%s]\n",
           a, pa, b, diff, bitwise ? "yes" : "no", ok ? "OK" : "MISMATCH");
}

/* ---- Matrix resolution --------------------------------------------------- */

static const char *g_matrix_dirs[3];
static int g_matrix_dirs_count = 0;

static DenseMatrix *load_ab(int n, char which) {
    char path[512];
    for (int d = 0; d < g_matrix_dirs_count; d++) {
        snprintf(path, sizeof(path), "%s\\%c_%d.csv", g_matrix_dirs[d], which, n);
        DenseMatrix *m = dm_load_csv(path);
        if (m) return m;
    }
    return NULL;
}

/* ---- Run primitives ------------------------------------------------------ */

typedef struct {
    int completed;
    double time_ms;
    double peak_mb;
} RunResult;

static RunResult run_once(const KernelVariant *v, const DenseMatrix *A,
                          const DenseMatrix *B, DenseMatrix *C, int row_tile,
                          int block) {
    RunResult r;
    MemSampler s;
    mem_sampler_start(&s);
    double deadline = a3_time_sec() + g_budget;
    double t0 = a3_time_sec();
    int ok = v->fn(A->data, B->data, C->data, A->n, row_tile, block, deadline);
    double t1 = a3_time_sec();
    r.peak_mb = mem_sampler_stop(&s);
    r.completed = ok;
    r.time_ms = (t1 - t0) * 1000.0;
    return r;
}

/* Warm-up + measured repetitions of one configuration. Returns 1 when every
 * run completed (C then holds a valid product). */
static int run_config(FILE *out, const char *experiment, const KernelVariant *v,
                      int p, const DenseMatrix *A, const DenseMatrix *B,
                      DenseMatrix *C, int row_tile, int block) {
    int n = A->n;
    int threads = v->threaded ? p : 1;
    omp_set_num_threads(threads);

    for (int rep = 0; rep <= g_reps; rep++) {
        if (rep == 0 && g_warmups == 0) continue;
        int runs = (rep == 0) ? g_warmups : 1;
        for (int w = 0; w < runs; w++) {
            RunResult r = run_once(v, A, B, C, row_tile, block);
            const char *status = r.completed ? "OK" : "TIMEOUT";
            csv_row(out, experiment, v, n, rep, threads, r.time_ms, r.peak_mb,
                    r.completed ? dm_trace(C) : 0.0, status, row_tile, block);
            if (rep == 0) {
                printf("   %-16s p=%-3d warm-up: %s (%.2f ms)\n", v->name,
                       threads, status, r.time_ms);
            } else {
                printf("   %-16s p=%-3d rep %d: %s (%.2f ms, peak %.1f MB)\n",
                       v->name, threads, rep, status, r.time_ms, r.peak_mb);
            }
            if (!r.completed) {
                printf("   %s exceeds the %.0f s per-run budget at n = %d "
                       "(p = %d); remaining repetitions skipped.\n",
                       v->name, g_budget, n, threads);
                return 0;
            }
        }
    }
    return 1;
}

/* Load A_n, B_n and allocate C and a reference buffer. Returns 0 (and
 * prints why) when the configuration must be skipped. */
static int prepare(int n, DenseMatrix **A, DenseMatrix **B, DenseMatrix **C,
                   DenseMatrix **R) {
    *A = load_ab(n, 'A');
    *B = load_ab(n, 'B');
    *C = (*A && *B) ? dm_alloc(n) : NULL;
    *R = *C ? dm_alloc(n) : NULL;
    if (!*A || !*B) {
        printf("   Matrices for n = %d not found; SKIPPING (run "
               "scripts/generate_matrices.py).\n", n);
    } else if (!*C || !*R) {
        printf("   Allocation failed (OOM) for n = %d; SKIPPING.\n", n);
    }
    if (*A && *B && *C && *R) return 1;
    dm_free(*A); dm_free(*B); dm_free(*C); dm_free(*R);
    return 0;
}

static void copy_matrix(DenseMatrix *dst, const DenseMatrix *src) {
    memcpy(dst->data, src->data, (size_t)src->n * (size_t)src->n * sizeof(double));
}

/* ---- Experiment: thread scaling ----------------------------------------- */

static void run_threads(FILE *out) {
    printf("\n############################################################\n");
    printf("THREAD SCALING (tiled_seq_vec baseline vs tiled_par_vec)\n");
    printf("############################################################\n");
    const KernelVariant *seqv = a3_find_variant("seq", "vec");
    const KernelVariant *parv = a3_find_variant("par", "vec");

    for (int si = 0; si < g_sizes_count; si++) {
        int n = g_sizes[si];
        printf("\n--- n = %d ---\n", n);
        DenseMatrix *A, *B, *C, *R;
        if (!prepare(n, &A, &B, &C, &R)) continue;

        int have_ref = run_config(out, "threads", seqv, 1, A, B, C, g_row_tile,
                                  g_block);
        if (have_ref) copy_matrix(R, C);

        for (int t = 0; t < g_threads_count; t++) {
            int p = g_threads[t];
            int ok = run_config(out, "threads", parv, p, A, B, C, g_row_tile,
                                g_block);
            if (ok && have_ref) {
                cross_log("threads", parv->name, p, seqv->name, n, C, R);
            }
        }
        dm_free(A); dm_free(B); dm_free(C); dm_free(R);
    }
}

/* ---- Experiment: SIMD vectorisation (optional extension) ---------------- */

static void run_simd(FILE *out) {
    printf("\n############################################################\n");
    printf("SIMD EXTENSION (novec vs vec builds; p = 1 and p = %d)\n",
           g_simd_threads);
    printf("############################################################\n");
    const char *order[4][2] = {{"seq", "novec"}, {"seq", "vec"},
                               {"par", "novec"}, {"par", "vec"}};

    for (int si = 0; si < g_simd_sizes_count; si++) {
        int n = g_simd_sizes[si];
        printf("\n--- n = %d ---\n", n);
        DenseMatrix *A, *B, *C, *R;
        if (!prepare(n, &A, &B, &C, &R)) continue;

        int have_ref = 0;
        for (int m = 0; m < 4; m++) {
            const KernelVariant *v = a3_find_variant(order[m][0], order[m][1]);
            int ok = run_config(out, "simd", v, g_simd_threads, A, B, C,
                                g_row_tile, g_block);
            if (!ok) continue;
            if (!have_ref) {
                copy_matrix(R, C);
                have_ref = 1;
            } else {
                cross_log("simd", v->name, v->threaded ? g_simd_threads : 1,
                          "tiled_seq_novec", n, C, R);
            }
        }
        dm_free(A); dm_free(B); dm_free(C); dm_free(R);
    }
}

/* ---- Experiment: row-tile tuning ---------------------------------------- */

static void run_tune(FILE *out) {
    printf("\n############################################################\n");
    printf("ROW-TILE TUNING (block %d)\n", g_block);
    printf("############################################################\n");
    const KernelVariant *seqv = a3_find_variant("seq", "vec");
    const KernelVariant *parv = a3_find_variant("par", "vec");

    for (int si = 0; si < g_sizes_count; si++) {
        int n = g_sizes[si];
        DenseMatrix *A, *B, *C, *R;
        if (!prepare(n, &A, &B, &C, &R)) continue;
        for (int b = 0; b < g_row_tiles_count; b++) {
            int bi = g_row_tiles[b];
            printf("\n--- n = %d, row tile = %d ---\n", n, bi);
            run_config(out, "tune", seqv, 1, A, B, C, bi, g_block);
            for (int t = 0; t < g_threads_count; t++) {
                if (g_threads[t] == 1) continue;
                run_config(out, "tune", parv, g_threads[t], A, B, C, bi, g_block);
            }
        }
        dm_free(A); dm_free(B); dm_free(C); dm_free(R);
    }
}

/* ---- CLI ----------------------------------------------------------------- */

static void print_usage(void) {
    printf("Usage: a3_bench [options]\n");
    printf("  --validate-only          run the correctness gate and exit\n");
    printf("  --experiment threads|simd|tune|all   (default: threads)\n");
    printf("  --sizes n1 n2 ...        sizes for threads/tune experiments\n");
    printf("  --threads p1 p2 ...      thread counts for threads/tune\n");
    printf("  --simd-sizes n1 ...      sizes for the SIMD experiment\n");
    printf("  --simd-threads p         thread count of the parallel SIMD runs\n");
    printf("  --row-tiles b1 b2 ...    row tiles for the tuning sweep\n");
    printf("  --row-tile b             row tile (default %d)\n", A3_DEFAULT_ROW_TILE);
    printf("  --block b                k/j tile (default %d)\n", A3_DEFAULT_BLOCK);
    printf("  --matrices-dir <dir>     directory with A_n.csv / B_n.csv\n");
    printf("  --out <file>             results CSV path\n");
    printf("  --reps <r>               measured repetitions (default %d)\n",
           A3_MEASURED_REPS);
    printf("  --warmups <w>            warm-up runs (default %d)\n", A3_WARMUP_RUNS);
    printf("  --budget <sec>           per-run watchdog budget (default %.0f)\n",
           A3_RUN_BUDGET_SEC);
    printf("  --sysinfo                print the runtime fingerprint and exit\n");
    printf("  --dump-result <n> <p> <file>  run tiled_par_vec with p threads on\n"
           "                           A_n x B_n and save C (NumPy cross-check)\n");
}

static int parse_int_list(char **argv, int argc, int *i, int *out) {
    int count = 0;
    while (*i + 1 < argc && argv[*i + 1][0] != '-') {
        (*i)++;
        if (count < A3_MAX_LIST) out[count++] = atoi(argv[*i]);
    }
    return count;
}

int main(int argc, char *argv[]) {
    int validate_only = 0, sysinfo_only = 0;
    int do_threads = 1, do_simd = 0, do_tune = 0;
    int dump_n = -1, dump_p = 1;
    const char *dump_path = NULL;
    const char *user_matrices_dir = NULL;
    const char *out_path_arg = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate-only") == 0) {
            validate_only = 1;
        } else if (strcmp(argv[i], "--sysinfo") == 0) {
            sysinfo_only = 1;
        } else if (strcmp(argv[i], "--experiment") == 0 && i + 1 < argc) {
            const char *e = argv[++i];
            do_threads = strcmp(e, "threads") == 0 || strcmp(e, "all") == 0;
            do_simd = strcmp(e, "simd") == 0 || strcmp(e, "all") == 0;
            do_tune = strcmp(e, "tune") == 0;
            if (!do_threads && !do_simd && !do_tune) {
                printf("Unknown experiment: %s\n", e);
                return 1;
            }
        } else if (strcmp(argv[i], "--sizes") == 0) {
            g_sizes_count = parse_int_list(argv, argc, &i, g_sizes);
        } else if (strcmp(argv[i], "--threads") == 0) {
            g_threads_count = parse_int_list(argv, argc, &i, g_threads);
        } else if (strcmp(argv[i], "--simd-sizes") == 0) {
            g_simd_sizes_count = parse_int_list(argv, argc, &i, g_simd_sizes);
        } else if (strcmp(argv[i], "--simd-threads") == 0 && i + 1 < argc) {
            g_simd_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--row-tiles") == 0) {
            g_row_tiles_count = parse_int_list(argv, argc, &i, g_row_tiles);
        } else if (strcmp(argv[i], "--row-tile") == 0 && i + 1 < argc) {
            g_row_tile = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            g_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--matrices-dir") == 0 && i + 1 < argc) {
            user_matrices_dir = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path_arg = argv[++i];
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            g_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            g_warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
            g_budget = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dump-result") == 0 && i + 3 < argc) {
            dump_n = atoi(argv[++i]);
            dump_p = atoi(argv[++i]);
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    /* Thread counts are controlled explicitly: no dynamic adjustment. */
    omp_set_dynamic(0);

    if (sysinfo_only) {
        a3_write_sysinfo(stdout);
        return 0;
    }

    printf("Assignment 3 benchmark harness (C + OpenMP)\n");
    printf("Warm-up runs: %d | Measured reps: %d | Per-run budget: %.0f s | "
           "Tiles: %d rows x %d\n",
           g_warmups, g_reps, g_budget, g_row_tile, g_block);

    if (user_matrices_dir) g_matrix_dirs[g_matrix_dirs_count++] = user_matrices_dir;
    g_matrix_dirs[g_matrix_dirs_count++] = A3_MATRICES_DIR_A3;
    g_matrix_dirs[g_matrix_dirs_count++] = A3_MATRICES_DIR_A1;

    /* Correctness gate: always first, always mandatory. */
    if (!a3_validate_all(g_validate_threads, g_validate_threads_count)) {
        printf("Correctness validation FAILED. Aborting.\n");
        return 1;
    }
    if (validate_only) return 0;

    /* Optional standalone result dump for the external NumPy cross-check. */
    if (dump_n > 0 && dump_path) {
        DenseMatrix *A = load_ab(dump_n, 'A');
        DenseMatrix *B = load_ab(dump_n, 'B');
        if (!A || !B) {
            printf("Cannot dump result: matrices for n = %d not found.\n", dump_n);
            return 1;
        }
        DenseMatrix *C = dm_alloc(dump_n);
        omp_set_num_threads(dump_p);
        k_tiled_par_vec(A->data, B->data, C->data, dump_n, g_row_tile, g_block, 0.0);
        int saved = dm_save_csv(C, dump_path);
        printf(saved ? "Result C (%d x %d, p = %d) written to %s\n"
                     : "Failed to write result (%d x %d, p = %d) to %s\n",
               dump_n, dump_n, dump_p, dump_path);
        dm_free(A); dm_free(B); dm_free(C);
        return saved ? 0 : 1;
    }

    CreateDirectoryA("assignment3", NULL);
    CreateDirectoryA("assignment3\\data", NULL);
    CreateDirectoryA(A3_RESULTS_DIR, NULL);

    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));

    /* Runtime fingerprint stored next to the results. */
    char info_path[512];
    snprintf(info_path, sizeof(info_path), "%s\\sysinfo_%s.csv", A3_RESULTS_DIR,
             timestamp);
    FILE *info = fopen(info_path, "w");
    if (info) {
        fprintf(info, "key,value\n");
        a3_write_sysinfo(info);
        fclose(info);
    }
    a3_write_sysinfo(stdout);

    char out_buf[512];
    const char *out_path = out_path_arg;
    if (!out_path) {
        snprintf(out_buf, sizeof(out_buf), "%s\\results_a3_%s.csv",
                 A3_RESULTS_DIR, timestamp);
        out_path = out_buf;
    }
    FILE *out = fopen(out_path, "w");
    if (!out) {
        printf("Failed to open output file %s\n", out_path);
        return 1;
    }
    csv_header(out);

    char cross_buf[512];
    snprintf(cross_buf, sizeof(cross_buf), "%s\\crosscheck_log_%s.csv",
             A3_RESULTS_DIR, timestamp);
    g_cross_file = fopen(cross_buf, "w");
    if (g_cross_file) {
        fprintf(g_cross_file, "assignment,experiment,method,threads,reference,"
                              "n,max_abs_diff,bitwise_equal,status\n");
    }
    printf("Results file: %s\nCross-check log: %s\n", out_path, cross_buf);

    double t_start = a3_time_sec();
    if (do_tune) run_tune(out);
    if (do_threads) run_threads(out);
    if (do_simd) run_simd(out);
    double t_total = a3_time_sec() - t_start;

    fclose(out);
    if (g_cross_file) fclose(g_cross_file);
    printf("\nBenchmark finished in %.1f s. Raw results: %s\n", t_total, out_path);
    return 0;
}
