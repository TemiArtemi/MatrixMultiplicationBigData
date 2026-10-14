/**
 * a2_bench.c - Benchmark harness for Assignment 2.
 *
 * Part A: dense methods (validated i-k-j-order baseline plus two optimised
 *         variants) on the reused Assignment 1 matrices.
 * Part B: sparse CSR x CSR product (Gustavson) versus the best dense method
 *         on identical logical sparse inputs, at several density levels.
 *
 * Protocol enforced here (see assignment2/README.md for the rationale):
 *   - correctness gate runs first and aborts the harness on failure;
 *   - inputs are generated/loaded OUTSIDE the timed region; the output
 *     matrix is allocated by the harness outside the timer and initialised
 *     by the kernel itself inside it (documented boundary);
 *   - one warm-up run (repetition_id = 0, kept in the CSV but excluded from
 *     statistics) plus N measured repetitions per configuration;
 *   - a monotonic timer (CLOCK_MONOTONIC) and an anti-dead-code-elimination
 *     checksum (trace of C) per repetition;
 *   - a per-run watchdog of --budget seconds aborts a run and records a
 *     TIMEOUT row with no invented time;
 *   - every raw measurement is appended to a CSV file with the metadata
 *     columns required by the common experimental protocol.
 */

#include "a2_config.h"
#include "a2_timer.h"
#include "a2_matrix.h"
#include "a2_csr.h"
#include "a2_kernels.h"
#include "a2_validate.h"

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Experiment defaults (overridable from the command line) ----------- */
static int g_sizes[] = {10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000};
static int g_sizes_count = (int)(sizeof(g_sizes) / sizeof(g_sizes[0]));

static int g_sparse_sizes[] = {200, 500, 1000, 2000, 3000};
static int g_sparse_sizes_count = (int)(sizeof(g_sparse_sizes) / sizeof(g_sparse_sizes[0]));

/* Only these densities are run at n = 3000 (large products become dense and
 * the dense side of the comparison dominates the budget). */
static double g_low_densities_at_top[] = {0.10, 0.01, 0.001};

static double g_densities[] = {0.50, 0.20, 0.10, 0.05, 0.01, 0.001};
static int g_densities_count = (int)(sizeof(g_densities) / sizeof(g_densities[0]));

static int g_reps = A2_MEASURED_REPS;
static int g_warmups = A2_WARMUP_RUNS;
static double g_budget = A2_RUN_BUDGET_SEC;
static const char *g_out_path = NULL;
static int g_crosscheck = 1;

static const char *g_method_names[8];
static int g_method_count = 0;

/* ---- CSV helpers -------------------------------------------------------- */

static const char *MEM_METRIC = "peak_rss_working_set_mb";

static void csv_header(FILE *f) {
    fprintf(f, "assignment,method,language,n,numeric_type,seed,warmup_runs,"
               "repetition_id,threads_or_workers,density,execution_time_ms,"
               "memory_metric_name,memory_value_mb,checksum,status,nnz_A,"
               "nnz_B,nnz_C,block_size,theoretical_storage_mb\n");
}

static void csv_row(FILE *f, const char *method, int n, int rep_id,
                    double density_pct, double time_ms, double peak_mb,
                    double checksum, const char *status,
                    long long nnz_a, long long nnz_b, long long nnz_c,
                    int block, double storage_mb) {
    fprintf(f, "2,%s,C,%d,float64,%u,%d,%d,1,%.4f,", method, n, A2_SEED,
            g_warmups, rep_id, density_pct);
    if (strcmp(status, "OK") == 0) {
        fprintf(f, "%.4f", time_ms);
    }
    /* TIMEOUT: write nothing; the trailing comma of the previous field
     * already leaves execution_time_ms empty (no invented time). */
    fprintf(f, ",%s,%.3f,%.6e,%s,%lld,%lld,%lld,", MEM_METRIC, peak_mb,
            checksum, status, nnz_a, nnz_b, nnz_c);
    if (block > 0) {
        fprintf(f, "%d", block);
    }
    /* block == 0: write nothing; the previous trailing comma already
     * leaves block_size empty. */
    fprintf(f, ",%.4f\n", storage_mb);
    fflush(f);
}

/* ---- Cross-check log ----------------------------------------------------- */

static FILE *g_cross_file = NULL;

static void cross_log(const char *part, const char *a, const char *b, int n,
                      double density_pct, double max_diff, int ok) {
    if (!g_cross_file) return;
    fprintf(g_cross_file, "2,%s,%s,%s,%d,%.4f,%.6e,%.1e,%.1e,%s\n", part, a, b,
            n, density_pct, max_diff, A2_CROSS_ABS_TOL, A2_REL_TOL,
            ok ? "OK" : "MISMATCH");
    fflush(g_cross_file);
    printf("   cross-check %s vs %s (n=%d, density=%.2f%%): max |diff| = %.3e"
           " [%s]\n",
           a, b, n, density_pct, max_diff, ok ? "OK" : "MISMATCH");
}

/* ---- Matrix resolution --------------------------------------------------- */

static const char *g_matrix_dirs[3];
static int g_matrix_dirs_count = 0;

static DenseMatrix *load_ab(int n, char which) {
    char path[512];
    for (int d = 0; d < g_matrix_dirs_count; d++) {
        snprintf(path, sizeof(path), "%s\\%c_%d.csv", g_matrix_dirs[d], which, n);
        DenseMatrix *m = dm_load_csv(path);
        if (m) {
            if (d > 0) {
                printf("   (loaded %s from %s)\n", path, g_matrix_dirs[d]);
            }
            return m;
        }
    }
    return NULL;
}

/* ---- Run primitives ------------------------------------------------------ */

typedef struct {
    int completed;
    double time_ms;
    double peak_mb;
} RunResult;

static RunResult run_dense(const DenseKernel *k, const DenseMatrix *A,
                           const DenseMatrix *B, DenseMatrix *C, int batches) {
    RunResult r;
    MemSampler s;
    mem_sampler_start(&s);
    double deadline = a2_time_sec() + g_budget;
    double t0 = a2_time_sec();
    int ok = 1;
    for (int b = 0; b < batches; b++) {
        ok = k->fn(A->data, B->data, C->data, A->n, deadline);
        if (!ok) break;
    }
    double t1 = a2_time_sec();
    r.peak_mb = mem_sampler_stop(&s);
    r.completed = ok;
    r.time_ms = (t1 - t0) * 1000.0 / batches;
    return r;
}

typedef struct {
    int completed;
    double time_ms;
    double peak_mb;
    CSRMatrix *c;
} SparseRunResult;

static SparseRunResult run_sparse(const CSRMatrix *A, const CSRMatrix *B) {
    SparseRunResult r;
    MemSampler s;
    mem_sampler_start(&s);
    double deadline = a2_time_sec() + g_budget;
    double t0 = a2_time_sec();
    CSRMatrix *C = NULL;
    int ok = spgemm_gustavson(A, B, &C, deadline);
    double t1 = a2_time_sec();
    r.peak_mb = mem_sampler_stop(&s);
    r.completed = (ok == 1);
    r.time_ms = (t1 - t0) * 1000.0;
    r.c = r.completed ? C : NULL;
    if (!r.completed && C) csr_free(C);
    return r;
}

/* ---- Part A -------------------------------------------------------------- */

static void run_part_a(FILE *out) {
    printf("\n############################################################\n");
    printf("PART A - DENSE OPTIMISATION (baseline ijk vs ikj vs blocked)\n");
    printf("############################################################\n");

    int kcount;
    const DenseKernel *ks = a2_dense_kernels(&kcount);

    /* Resolve which kernels to run, preserving registry order. */
    const DenseKernel *selected[8];
    int sel_count = 0;
    if (g_method_count == 0) {
        for (int i = 0; i < kcount; i++) selected[sel_count++] = &ks[i];
    } else {
        for (int i = 0; i < kcount; i++) {
            for (int m = 0; m < g_method_count; m++) {
                if (strcmp(ks[i].name, g_method_names[m]) == 0) {
                    selected[sel_count++] = &ks[i];
                }
            }
        }
    }

    for (int si = 0; si < g_sizes_count; si++) {
        int n = g_sizes[si];
        printf("\n--- Part A: n = %d ---\n", n);

        DenseMatrix *A = load_ab(n, 'A');
        DenseMatrix *B = load_ab(n, 'B');
        if (!A || !B) {
            printf("   Matrices for n = %d not found in the search path; "
                   "SKIPPING (run scripts/generate_matrices.py to create "
                   "identical files).\n", n);
            dm_free(A);
            dm_free(B);
            continue;
        }
        long long nnz = (long long)n * (long long)n;
        double storage_mb = 3.0 * 8.0 * (double)nnz / (1024.0 * 1024.0);
        int batches = (n <= A2_TINY_MAX_N) ? A2_TINY_BATCH : 1;

        DenseMatrix *C = dm_alloc(n);
        DenseMatrix *Cref = dm_alloc(n);
        if (!C || !Cref) {
            printf("   Allocation failed (OOM) for n = %d\n", n);
            csv_row(out, "harness", n, 0, 100.0, 0.0, 0.0, 0.0, "OOM", nnz,
                    nnz, nnz, 0, storage_mb);
            dm_free(A); dm_free(B); dm_free(C); dm_free(Cref);
            continue;
        }

        int have_ref = 0;
        const char *ref_name = "";

        for (int m = 0; m < sel_count; m++) {
            const DenseKernel *k = selected[m];
            int block = k->uses_block ? g_block_size : 0;
            int last_ok = (g_warmups == 0); /* C valid once something ran */

            /* Warm-up runs (repetition_id = 0, excluded from statistics). */
            for (int w = 0; w < g_warmups; w++) {
                RunResult r = run_dense(k, A, B, C, batches);
                csv_row(out, k->name, n, 0, 100.0, r.time_ms, r.peak_mb,
                        dm_trace(C), r.completed ? "OK" : "TIMEOUT", nnz, nnz,
                        nnz, block, storage_mb);
                printf("   %-20s warm-up: %s (%.2f ms)\n", k->name,
                       r.completed ? "OK" : "TIMEOUT", r.time_ms);
                if (!r.completed) {
                    printf("   %s exceeds the %.0f s per-run budget at n = %d; "
                           "skipping measured repetitions.\n",
                           k->name, g_budget, n);
                    break;
                }
                last_ok = 1;
            }

            /* Measured repetitions. */
            for (int rep = 1; rep <= g_reps && last_ok; rep++) {
                RunResult r = run_dense(k, A, B, C, batches);
                csv_row(out, k->name, n, rep, 100.0, r.time_ms, r.peak_mb,
                        dm_trace(C), r.completed ? "OK" : "TIMEOUT", nnz, nnz,
                        nnz, block, storage_mb);
                printf("   %-20s rep %d: %s (%.2f ms, peak %.1f MB)\n",
                       k->name, rep, r.completed ? "OK" : "TIMEOUT",
                       r.time_ms, r.peak_mb);
                if (!r.completed) break;
                last_ok = 1;
            }

            /* Cross-method correctness on the real input. */
            if (g_crosscheck) {
                if (!have_ref && last_ok) {
                    memcpy(Cref->data, C->data,
                           (size_t)n * (size_t)n * sizeof(double));
                    have_ref = 1;
                    ref_name = k->name;
                } else if (have_ref && last_ok && k->name != ref_name) {
                    int ok = 0;
                    double diff = dm_max_abs_diff(C, Cref, &ok,
                                                  A2_CROSS_ABS_TOL, A2_REL_TOL);
                    cross_log("A", k->name, ref_name, n, 100.0, diff, ok);
                }
            }
        }

        dm_free(A);
        dm_free(B);
        dm_free(C);
        dm_free(Cref);
    }
}

/* ---- Part B -------------------------------------------------------------- */

static int density_in_low_list(double d) {
    int count = (int)(sizeof(g_low_densities_at_top) /
                      sizeof(g_low_densities_at_top[0]));
    for (int i = 0; i < count; i++) {
        if (d == g_low_densities_at_top[i]) return 1;
    }
    return 0;
}

static void run_part_b(FILE *out) {
    printf("\n############################################################\n");
    printf("PART B - SPARSE CSR (Gustavson) vs best dense method\n");
    printf("############################################################\n");

    const DenseKernel *dense_best = a2_find_kernel("dense_blocked");
    if (!dense_best) {
        printf("dense_blocked kernel not found; cannot run Part B.\n");
        return;
    }

    for (int si = 0; si < g_sparse_sizes_count; si++) {
        int n = g_sparse_sizes[si];

        for (int di = 0; di < g_densities_count; di++) {
            double d = g_densities[di];

            /* At the largest sparse size only the low-density levels run
             * (budget control, declared in the report). */
            if (n == 3000 && !density_in_low_list(d)) continue;

            printf("\n--- Part B: n = %d, target density %.3f%% ---\n", n,
                   d * 100.0);

            /* Deterministic generation: the stream restarts from A2_SEED for
             * every configuration, so zero patterns are nested across
             * densities at fixed n (documented in the report). */
            unsigned int seed = A2_SEED;
            DenseMatrix *A = dm_generate_sparse_pattern(n, d, &seed);
            DenseMatrix *B = dm_generate_sparse_pattern(n, d, &seed);
            if (!A || !B) {
                printf("   Generation failed (OOM); skipping configuration.\n");
                csv_row(out, "harness", n, 0, d * 100.0, 0.0, 0.0, 0.0, "OOM",
                        0, 0, 0, 0, 0.0);
                dm_free(A);
                dm_free(B);
                continue;
            }

            long long nnz_a = dm_count_nnz(A);
            long long nnz_b = dm_count_nnz(B);
            long long dense_storage_bytes =
                3LL * 8LL * (long long)n * (long long)n;

            DenseMatrix *C = dm_alloc(n);
            DenseMatrix *Sc = NULL; /* product kept for cross-checks */
            CSRMatrix *Sa = NULL, *Sb = NULL, *Csr = NULL;
            if (!C) {
                printf("   Allocation failed (OOM); skipping configuration.\n");
                csv_row(out, "dense_blocked", n, 0, d * 100.0, 0.0, 0.0, 0.0,
                        "OOM", nnz_a, nnz_b, 0, g_block_size, 0.0);
                dm_free(A); dm_free(B); dm_free(C);
                continue;
            }

            /* -- 1) Best dense method on the identical logical inputs. */
            double dense_storage_mb =
                (double)dense_storage_bytes / (1024.0 * 1024.0);
            long long dense_nnz_c = 0;
            int last_dense_ok = 0;
            for (int w = 0; w < g_warmups; w++) {
                RunResult r = run_dense(dense_best, A, B, C, 1);
                if (r.completed) dense_nnz_c = dm_count_nnz(C);
                csv_row(out, "dense_blocked", n, 0, d * 100.0, r.time_ms,
                        r.peak_mb, dm_trace(C), r.completed ? "OK" : "TIMEOUT",
                        nnz_a, nnz_b, dense_nnz_c, g_block_size,
                        dense_storage_mb);
                printf("   dense_blocked         warm-up: %s (%.2f ms)\n",
                       r.completed ? "OK" : "TIMEOUT", r.time_ms);
                last_dense_ok = r.completed;
                if (!r.completed) break;
            }
            for (int rep = 1; rep <= g_reps && last_dense_ok; rep++) {
                RunResult r = run_dense(dense_best, A, B, C, 1);
                if (r.completed) dense_nnz_c = dm_count_nnz(C);
                csv_row(out, "dense_blocked", n, rep, d * 100.0, r.time_ms,
                        r.peak_mb, dm_trace(C), r.completed ? "OK" : "TIMEOUT",
                        nnz_a, nnz_b, dense_nnz_c, g_block_size,
                        dense_storage_mb);
                printf("   dense_blocked         rep %d: %s (%.2f ms)\n", rep,
                       r.completed ? "OK" : "TIMEOUT", r.time_ms);
                if (!r.completed) break;
            }
            if (last_dense_ok) {
                Sc = dm_alloc(n);
                if (Sc) {
                    memcpy(Sc->data, C->data,
                           (size_t)n * (size_t)n * sizeof(double));
                }
            }

            /* -- 2) Dense -> CSR conversion, timed separately.
             * Same repetition protocol as every other configuration:
             * g_warmups rows with repetition_id = 0 (excluded from the
             * statistics) followed by g_reps measured rows. */
            double conv_storage_mb = 0.0;
            for (int rep = 0; rep < g_warmups + g_reps; rep++) {
                double t0 = a2_time_sec();
                CSRMatrix *ta = csr_from_dense(A);
                CSRMatrix *tb = csr_from_dense(B);
                double t1 = a2_time_sec();
                if (!ta || !tb) {
                    printf("   conversion OOM at rep %d\n", rep + 1);
                    csr_free(ta);
                    csr_free(tb);
                    break;
                }
                conv_storage_mb = (double)(csr_storage_bytes(ta) +
                                           csr_storage_bytes(tb)) /
                                  (1024.0 * 1024.0);
                csv_row(out, "dense_to_csr_conversion", n, rep, d * 100.0,
                        (t1 - t0) * 1000.0, mem_current_rss_mb(),
                        (double)(ta->nnz + tb->nnz), "OK", nnz_a, nnz_b,
                        dense_nnz_c, 0, conv_storage_mb);
                if (rep == g_warmups) {
                    printf("   dense_to_csr_conv     rep %d: OK (%.2f ms)\n",
                           rep, (t1 - t0) * 1000.0);
                }
                csr_free(Sa);
                csr_free(Sb);
                Sa = ta;
                Sb = tb;
            }

            /* -- 3) Sparse CSR kernel (Gustavson). */
            long long sparse_nnz_c = 0;
            int last_sparse_ok = (g_warmups == 0);
            double sparse_storage_mb = 0.0;
            for (int w = 0; w < g_warmups; w++) {
                SparseRunResult r = run_sparse(Sa, Sb);
                if (r.c) {
                    sparse_nnz_c = r.c->nnz;
                    sparse_storage_mb =
                        (double)(csr_storage_bytes(Sa) + csr_storage_bytes(Sb) +
                                 csr_storage_bytes(r.c) +
                                 (size_t)n * (sizeof(double) + 2 * sizeof(int) +
                                              sizeof(int) + 16)) /
                        (1024.0 * 1024.0);
                }
                csv_row(out, "sparse_csr_gustavson", n, 0, d * 100.0, r.time_ms,
                        r.peak_mb, r.c ? csr_trace(r.c) : 0.0,
                        r.completed ? "OK" : "TIMEOUT", nnz_a, nnz_b,
                        sparse_nnz_c, 0, sparse_storage_mb);
                printf("   sparse_csr_gustavson  warm-up: %s (%.2f ms, "
                       "nnz_C = %lld)\n",
                       r.completed ? "OK" : "TIMEOUT", r.time_ms, sparse_nnz_c);
                last_sparse_ok = r.completed;
                if (!r.completed) break;
            }
            for (int rep = 1; rep <= g_reps && last_sparse_ok; rep++) {
                SparseRunResult r = run_sparse(Sa, Sb);
                if (r.c) {
                    sparse_nnz_c = r.c->nnz;
                    sparse_storage_mb =
                        (double)(csr_storage_bytes(Sa) + csr_storage_bytes(Sb) +
                                 csr_storage_bytes(r.c) +
                                 (size_t)n * (sizeof(double) + 2 * sizeof(int) +
                                              sizeof(int) + 16)) /
                        (1024.0 * 1024.0);
                    if (Csr) csr_free(Csr);
                    Csr = r.c; /* keep the last product for the cross-check */
                }
                csv_row(out, "sparse_csr_gustavson", n, rep, d * 100.0,
                        r.time_ms, r.peak_mb,
                        Csr ? csr_trace(Csr) : 0.0,
                        r.completed ? "OK" : "TIMEOUT", nnz_a, nnz_b,
                        sparse_nnz_c, 0, sparse_storage_mb);
                printf("   sparse_csr_gustavson  rep %d: %s (%.2f ms, "
                       "nnz_C = %lld)\n",
                       rep, r.completed ? "OK" : "TIMEOUT", r.time_ms,
                       sparse_nnz_c);
                if (!r.completed) break;
            }

            /* -- 4) Cross-check sparse vs dense on the identical inputs. */
            if (g_crosscheck && Csr && Sc) {
                DenseMatrix *Csp = csr_to_dense(Csr);
                if (Csp) {
                    int ok = 0;
                    double diff = dm_max_abs_diff(Csp, Sc, &ok,
                                                  A2_CROSS_ABS_TOL, A2_REL_TOL);
                    cross_log("B", "sparse_csr_gustavson", "dense_blocked", n,
                              d * 100.0, diff, ok);
                    dm_free(Csp);
                }
            }

            dm_free(A);
            dm_free(B);
            dm_free(C);
            dm_free(Sc);
            csr_free(Sa);
            csr_free(Sb);
            csr_free(Csr);
        }
    }
}

/* ---- CLI ----------------------------------------------------------------- */

static void print_usage(void) {
    printf("Usage: a2_bench [options]\n");
    printf("  --validate-only        run the correctness gate and exit\n");
    printf("  --part a|b|all         which experiment to run (default: all)\n");
    printf("  --sizes n1 n2 ...      Part A matrix sizes\n");
    printf("  --sparse-sizes n1 ...  Part B matrix sizes\n");
    printf("  --densities d1 ...     Part B densities as fractions (0.01 = 1%%)\n");
    printf("  --methods a,b,c        dense methods to run in Part A\n");
    printf("  --matrices-dir <dir>   directory with A_n.csv / B_n.csv\n");
    printf("  --out <file>           results CSV path\n");
    printf("  --block <b>            cache block size (default %d)\n",
           A2_DEFAULT_BLOCK);
    printf("  --reps <r>             measured repetitions (default %d)\n",
           A2_MEASURED_REPS);
    printf("  --warmups <w>          warm-up runs (default %d)\n",
           A2_WARMUP_RUNS);
    printf("  --budget <sec>         per-run watchdog budget (default %.0f)\n",
           A2_RUN_BUDGET_SEC);
    printf("  --no-crosscheck        skip cross-method correctness checks\n");
    printf("  --dump-result <n> <p>  run baseline on A_n x B_n, save C to CSV\n");
    printf("  --list-methods         print registered dense methods\n");
}

static int parse_int_list(char **argv, int argc, int *i, int *out, int max) {
    int count = 0;
    while (*i + 1 < argc && argv[*i + 1][0] != '-') {
        (*i)++;
        if (count < max) out[count++] = atoi(argv[*i]);
    }
    return count;
}

static int parse_double_list(char **argv, int argc, int *i, double *out,
                             int max) {
    int count = 0;
    while (*i + 1 < argc && argv[*i + 1][0] != '-') {
        (*i)++;
        if (count < max) out[count++] = atof(argv[*i]);
    }
    return count;
}

int main(int argc, char *argv[]) {
    int validate_only = 0;
    int part_a = 1, part_b = 1;
    int dump_n = -1;
    const char *dump_path = NULL;
    const char *user_matrices_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate-only") == 0) {
            validate_only = 1;
        } else if (strcmp(argv[i], "--part") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "a") == 0) {
                part_a = 1; part_b = 0;
            } else if (strcmp(argv[i], "b") == 0) {
                part_a = 0; part_b = 1;
            } else {
                part_a = 1; part_b = 1;
            }
        } else if (strcmp(argv[i], "--sizes") == 0) {
            g_sizes_count = parse_int_list(argv, argc, &i, g_sizes,
                                           (int)(sizeof(g_sizes) / sizeof(g_sizes[0])));
        } else if (strcmp(argv[i], "--sparse-sizes") == 0) {
            g_sparse_sizes_count = parse_int_list(
                argv, argc, &i, g_sparse_sizes,
                (int)(sizeof(g_sparse_sizes) / sizeof(g_sparse_sizes[0])));
        } else if (strcmp(argv[i], "--densities") == 0) {
            g_densities_count = parse_double_list(
                argv, argc, &i, g_densities,
                (int)(sizeof(g_densities) / sizeof(g_densities[0])));
        } else if (strcmp(argv[i], "--methods") == 0 && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && g_method_count < 8) {
                if (!a2_find_kernel(tok)) {
                    printf("Unknown method: %s\n", tok);
                    return 1;
                }
                g_method_names[g_method_count++] = tok;
                tok = strtok(NULL, ",");
            }
        } else if (strcmp(argv[i], "--matrices-dir") == 0 && i + 1 < argc) {
            user_matrices_dir = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            g_out_path = argv[++i];
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            g_block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            g_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            g_warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
            g_budget = atof(argv[++i]);
        } else if (strcmp(argv[i], "--no-crosscheck") == 0) {
            g_crosscheck = 0;
        } else if (strcmp(argv[i], "--dump-result") == 0 && i + 2 < argc) {
            dump_n = atoi(argv[++i]);
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--list-methods") == 0) {
            int kc;
            const DenseKernel *ks = a2_dense_kernels(&kc);
            for (int m = 0; m < kc; m++) printf("%s\n", ks[m].name);
            return 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    printf("Assignment 2 benchmark harness (C)\n");
    printf("Warm-up runs: %d | Measured reps: %d | Per-run budget: %.0f s | "
           "Block size: %d\n",
           g_warmups, g_reps, g_budget, g_block_size);

    /* Matrix search path: explicit dir, then Assignment 2, then Assignment 1
     * (reused dataset). Must be ready before any load, including dumps. */
    if (user_matrices_dir) {
        g_matrix_dirs[g_matrix_dirs_count++] = user_matrices_dir;
    }
    g_matrix_dirs[g_matrix_dirs_count++] = A2_MATRICES_DIR_A2;
    g_matrix_dirs[g_matrix_dirs_count++] = A2_MATRICES_DIR_A1;

    /* Correctness gate: always first, always mandatory. */
    if (!a2_validate_all()) {
        printf("Correctness validation FAILED. Aborting.\n");
        return 1;
    }
    if (validate_only) {
        return 0;
    }

    /* Optional standalone result dump for the external NumPy cross-check. */
    if (dump_n > 0 && dump_path) {
        DenseMatrix *A = load_ab(dump_n, 'A');
        DenseMatrix *B = load_ab(dump_n, 'B');
        if (!A || !B) {
            printf("Cannot dump result: matrices for n = %d not found.\n", dump_n);
            return 1;
        }
        DenseMatrix *C = dm_alloc(dump_n);
        k_ijk(A->data, B->data, C->data, dump_n, 0.0);
        if (dm_save_csv(C, dump_path)) {
            printf("Result C (%d x %d) written to %s\n", dump_n, dump_n,
                   dump_path);
        } else {
            printf("Failed to write %s\n", dump_path);
            return 1;
        }
        dm_free(A); dm_free(B); dm_free(C);
        return 0;
    }

    CreateDirectoryA("assignment2", NULL);
    CreateDirectoryA("assignment2\\data", NULL);
    CreateDirectoryA(A2_RESULTS_DIR, NULL);

    /* Results CSV. */
    char timestamp[32];
    time_t now = time(NULL);
    struct tm *ltm = localtime(&now);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", ltm);

    char out_buf[512];
    const char *out_path = g_out_path;
    if (!out_path) {
        snprintf(out_buf, sizeof(out_buf), "%s\\results_a2_%s.csv",
                 A2_RESULTS_DIR, timestamp);
        out_path = out_buf;
    }
    FILE *out = fopen(out_path, "w");
    if (!out) {
        printf("Failed to open output file %s\n", out_path);
        return 1;
    }
    csv_header(out);

    /* Cross-check log. */
    if (g_crosscheck) {
        char cross_buf[512];
        snprintf(cross_buf, sizeof(cross_buf), "%s\\crosscheck_log_%s.csv",
                 A2_RESULTS_DIR, timestamp);
        g_cross_file = fopen(cross_buf, "w");
        if (g_cross_file) {
            fprintf(g_cross_file, "assignment,part,method_a,method_b,n,"
                                  "density,max_abs_diff,abs_tol,rel_tol,"
                                  "status\n");
            printf("Cross-check log: %s\n", cross_buf);
        }
    }

    printf("Results file: %s\n", out_path);

    double t_start = a2_time_sec();
    if (part_a) run_part_a(out);
    if (part_b) run_part_b(out);
    double t_total = a2_time_sec() - t_start;

    fclose(out);
    if (g_cross_file) fclose(g_cross_file);

    printf("\nBenchmark finished in %.1f s. Raw results: %s\n", t_total,
           out_path);
    return 0;
}
