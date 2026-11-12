/**
 * a4_bench.c - MPI benchmark harness for Assignment 4.
 *
 * Launched as:  mpiexec -n <p> a4_bench.exe [options]
 * One launch measures one worker count p (= number of MPI ranks, one
 * single-threaded process per worker) for every requested matrix size.
 *
 * Protocol enforced here (see assignment4/README.md for the rationale):
 *   - the correctness gate runs first on every launch and aborts all ranks
 *     on failure;
 *   - rank 0 loads A_n and B_n from CSV (outside every timer) and owns the
 *     full matrices; the other ranks start empty;
 *   - the end-to-end boundary is "A and B in rank-0 memory" -> "C in rank-0
 *     memory": scatter + broadcast + compute + gather, measured between
 *     barriers so the four phases are separable (see a4_partition.h);
 *     CSV loading and MPI start-up are outside and are measured by the
 *     driver script (launcher wall clock);
 *   - per-rank buffers are allocated and touched outside the timer;
 *   - g_warmups warm-up runs (repetition_id = 0, kept in the CSV, excluded
 *     from statistics) plus g_reps measured repetitions per configuration;
 *   - a monotonic timer (CLOCK_MONOTONIC) and the trace of C (anti dead-code
 *     checksum) per repetition;
 *   - a per-run watchdog of --budget seconds yields a TIMEOUT row with no
 *     time (collective decision, no invented value);
 *   - after each configuration C is compared bitwise with the single-rank
 *     result of the same kernel and logged.
 *
 * Memory metric: peak process working set (resident memory) of each rank,
 * sampled every 5 ms; reported as the sum over ranks plus rank 0 and the
 * largest non-root rank separately.
 */

#include "a4_config.h"
#include "a4_timer.h"
#include "a4_matrix.h"
#include "a4_kernels.h"
#include "a4_partition.h"
#include "a4_validate.h"
#include "a4_sysinfo.h"
#include "a4_affinity.h"

#include <mpi.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define A4_MAX_LIST 32
#define A4_MAX_N 40000   /* keeps rows * n inside the int counts of MPI */

/* ---- Experiment defaults (overridable from the command line) ----------- */
static int g_sizes[A4_MAX_LIST] = {1000, 2000, 3000, 4000};
static int g_sizes_count = 4;

static int g_row_tile = A4_DEFAULT_ROW_TILE;
static int g_block = A4_DEFAULT_BLOCK;
static int g_reps = A4_MEASURED_REPS;
static int g_warmups = A4_WARMUP_RUNS;
static double g_budget = A4_RUN_BUDGET_SEC;
static int g_reference = 1;
static int g_pin_single_to_pcores = 1;
static const char *g_experiment = "scaling";
static const char *g_partitioner = "mpi_rowblock";

static int g_rank = 0, g_size = 1;

/* ---- CSV helpers (rank 0 only) ------------------------------------------ */

static const char *MEM_METRIC = "peak_rss_working_set_mb_sum_over_ranks";

static void csv_header(FILE *f) {
    fprintf(f, "assignment,method,language,n,numeric_type,seed,warmup_runs,"
               "repetition_id,threads_or_workers,density,execution_time_ms,"
               "memory_metric_name,memory_value_mb,checksum,status,"
               "partitioner,row_tile,block_size,t_scatter_ms,t_bcast_ms,"
               "t_compute_ms,t_gather_ms,t_kernel_min_ms,t_kernel_max_ms,"
               "mem_rank0_mb,mem_max_nonroot_mb,placement,experiment,"
               "theoretical_storage_mb,t_load_ms\n");
}

typedef struct {
    int completed;
    PhaseTimes t;
    double mem_sum, mem_root, mem_max_nonroot;
} RunResult;

static void csv_row(FILE *f, int n, int rep_id, const RunResult *r,
                    double checksum, double storage_mb,
                    const char *placement, double load_ms) {
    fprintf(f, "4,%s,C,%d,float64,%u,%d,%d,%d,100.0000,", g_partitioner, n,
            A4_SEED, g_warmups, rep_id, g_size);
    if (r->completed) {
        fprintf(f, "%.4f", r->t.t_total * 1000.0);
    }
    /* TIMEOUT: every time field stays empty (no invented time). */
    fprintf(f, ",%s,%.3f,%.6e,%s,%s,%d,%d,", MEM_METRIC, r->mem_sum, checksum,
            r->completed ? "OK" : "TIMEOUT", g_partitioner, g_row_tile,
            g_block);
    if (r->completed) {
        fprintf(f, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", r->t.t_scatter * 1000.0,
                r->t.t_bcast * 1000.0, r->t.t_compute * 1000.0,
                r->t.t_gather * 1000.0, r->t.t_kernel_min * 1000.0,
                r->t.t_kernel_max * 1000.0);
    } else {
        fprintf(f, ",,,,,");
    }
    /* t_load_ms: rank-0 CSV loading of A_n and B_n, OUTSIDE the boundary. */
    fprintf(f, ",%.3f,%.3f,%s,%s,%.4f,%.1f\n", r->mem_root, r->mem_max_nonroot,
            placement, g_experiment, storage_mb, load_ms);
    fflush(f);
}

/* ---- Cross-check log (rank 0 only) --------------------------------------- */

static FILE *g_cross_file = NULL;

static void cross_log(int n, const DenseMatrix *C, const DenseMatrix *ref) {
    int ok = 0;
    double diff = dm_max_abs_diff(C, ref, &ok, A4_CROSS_ABS_TOL, A4_REL_TOL);
    int bitwise = memcmp(C->data, ref->data,
                         (size_t)n * (size_t)n * sizeof(double)) == 0;
    if (g_cross_file) {
        fprintf(g_cross_file, "4,%s,%s,%d,%d,%.6e,%d,%s\n", g_experiment,
                g_partitioner, g_size, n, diff, bitwise, ok ? "OK" : "MISMATCH");
        fflush(g_cross_file);
    }
    printf("   cross-check p=%d vs single-rank kernel: max |diff| = %.3e, "
           "bitwise %s [%s]\n", g_size, diff, bitwise ? "yes" : "no",
           ok ? "OK" : "MISMATCH");
}

/* ---- Matrix resolution (rank 0 only) ------------------------------------- */

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

/* ---- One measured run (collective) ---------------------------------------- */

static RunResult run_once(const Partitioner *part, void *ctx,
                          const DenseMatrix *A, const DenseMatrix *B,
                          DenseMatrix *C) {
    RunResult r;
    memset(&r, 0, sizeof(r));
    MemSampler s;
    mem_sampler_start(&s);

    double deadline = a4_time_sec() + g_budget;
    r.completed = part->multiply(ctx, g_rank == 0 ? A->data : NULL,
                                 g_rank == 0 ? B->data : NULL,
                                 g_rank == 0 ? C->data : NULL, g_row_tile,
                                 g_block, deadline, &r.t);
    double peak = mem_sampler_stop(&s);

    /* Memory is gathered after the timed region. */
    double *all = g_rank == 0 ? (double *)malloc((size_t)g_size * sizeof(double))
                              : NULL;
    MPI_Gather(&peak, 1, MPI_DOUBLE, all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (g_rank == 0) {
        r.mem_root = all[0];
        for (int i = 0; i < g_size; i++) {
            r.mem_sum += all[i];
            if (i > 0 && all[i] > r.mem_max_nonroot) r.mem_max_nonroot = all[i];
        }
        free(all);
    }
    return r;
}

/* ---- Experiment: strong scaling at this launch's worker count ------------ */

static void run_scaling(const Partitioner *part, FILE *out) {
    const char *placement =
        (g_size == 1 && g_pin_single_to_pcores) ? "ecoqos_off_pcores_only"
                                               : "ecoqos_off_all_cores";
    for (int si = 0; si < g_sizes_count; si++) {
        int n = g_sizes[si];
        DenseMatrix *A = NULL, *B = NULL, *C = NULL, *ref = NULL;
        int ready = 1;
        double load_ms = 0.0;
        if (g_rank == 0) {
            printf("\n--- n = %d, p = %d ranks ---\n", n, g_size);
            double l0 = a4_time_sec();
            A = load_ab(n, 'A');
            B = load_ab(n, 'B');
            load_ms = (a4_time_sec() - l0) * 1000.0;
            C = (A && B) ? dm_alloc(n) : NULL;
            ref = (C && g_reference) ? dm_alloc(n) : NULL;
            if (!A || !B) {
                printf("   Matrices for n = %d not found; SKIPPING (run "
                       "scripts/generate_matrices.py).\n", n);
                ready = 0;
            } else if (!C || (g_reference && !ref)) {
                printf("   Allocation failed (OOM) for n = %d; SKIPPING.\n", n);
                ready = 0;
            }
        }
        MPI_Bcast(&ready, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ready) {
            dm_free(A); dm_free(B); dm_free(C); dm_free(ref);
            continue;
        }

        /* Single-rank reference of the same kernel (outside every timer). */
        if (g_rank == 0 && g_reference) {
            a4_tiled_rows(A->data, B->data, ref->data, n, n, g_row_tile,
                          g_block, 0.0);
        }

        double storage_mb = 0.0;
        for (int r = 0; r < g_size; r++) {
            storage_mb += part->storage_bytes(r, g_size, n);
        }
        storage_mb /= (1024.0 * 1024.0);

        void *ctx = part->create(MPI_COMM_WORLD, n);
        int last_ok = 1;
        for (int rep = 0; rep <= g_reps && last_ok; rep++) {
            int runs = (rep == 0) ? g_warmups : 1;
            for (int w = 0; w < runs && last_ok; w++) {
                RunResult r = run_once(part, ctx, A, B, C);
                if (g_rank == 0) {
                    csv_row(out, n, rep, &r, r.completed ? dm_trace(C) : 0.0,
                            storage_mb, placement, load_ms);
                    if (rep == 0) {
                        printf("   warm-up: %s (total %.2f ms)\n",
                               r.completed ? "OK" : "TIMEOUT",
                               r.t.t_total * 1000.0);
                    } else if (r.completed) {
                        printf("   rep %d: OK total %.2f ms = scatter %.2f + "
                               "bcast %.2f + compute %.2f + gather %.2f "
                               "(kernel min/max %.2f/%.2f), mem sum %.0f MB\n",
                               rep, r.t.t_total * 1000.0,
                               r.t.t_scatter * 1000.0, r.t.t_bcast * 1000.0,
                               r.t.t_compute * 1000.0, r.t.t_gather * 1000.0,
                               r.t.t_kernel_min * 1000.0,
                               r.t.t_kernel_max * 1000.0, r.mem_sum);
                    } else {
                        printf("   rep %d: TIMEOUT (> %.0f s budget); remaining "
                               "repetitions skipped.\n", rep, g_budget);
                    }
                }
                last_ok = r.completed;
            }
        }
        if (g_rank == 0 && last_ok && g_reference) cross_log(n, C, ref);
        part->destroy(ctx);
        dm_free(A); dm_free(B); dm_free(C); dm_free(ref);
    }
}

/* ---- CLI ----------------------------------------------------------------- */

static void print_usage(void) {
    printf("Usage: mpiexec -n <p> a4_bench [options]\n");
    printf("  --validate-only          run the correctness gate and exit\n");
    printf("  --sizes n1 n2 ...        matrix sizes (default 1000 2000 3000 4000)\n");
    printf("  --row-tile b             kernel row tile (default %d)\n", A4_DEFAULT_ROW_TILE);
    printf("  --block b                kernel k/j tile (default %d)\n", A4_DEFAULT_BLOCK);
    printf("  --matrices-dir <dir>     directory with A_n.csv / B_n.csv\n");
    printf("  --out <file>             results CSV (appended; header if new)\n");
    printf("  --experiment <tag>       value of the 'experiment' column (default scaling)\n");
    printf("  --partitioner <name>     distribution strategy (default mpi_rowblock)\n");
    printf("  --reps <r>               measured repetitions (default %d)\n", A4_MEASURED_REPS);
    printf("  --warmups <w>            warm-up runs (default %d)\n", A4_WARMUP_RUNS);
    printf("  --budget <sec>           per-run watchdog budget (default %.0f)\n", A4_RUN_BUDGET_SEC);
    printf("  --no-reference           skip the single-rank reference/bitwise check\n");
    printf("  --single-rank-any-core   do not restrict p = 1 to the P-cores\n");
    printf("  --sysinfo                print the runtime fingerprint and exit\n");
    printf("  --dump-result <n> <file> run the product on A_n x B_n, rank 0 saves C\n");
}

static int parse_int_list(char **argv, int argc, int *i, int *out) {
    int count = 0;
    while (*i + 1 < argc && argv[*i + 1][0] != '-') {
        (*i)++;
        if (count < A4_MAX_LIST) out[count++] = atoi(argv[*i]);
    }
    return count;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    int validate_only = 0, sysinfo_only = 0, dump_n = -1;
    const char *dump_path = NULL;
    const char *user_matrices_dir = NULL;
    const char *out_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate-only") == 0) {
            validate_only = 1;
        } else if (strcmp(argv[i], "--sysinfo") == 0) {
            sysinfo_only = 1;
        } else if (strcmp(argv[i], "--sizes") == 0) {
            g_sizes_count = parse_int_list(argv, argc, &i, g_sizes);
        } else if (strcmp(argv[i], "--row-tile") == 0 && i + 1 < argc) {
            g_row_tile = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            g_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--matrices-dir") == 0 && i + 1 < argc) {
            user_matrices_dir = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--experiment") == 0 && i + 1 < argc) {
            g_experiment = argv[++i];
        } else if (strcmp(argv[i], "--partitioner") == 0 && i + 1 < argc) {
            g_partitioner = argv[++i];
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            g_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            g_warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
            g_budget = atof(argv[++i]);
        } else if (strcmp(argv[i], "--no-reference") == 0) {
            g_reference = 0;
        } else if (strcmp(argv[i], "--single-rank-any-core") == 0) {
            g_pin_single_to_pcores = 0;
        } else if (strcmp(argv[i], "--dump-result") == 0 && i + 2 < argc) {
            dump_n = atoi(argv[++i]);
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            if (g_rank == 0) print_usage();
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                print_usage();
            }
            MPI_Finalize();
            return 1;
        }
    }

    for (int i = 0; i < g_sizes_count; i++) {
        if (g_sizes[i] < 1 || g_sizes[i] > A4_MAX_N) {
            if (g_rank == 0) printf("Unsupported matrix size %d\n", g_sizes[i]);
            MPI_Finalize();
            return 1;
        }
    }

    const Partitioner *part = a4_find_partitioner(g_partitioner);
    if (!part) {
        if (g_rank == 0) printf("Unknown partitioner: %s\n", g_partitioner);
        MPI_Finalize();
        return 1;
    }

    /* Placement policy (lesson of Assignment 3): every rank opts out of
     * EcoQoS power throttling; a single rank is pinned to the P-cores. */
    int ecoqos = a4_disable_power_throttling();
    if (g_size == 1 && g_pin_single_to_pcores) a4_restrict_to_pcores();

    if (sysinfo_only) {
        if (g_rank == 0) a4_write_sysinfo(stdout);
        MPI_Finalize();
        return 0;
    }

    if (g_rank == 0) {
        printf("Assignment 4 benchmark harness (C + MPI)\n");
        printf("Ranks (workers): %d | Warm-up runs: %d | Measured reps: %d | "
               "Per-run budget: %.0f s | Tiles: %d rows x %d\n",
               g_size, g_warmups, g_reps, g_budget, g_row_tile, g_block);
        printf("EcoQoS opt-out (rank 0): %s | p = 1 on P-cores only: %s\n",
               ecoqos ? "on" : "FAILED",
               (g_size == 1 && g_pin_single_to_pcores) ? "yes" : "no");
    }

    if (user_matrices_dir) g_matrix_dirs[g_matrix_dirs_count++] = user_matrices_dir;
    g_matrix_dirs[g_matrix_dirs_count++] = A4_MATRICES_DIR_A4;
    g_matrix_dirs[g_matrix_dirs_count++] = A4_MATRICES_DIR_A1;

    /* Correctness gate: always first, always mandatory (collective). */
    if (!a4_validate_all(MPI_COMM_WORLD, part)) {
        if (g_rank == 0) printf("Correctness validation FAILED. Aborting.\n");
        MPI_Finalize();
        return 1;
    }
    if (validate_only) {
        MPI_Finalize();
        return 0;
    }

    /* Optional standalone result dump for the external NumPy cross-check. */
    if (dump_n > 0 && dump_path) {
        DenseMatrix *A = NULL, *B = NULL, *C = NULL;
        int ready = 1;
        if (g_rank == 0) {
            A = load_ab(dump_n, 'A');
            B = load_ab(dump_n, 'B');
            C = (A && B) ? dm_alloc(dump_n) : NULL;
            ready = (C != NULL);
            if (!ready) printf("Cannot dump result: matrices for n = %d not found.\n", dump_n);
        }
        MPI_Bcast(&ready, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int rc = 0;
        if (ready) {
            void *ctx = part->create(MPI_COMM_WORLD, dump_n);
            PhaseTimes t;
            int ok = part->multiply(ctx, g_rank == 0 ? A->data : NULL,
                                    g_rank == 0 ? B->data : NULL,
                                    g_rank == 0 ? C->data : NULL, g_row_tile,
                                    g_block, 0.0, &t);
            part->destroy(ctx);
            if (g_rank == 0) {
                int saved = ok && dm_save_csv(C, dump_path);
                printf(saved ? "Result C (%d x %d, p = %d ranks) written to %s\n"
                             : "Failed to write result (%d x %d, p = %d) to %s\n",
                       dump_n, dump_n, g_size, dump_path);
                rc = saved ? 0 : 1;
            }
        } else {
            rc = 1;
        }
        dm_free(A); dm_free(B); dm_free(C);
        MPI_Finalize();
        return rc;
    }

    FILE *out = NULL;
    if (g_rank == 0) {
        CreateDirectoryA("assignment4", NULL);
        CreateDirectoryA("assignment4\\data", NULL);
        CreateDirectoryA(A4_RESULTS_DIR, NULL);

        char out_buf[512];
        if (!out_path) {
            char timestamp[32];
            time_t now = time(NULL);
            strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
            snprintf(out_buf, sizeof(out_buf), "%s\\results_a4_%s.csv",
                     A4_RESULTS_DIR, timestamp);
            out_path = out_buf;
        }
        /* Append so one file collects the launches of all worker counts. */
        FILE *probe = fopen(out_path, "r");
        int is_new = 1;
        if (probe) {
            fseek(probe, 0, SEEK_END);
            is_new = (ftell(probe) == 0);
            fclose(probe);
        }
        out = fopen(out_path, "a");
        if (!out) {
            printf("Failed to open output file %s\n", out_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (is_new) csv_header(out);

        char cross_path[560];
        size_t len = strlen(out_path);
        if (len > 4 && strcmp(out_path + len - 4, ".csv") == 0) {
            snprintf(cross_path, sizeof(cross_path), "%.*s_crosscheck.csv",
                     (int)(len - 4), out_path);
        } else {
            snprintf(cross_path, sizeof(cross_path), "%s_crosscheck.csv", out_path);
        }
        FILE *cprobe = fopen(cross_path, "r");
        int cross_new = 1;
        if (cprobe) {
            fseek(cprobe, 0, SEEK_END);
            cross_new = (ftell(cprobe) == 0);
            fclose(cprobe);
        }
        g_cross_file = fopen(cross_path, "a");
        if (g_cross_file && cross_new) {
            fprintf(g_cross_file, "assignment,experiment,partitioner,workers,n,"
                                  "max_abs_diff,bitwise_equal,status\n");
        }
        printf("Results file: %s\nCross-check log: %s\n", out_path, cross_path);
    }

    double t_start = a4_time_sec();
    run_scaling(part, out);
    double t_total = a4_time_sec() - t_start;

    if (g_rank == 0) {
        fclose(out);
        if (g_cross_file) fclose(g_cross_file);
        printf("\nBenchmark finished in %.1f s (p = %d).\n", t_total, g_size);
    }
    MPI_Finalize();
    return 0;
}
