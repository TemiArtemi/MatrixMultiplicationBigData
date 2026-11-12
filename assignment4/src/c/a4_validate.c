/**
 * a4_validate.c - Correctness validation of the distributed product.
 *
 * Tests (all run through the Partitioner on the launched rank count p):
 *   1. A x I = A and A x 0 = 0 for n in {1, 2, 4, 8, 16}.
 *   2. A known 2 x 2 product.
 *   3. Agreement with an independent k-i-j reference for
 *      n in {1, 2, 4, 8, 16, 17, 33, 100}, with the default tiles and with
 *      tiny tiles (row tile 3, block 8) that exercise every remainder path.
 *      Small n with large p yields ranks that own zero rows (p > n) and
 *      n not divisible by p yields uneven Scatterv/Gatherv counts.
 *   4. Equivalence: the distributed result must be bit-identical to the
 *      single-block result of the same kernel (memcmp), because every
 *      output element is accumulated in the same order whatever the split.
 *
 * Tolerances: A4_ABS_TOL / A4_REL_TOL (1e-9 / 1e-9), as in Assignment 3.
 * Rank 0 generates the inputs and judges the results; the verdict is
 * broadcast so all ranks return the same value.
 */

#include "a4_validate.h"
#include "a4_config.h"
#include "a4_matrix.h"
#include "a4_kernels.h"

#include <stdio.h>
#include <string.h>

/* Independent reference kernel (k-i-j accumulation order), validation only. */
static void reference_kij(const double *A, const double *B, double *C, int n) {
    size_t total = (size_t)n * (size_t)n;
    memset(C, 0, total * sizeof(double));
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            double aik = A[(size_t)i * n + k];
            const double *Bk = B + (size_t)k * n;
            double *Ci = C + (size_t)i * n;
            for (int j = 0; j < n; j++) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

static int g_failures = 0;
static int g_checks = 0;
static int g_rank = 0;

static void check(int ok, const char *label) {
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("   FAIL: %s\n", label);
    }
}

static int within_tol(const DenseMatrix *got, const DenseMatrix *expected) {
    int ok = 0;
    dm_max_abs_diff(got, expected, &ok, A4_ABS_TOL, A4_REL_TOL);
    return ok;
}

static void poison(DenseMatrix *C) {
    size_t total = (size_t)C->n * (size_t)C->n;
    for (size_t k = 0; k < total; k++) C->data[k] = 12345.678;
}

/* Collective helper: one distributed product. Only rank 0 supplies matrices;
 * C is poisoned beforehand so a missing gather or zeroing is detected. */
static int distributed(MPI_Comm comm, const Partitioner *part, int n,
                       const DenseMatrix *A, const DenseMatrix *B,
                       DenseMatrix *C, int row_tile, int block) {
    void *ctx = part->create(comm, n);
    if (g_rank == 0) poison(C);
    PhaseTimes t;
    int ok = part->multiply(ctx, g_rank == 0 ? A->data : NULL,
                            g_rank == 0 ? B->data : NULL,
                            g_rank == 0 ? C->data : NULL, row_tile, block, 0.0,
                            &t);
    part->destroy(ctx);
    return ok;
}

int a4_validate_all(MPI_Comm comm, const Partitioner *part) {
    int size;
    MPI_Comm_rank(comm, &g_rank);
    MPI_Comm_size(comm, &size);
    g_failures = g_checks = 0;
    const int root = (g_rank == 0);
    char label[160];

    if (root) {
        printf("============================================================\n");
        printf("CORRECTNESS VALIDATION (Assignment 4, %s, p = %d ranks)\n",
               part->name, size);
        printf("============================================================\n");
    }

    /* ---- 1. Identity and zero matrices -------------------------------- */
    int small[5] = {1, 2, 4, 8, 16};
    for (int s = 0; s < 5; s++) {
        int n = small[s];
        DenseMatrix *A = NULL, *I = NULL, *Z = NULL, *C = NULL;
        if (root) {
            unsigned int seed = A4_SEED;
            A = dm_generate_uniform(n, &seed);
            I = dm_identity(n);
            Z = dm_zero_matrix(n);
            C = dm_alloc(n);
        }
        distributed(comm, part, n, A, I, C, A4_DEFAULT_ROW_TILE, A4_DEFAULT_BLOCK);
        if (root) {
            snprintf(label, sizeof(label), "p=%d n=%d A x I = A", size, n);
            check(within_tol(C, A), label);
        }
        distributed(comm, part, n, A, Z, C, A4_DEFAULT_ROW_TILE, A4_DEFAULT_BLOCK);
        if (root) {
            snprintf(label, sizeof(label), "p=%d n=%d A x 0 = 0", size, n);
            check(within_tol(C, Z), label);
            dm_free(A); dm_free(I); dm_free(Z); dm_free(C);
        }
    }

    /* ---- 2. Known 2 x 2 product ---------------------------------------- */
    {
        DenseMatrix *A = NULL, *B = NULL, *E = NULL, *C = NULL;
        if (root) {
            A = dm_alloc(2); B = dm_alloc(2); E = dm_alloc(2); C = dm_alloc(2);
            double a[4] = {1.0, 2.0, 3.0, 4.0};
            double b[4] = {5.0, 6.0, 7.0, 8.0};
            double e[4] = {19.0, 22.0, 43.0, 50.0};
            memcpy(A->data, a, sizeof(a));
            memcpy(B->data, b, sizeof(b));
            memcpy(E->data, e, sizeof(e));
        }
        distributed(comm, part, 2, A, B, C, A4_DEFAULT_ROW_TILE, A4_DEFAULT_BLOCK);
        if (root) {
            snprintf(label, sizeof(label), "p=%d known 2x2", size);
            check(within_tol(C, E), label);
            dm_free(A); dm_free(B); dm_free(E); dm_free(C);
        }
    }

    /* ---- 3-4. Reference agreement and bitwise equivalence -------------- */
    int sizes[8] = {1, 2, 4, 8, 16, 17, 33, 100};
    int tiles[2][2] = {{A4_DEFAULT_ROW_TILE, A4_DEFAULT_BLOCK}, {3, 8}};
    for (int s = 0; s < 8; s++) {
        int n = sizes[s];
        DenseMatrix *A = NULL, *B = NULL, *ref = NULL, *single = NULL, *C = NULL;
        if (root) {
            unsigned int seed = A4_SEED + (unsigned int)n;
            A = dm_generate_uniform(n, &seed);
            B = dm_generate_uniform(n, &seed);
            ref = dm_alloc(n);
            single = dm_alloc(n);
            C = dm_alloc(n);
            reference_kij(A->data, B->data, ref->data, n);
        }
        for (int tl = 0; tl < 2; tl++) {
            int bi = tiles[tl][0], bs = tiles[tl][1];
            if (root) {
                a4_tiled_rows(A->data, B->data, single->data, n, n, bi, bs, 0.0);
            }
            distributed(comm, part, n, A, B, C, bi, bs);
            if (root) {
                snprintf(label, sizeof(label),
                         "p=%d n=%d tiles=%dx%d vs independent reference", size,
                         n, bi, bs);
                check(within_tol(C, ref), label);
                snprintf(label, sizeof(label),
                         "p=%d n=%d tiles=%dx%d bit-identical to single block",
                         size, n, bi, bs);
                check(memcmp(C->data, single->data,
                             (size_t)n * (size_t)n * sizeof(double)) == 0, label);
            }
        }
        if (root) {
            dm_free(A); dm_free(B); dm_free(ref); dm_free(single); dm_free(C);
        }
    }

    int passed = (g_failures == 0);
    MPI_Bcast(&passed, 1, MPI_INT, 0, comm);
    if (root) {
        printf("%d checks, %d failures -> %s\n", g_checks, g_failures,
               passed ? "ALL CORRECTNESS TESTS PASSED"
                      : "SOME CORRECTNESS TESTS FAILED");
    }
    return passed;
}
