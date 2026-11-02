/**
 * a3_validate.c - Correctness validation for the Assignment 3 kernels.
 *
 * Tests (run for all four variants; parallel variants for every thread
 * count of the gate, including counts larger than the number of tiles):
 *   1. A x I = A and A x 0 = 0 for n in {1, 2, 4, 8, 16}.
 *   2. Known 2 x 2 product.
 *   3. Agreement with an independent k-i-j reference kernel for
 *      n in {1, 2, 4, 8, 16, 17, 33, 100}, both with the default tiles and
 *      with tiny tiles (row_tile 3, block 8) that exercise every remainder.
 *   4. Race-freedom / equivalence: the parallel result must be bit-identical
 *      to the sequential result of the same build (memcmp), because every
 *      output element is accumulated in the same order by one thread.
 *   5. vec vs novec builds agree within tolerance (bitwise equality is
 *      reported but not required).
 *
 * Tolerances: A3_ABS_TOL / A3_REL_TOL (1e-9 / 1e-9), as in Assignment 2.
 */

#include "a3_validate.h"
#include "a3_config.h"
#include "a3_matrix.h"
#include "a3_kernels.h"

#include <omp.h>
#include <stdio.h>
#include <string.h>

/* Independent reference kernel (k-i-j accumulation order), used only for
 * validation, never benchmarked. */
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

static void check(int ok, const char *label) {
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("   FAIL: %s\n", label);
    }
}

static int within_tol(const DenseMatrix *got, const DenseMatrix *expected) {
    int ok = 0;
    dm_max_abs_diff(got, expected, &ok, A3_ABS_TOL, A3_REL_TOL);
    return ok;
}

/* Run variant v with p threads (ignored by sequential variants). */
static void run_variant(const KernelVariant *v, int p, const DenseMatrix *A,
                        const DenseMatrix *B, DenseMatrix *C, int row_tile,
                        int block) {
    omp_set_num_threads(p);
    v->fn(A->data, B->data, C->data, A->n, row_tile, block, 0.0);
}

/* Fill C with garbage so a kernel that forgets to zero C is detected. */
static void poison(DenseMatrix *C) {
    size_t total = (size_t)C->n * (size_t)C->n;
    for (size_t k = 0; k < total; k++) C->data[k] = 12345.678;
}

int a3_validate_all(const int *threads, int thread_count) {
    printf("============================================================\n");
    printf("CORRECTNESS VALIDATION (Assignment 3)\n");
    printf("============================================================\n");

    int vcount;
    const KernelVariant *vs = a3_kernel_variants(&vcount);
    char label[160];
    g_failures = g_checks = 0;

    /* ---- 1. Identity and zero matrices -------------------------------- */
    printf("\n1. A x I = A and A x 0 = 0, n in {1,2,4,8,16}...\n");
    {
        int sizes[5] = {1, 2, 4, 8, 16};
        for (int s = 0; s < 5; s++) {
            int n = sizes[s];
            unsigned int seed = A3_SEED;
            DenseMatrix *A = dm_generate_uniform(n, &seed);
            DenseMatrix *I = dm_identity(n);
            DenseMatrix *Z = dm_zero_matrix(n);
            DenseMatrix *C = dm_alloc(n);
            for (int m = 0; m < vcount; m++) {
                int tc = vs[m].threaded ? thread_count : 1;
                for (int t = 0; t < tc; t++) {
                    int p = vs[m].threaded ? threads[t] : 1;
                    poison(C);
                    run_variant(&vs[m], p, A, I, C, A3_DEFAULT_ROW_TILE,
                                A3_DEFAULT_BLOCK);
                    snprintf(label, sizeof(label), "%s p=%d n=%d A x I = A",
                             vs[m].name, p, n);
                    check(within_tol(C, A), label);
                    poison(C);
                    run_variant(&vs[m], p, A, Z, C, A3_DEFAULT_ROW_TILE,
                                A3_DEFAULT_BLOCK);
                    snprintf(label, sizeof(label), "%s p=%d n=%d A x 0 = 0",
                             vs[m].name, p, n);
                    check(within_tol(C, Z), label);
                }
            }
            dm_free(A); dm_free(I); dm_free(Z); dm_free(C);
        }
    }
    printf("   done (%d checks so far, %d failures)\n", g_checks, g_failures);

    /* ---- 2. Known 2 x 2 product ---------------------------------------- */
    printf("\n2. Known 2x2 product...\n");
    {
        DenseMatrix *A = dm_alloc(2), *B = dm_alloc(2), *E = dm_alloc(2);
        DenseMatrix *C = dm_alloc(2);
        double a[4] = {1.0, 2.0, 3.0, 4.0};
        double b[4] = {5.0, 6.0, 7.0, 8.0};
        double e[4] = {19.0, 22.0, 43.0, 50.0};
        memcpy(A->data, a, sizeof(a));
        memcpy(B->data, b, sizeof(b));
        memcpy(E->data, e, sizeof(e));
        for (int m = 0; m < vcount; m++) {
            int p = vs[m].threaded ? threads[thread_count - 1] : 1;
            poison(C);
            run_variant(&vs[m], p, A, B, C, A3_DEFAULT_ROW_TILE, A3_DEFAULT_BLOCK);
            snprintf(label, sizeof(label), "%s p=%d known 2x2", vs[m].name, p);
            check(within_tol(C, E), label);
        }
        dm_free(A); dm_free(B); dm_free(E); dm_free(C);
    }
    printf("   done (%d checks so far, %d failures)\n", g_checks, g_failures);

    /* ---- 3-5. Reference, bitwise seq/par equality, vec vs novec -------- */
    printf("\n3-5. Reference agreement, seq/par bitwise equality, vec vs novec\n"
           "     (n in {1,2,4,8,16,17,33,100}; default and tiny tiles)...\n");
    {
        int sizes[8] = {1, 2, 4, 8, 16, 17, 33, 100};
        int tiles[2][2] = {{A3_DEFAULT_ROW_TILE, A3_DEFAULT_BLOCK}, {3, 8}};
        int bitwise_vec_novec = 1;
        for (int s = 0; s < 8; s++) {
            int n = sizes[s];
            unsigned int seed = A3_SEED + (unsigned int)n;
            DenseMatrix *A = dm_generate_uniform(n, &seed);
            DenseMatrix *B = dm_generate_uniform(n, &seed);
            DenseMatrix *ref = dm_alloc(n);
            reference_kij(A->data, B->data, ref->data, n);
            DenseMatrix *Cseq = dm_alloc(n);
            DenseMatrix *Cpar = dm_alloc(n);
            DenseMatrix *Cvec = dm_alloc(n);
            size_t bytes = (size_t)n * (size_t)n * sizeof(double);

            for (int tl = 0; tl < 2; tl++) {
                int bi = tiles[tl][0], bs = tiles[tl][1];
                const char *simds[2] = {"vec", "novec"};
                for (int sm = 0; sm < 2; sm++) {
                    const KernelVariant *seqv = a3_find_variant("seq", simds[sm]);
                    const KernelVariant *parv = a3_find_variant("par", simds[sm]);

                    poison(Cseq);
                    run_variant(seqv, 1, A, B, Cseq, bi, bs);
                    snprintf(label, sizeof(label), "%s n=%d tiles=%dx%d vs reference",
                             seqv->name, n, bi, bs);
                    check(within_tol(Cseq, ref), label);

                    for (int t = 0; t < thread_count; t++) {
                        int p = threads[t];
                        poison(Cpar);
                        run_variant(parv, p, A, B, Cpar, bi, bs);
                        snprintf(label, sizeof(label),
                                 "%s p=%d n=%d tiles=%dx%d bit-identical to %s",
                                 parv->name, p, n, bi, bs, seqv->name);
                        check(memcmp(Cpar->data, Cseq->data, bytes) == 0, label);
                    }

                    if (sm == 0) {
                        memcpy(Cvec->data, Cseq->data, bytes);
                    } else {
                        snprintf(label, sizeof(label),
                                 "novec vs vec n=%d tiles=%dx%d within tolerance",
                                 n, bi, bs);
                        check(within_tol(Cseq, Cvec), label);
                        if (memcmp(Cseq->data, Cvec->data, bytes) != 0) {
                            bitwise_vec_novec = 0;
                        }
                    }
                }
            }
            dm_free(A); dm_free(B); dm_free(ref);
            dm_free(Cseq); dm_free(Cpar); dm_free(Cvec);
        }
        printf("   vec and novec builds bit-identical on every test: %s\n",
               bitwise_vec_novec ? "yes" : "no (agree within tolerance)");
    }

    printf("\n============================================================\n");
    printf("%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) {
        printf("ALL CORRECTNESS TESTS PASSED\n");
    } else {
        printf("SOME CORRECTNESS TESTS FAILED\n");
    }
    printf("============================================================\n");
    return g_failures == 0;
}
