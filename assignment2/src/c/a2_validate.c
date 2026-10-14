/**
 * a2_validate.c - Correctness validation for the Assignment 2 kernels.
 *
 * Every test compares against a trusted result: a known small product, an
 * algebraic identity (identity / zero matrices), an independent reference
 * kernel with a different loop order (k-i-j), or the validated dense
 * baseline evaluated on the same logical input.
 *
 * Tolerances: A2_ABS_TOL / A2_REL_TOL (1e-9 / 1e-9) for the small tests;
 * the worst-case rounding error of an O(n) dot-product accumulation stays
 * far below these bounds for the tested sizes.
 */

#include "a2_validate.h"
#include "a2_config.h"
#include "a2_matrix.h"
#include "a2_csr.h"
#include "a2_kernels.h"

#include <stdio.h>
#include <stdlib.h>
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

static int compare_dense(const DenseMatrix *got, const DenseMatrix *expected,
                         const char *label) {
    int ok = 0;
    double diff = dm_max_abs_diff(got, expected, &ok, A2_ABS_TOL, A2_REL_TOL);
    if (ok) {
        printf("   PASS: %s (max |diff| = %.3e)\n", label, diff);
        return 1;
    }
    printf("   FAIL: %s (max |diff| = %.3e)\n", label, diff);
    return 0;
}

/* ---- Part A: dense kernels --------------------------------------------- */

static int validate_dense(void) {
    int passed = 1;
    unsigned int seed = A2_SEED;
    int kcount;
    const DenseKernel *ks = a2_dense_kernels(&kcount);

    /* Test A1: A x I = A (baseline kernel). */
    printf("\n[A] 1. Identity matrix: A x I = A (n = 5, baseline kernel)...\n");
    {
        int n = 5;
        DenseMatrix *A = dm_generate_uniform(n, &seed);
        DenseMatrix *I = dm_identity(n);
        DenseMatrix *C = dm_alloc(n);
        ks[0].fn(A->data, I->data, C->data, n, 0.0);
        passed &= compare_dense(C, A, "A x I = A");
        dm_free(A); dm_free(I); dm_free(C);
    }

    /* Test A2: A x 0 = 0 (baseline kernel). */
    printf("\n[A] 2. Zero matrix: A x 0 = 0 (n = 5, baseline kernel)...\n");
    {
        int n = 5;
        seed = A2_SEED;
        DenseMatrix *A = dm_generate_uniform(n, &seed);
        DenseMatrix *Z = dm_zero_matrix(n);
        DenseMatrix *C = dm_alloc(n);
        DenseMatrix *expected = dm_zero_matrix(n);
        ks[0].fn(A->data, Z->data, C->data, n, 0.0);
        passed &= compare_dense(C, expected, "A x 0 = 0");
        dm_free(A); dm_free(Z); dm_free(C); dm_free(expected);
    }

    /* Test A3: known 2x2 product, all dense kernels. */
    printf("\n[A] 3. Known 2x2 product, every dense kernel...\n");
    {
        double a[4] = {1.0, 2.0, 3.0, 4.0};
        double b[4] = {5.0, 6.0, 7.0, 8.0};
        double expected[4] = {19.0, 22.0, 43.0, 50.0};
        double c[4];
        for (int m = 0; m < kcount; m++) {
            ks[m].fn(a, b, c, 2, 0.0);
            int ok = 1;
            for (int t = 0; t < 4; t++) {
                double diff = c[t] - expected[t];
                if (diff < 0) diff = -diff;
                if (diff > A2_ABS_TOL) ok = 0;
            }
            printf("   %s: %s\n", ks[m].name, ok ? "PASS" : "FAIL");
            passed &= ok;
        }
    }

    /* Test A4: every dense kernel agrees with the independent k-i-j
     * reference on odd (non-multiple of block) and block-multiple sizes. */
    printf("\n[A] 4. Dense kernels vs independent k-i-j reference (n = 33, 64, 65)...\n");
    {
        int sizes[3] = {33, 64, 65};
        for (int s = 0; s < 3; s++) {
            int n = sizes[s];
            DenseMatrix *A = dm_generate_uniform(n, &seed);
            DenseMatrix *B = dm_generate_uniform(n, &seed);
            DenseMatrix *ref = dm_alloc(n);
            reference_kij(A->data, B->data, ref->data, n);
            DenseMatrix *C = dm_alloc(n);
            for (int m = 0; m < kcount; m++) {
                ks[m].fn(A->data, B->data, C->data, n, 0.0);
                char label[128];
                snprintf(label, sizeof(label), "%s vs reference (n = %d)",
                         ks[m].name, n);
                passed &= compare_dense(C, ref, label);
            }
            dm_free(A); dm_free(B); dm_free(ref); dm_free(C);
        }
    }

    return passed;
}

/* ---- Part B: sparse CSR product ---------------------------------------- */

static int validate_sparse(void) {
    int passed = 1;
    unsigned int seed = A2_SEED;

    /* Test B1: CSR round-trip dense -> CSR -> dense is lossless. */
    printf("\n[B] 1. CSR round-trip: dense -> CSR -> dense (n = 40, density 0.3)...\n");
    {
        int n = 40;
        DenseMatrix *A = dm_generate_sparse_pattern(n, 0.3, &seed);
        CSRMatrix *S = csr_from_dense(A);
        DenseMatrix *A2 = csr_to_dense(S);
        int ok = 0;
        double diff = dm_max_abs_diff(A, A2, &ok, 0.0, 0.0);
        printf("   %s: %s (max |diff| = %.3e, nnz = %d)\n",
               ok && diff == 0.0 ? "PASS" : "FAIL", "CSR round-trip lossless",
               diff, S->nnz);
        passed &= (ok && diff == 0.0);
        dm_free(A); dm_free(A2); csr_free(S);
    }

    /* Test B2: Gustavson product equals the dense baseline product. */
    printf("\n[B] 2. CSR product vs dense baseline (n = 40, densities 0.05/0.3/0.7/1.0)...\n");
    {
        int n = 40;
        double densities[4] = {0.05, 0.3, 0.7, 1.0};
        for (int d = 0; d < 4; d++) {
            seed = A2_SEED;
            DenseMatrix *A = dm_generate_sparse_pattern(n, densities[d], &seed);
            DenseMatrix *B = dm_generate_sparse_pattern(n, densities[d], &seed);

            DenseMatrix *Cdense = dm_alloc(n);
            k_ijk(A->data, B->data, Cdense->data, n, 0.0);

            CSRMatrix *Sa = csr_from_dense(A);
            CSRMatrix *Sb = csr_from_dense(B);
            CSRMatrix *Sc = NULL;
            int rc = spgemm_gustavson(Sa, Sb, &Sc, 0.0);
            DenseMatrix *Csparse = (rc == 1) ? csr_to_dense(Sc) : NULL;

            char label[128];
            snprintf(label, sizeof(label), "Gustavson vs ijk (density %.2f)",
                     densities[d]);
            if (Csparse) {
                passed &= compare_dense(Csparse, Cdense, label);
            } else {
                printf("   FAIL: %s (spgemm returned %d)\n", label, rc);
                passed = 0;
            }

            dm_free(A); dm_free(B); dm_free(Cdense); dm_free(Csparse);
            csr_free(Sa); csr_free(Sb); csr_free(Sc);
        }
    }

    /* Test B3: I(sparse) x S = S and 0(sparse) x S = 0. */
    printf("\n[B] 3. Sparse identity and zero products (n = 32, density 0.5)...\n");
    {
        int n = 32;
        seed = A2_SEED;
        DenseMatrix *D = dm_generate_sparse_pattern(n, 0.5, &seed);
        CSRMatrix *S = csr_from_dense(D);

        DenseMatrix *Id = dm_identity(n);
        CSRMatrix *Si = csr_from_dense(Id);
        CSRMatrix *P1 = NULL;
        spgemm_gustavson(Si, S, &P1, 0.0);
        DenseMatrix *P1d = csr_to_dense(P1);
        passed &= compare_dense(P1d, D, "I(sparse) x S = S");

        DenseMatrix *Zd = dm_zero_matrix(n);
        CSRMatrix *Sz = csr_from_dense(Zd);
        CSRMatrix *P2 = NULL;
        spgemm_gustavson(Sz, S, &P2, 0.0);
        DenseMatrix *P2d = csr_to_dense(P2);
        int ok = 0;
        double diff = dm_max_abs_diff(P2d, Zd, &ok, A2_ABS_TOL, A2_REL_TOL);
        if (ok && P2->nnz == 0) {
            printf("   PASS: 0(sparse) x S = 0 (product nnz = %d)\n", P2->nnz);
        } else {
            printf("   FAIL: 0(sparse) x S = 0 (nnz = %d, max |diff| = %.3e)\n",
                   P2->nnz, diff);
            passed = 0;
        }

        dm_free(D); dm_free(Id); dm_free(Zd); dm_free(P1d); dm_free(P2d);
        csr_free(S); csr_free(Si); csr_free(Sz); csr_free(P1); csr_free(P2);
    }

    return passed;
}

int a2_validate_all(void) {
    printf("============================================================\n");
    printf("CORRECTNESS VALIDATION (Assignment 2)\n");
    printf("============================================================\n");

    int passed = 1;
    passed &= validate_dense();
    passed &= validate_sparse();

    printf("\n============================================================\n");
    if (passed) {
        printf("ALL CORRECTNESS TESTS PASSED\n");
    } else {
        printf("SOME CORRECTNESS TESTS FAILED\n");
    }
    printf("============================================================\n");
    return passed;
}
