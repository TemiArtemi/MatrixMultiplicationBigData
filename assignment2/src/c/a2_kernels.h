/**
 * a2_kernels.h - Dense multiplication kernels with a uniform contract.
 *
 * Uniform kernel contract (substitutability across methods):
 *   - Inputs A and B are row-major n x n double arrays; C receives A x B.
 *   - The kernel must produce a correct C regardless of C's previous
 *     contents (kernels that accumulate zero C themselves INSIDE the timed
 *     region; allocation is always performed by the harness outside it).
 *   - No value-dependent shortcuts (e.g. skipping zero entries) are allowed,
 *     so every dense kernel performs exactly the same arithmetic work on the
 *     same logical inputs.
 *   - If deadline > 0 and monotonic time exceeds it, the kernel aborts,
 *     returns 0 (timeout) and C must be treated as invalid. Returns 1 on
 *     completion. The deadline is only checked once per outer iteration so
 *     the overhead is O(n) against O(n^3) work.
 */

#ifndef A2_KERNELS_H
#define A2_KERNELS_H

#include "a2_csr.h"

typedef int (*DenseKernelFn)(const double *A, const double *B, double *C,
                             int n, double deadline);

typedef struct {
    const char *name;    /* method label written to the results CSV */
    DenseKernelFn fn;
    int uses_block;      /* 1 when the kernel honours g_block_size  */
} DenseKernel;

/* Baseline: textbook i-j-k loop order (validated Assignment 1 baseline). */
int k_ijk(const double *A, const double *B, double *C, int n, double deadline);

/* Optimisation 1: i-k-j loop reordering (row-wise streaming of B). */
int k_ikj(const double *A, const double *B, double *C, int n, double deadline);

/* Optimisation 2: cache blocking (tiled i-k-j, block size = g_block_size). */
int k_blocked(const double *A, const double *B, double *C, int n, double deadline);

/* Method registry used by the harness (OCP: new kernels are added here). */
const DenseKernel *a2_dense_kernels(int *count);
const DenseKernel *a2_find_kernel(const char *name);

/* Sparse SpGEMM: C_csr = A_csr x B_csr with Gustavson's algorithm and a
 * dense accumulation workspace (implemented in a2_sparse.c). Returns 1 on
 * completion, 0 when the deadline is exceeded (then *C_out is freed and set
 * to NULL). On success *C_out owns a canonical CSR matrix (sorted columns). */
int spgemm_gustavson(const CSRMatrix *A, const CSRMatrix *B,
                     CSRMatrix **C_out, double deadline);

#endif /* A2_KERNELS_H */
