/**
 * a2_matrix.h - Dense matrix representation, CSV I/O and deterministic RNG.
 *
 * Storage: contiguous 1D array in row-major order (A[i][j] == data[i*n + j]),
 * identical to the Assignment 1 C baseline so that Part A reuses the
 * validated implementation without layout confounders.
 */

#ifndef A2_MATRIX_H
#define A2_MATRIX_H

#include <stddef.h>

typedef struct {
    int n;        /* side length (n x n square matrix) */
    double *data; /* row-major storage, n*n doubles    */
} DenseMatrix;

/* Allocation / deallocation. dm_alloc does NOT zero the contents. */
DenseMatrix *dm_alloc(int n);
void dm_free(DenseMatrix *m);

/* Element access (bounds are not checked for speed). */
static inline double dm_get(const DenseMatrix *m, int i, int j) {
    return m->data[(size_t)i * (size_t)m->n + (size_t)j];
}
static inline void dm_set(DenseMatrix *m, int i, int j, double v) {
    m->data[(size_t)i * (size_t)m->n + (size_t)j] = v;
}

/* Zero the whole matrix (memset). */
void dm_zero(DenseMatrix *m);

/* Special matrices. */
DenseMatrix *dm_identity(int n);
DenseMatrix *dm_zero_matrix(int n);

/* CSV I/O (one matrix row per line, comma-separated values). */
DenseMatrix *dm_load_csv(const char *path);
int dm_save_csv(const DenseMatrix *m, const char *path);

/* ---- Deterministic RNG (32-bit LCG, identical formula to Assignment 1) --
 * state = state * 1103515245 + 12345; uniform double in [0, 1) from the
 * upper bits. Every generator below takes an explicit state pointer so the
 * harness can reseed per configuration and document the exact stream. */
double a2_rand_uniform01(unsigned int *state);

/* Random dense matrix, all entries uniform in [VALUE_MIN, VALUE_MAX]. */
DenseMatrix *dm_generate_uniform(int n, unsigned int *state);

/* Random sparse-pattern dense matrix: each entry is 0.0 with probability
 * (1 - density); otherwise uniform in [VALUE_MIN, VALUE_MAX]. The matrix is
 * stored densely so that the dense kernels and the CSR kernel consume the
 * exact same logical input. */
DenseMatrix *dm_generate_sparse_pattern(int n, double density, unsigned int *state);

/* Checksums / summary values. */
double dm_trace(const DenseMatrix *m);
double dm_frobenius_sq(const DenseMatrix *m);

/* Elementwise comparison. Returns the maximum absolute difference and sets
 * *ok to 1 when every entry satisfies |a-b| <= abs_tol + rel_tol*|b|. */
double dm_max_abs_diff(const DenseMatrix *a, const DenseMatrix *b, int *ok,
                       double abs_tol, double rel_tol);

#endif /* A2_MATRIX_H */
