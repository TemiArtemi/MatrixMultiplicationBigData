/**
 * a2_kernels.c - Dense multiplication kernels (baseline and optimised).
 */

#include "a2_kernels.h"
#include "a2_config.h"
#include "a2_timer.h"

#include <string.h>

int g_block_size = A2_DEFAULT_BLOCK;

/* Zero an n x n row-major output buffer (part of the kernels' contract). */
static inline void zero_out(double *C, int n) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(double));
}

/* ---- Baseline: i-j-k loop order (Assignment 1 validated baseline) ------ */
int k_ijk(const double *A, const double *B, double *C, int n, double deadline) {
    for (int i = 0; i < n; i++) {
        if (deadline > 0.0 && a2_time_sec() > deadline) return 0;
        const double *Ai = A + (size_t)i * n;
        double *Ci = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += Ai[k] * B[(size_t)k * n + j];
            }
            Ci[j] = s;
        }
    }
    return 1;
}

/* ---- Optimisation 1: i-k-j loop reordering ------------------------------
 * The innermost loop walks B and C row-wise (unit stride), which makes both
 * arrays cache- and vectorisation-friendly; the cost is one extra pass over
 * C per k, which is O(n^2) per k level (negligible against the O(n^2) MACs). */
int k_ikj(const double *A, const double *B, double *C, int n, double deadline) {
    zero_out(C, n);
    for (int i = 0; i < n; i++) {
        if (deadline > 0.0 && a2_time_sec() > deadline) return 0;
        const double *Ai = A + (size_t)i * n;
        double *Ci = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            double aik = Ai[k];
            const double *Bk = B + (size_t)k * n;
            for (int j = 0; j < n; j++) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
    return 1;
}

/* ---- Optimisation 2: cache blocking (tiled i-k-j) -----------------------
 * The matrices are processed in block_size x block_size panels so that the
 * working set of the inner triple loop (one A panel, one B panel, one C
 * panel) stays inside the cache hierarchy; identical arithmetic work and
 * accumulation order per output element as k_ikj. */
int k_blocked(const double *A, const double *B, double *C, int n, double deadline) {
    int bs = g_block_size;
    if (bs < 1) bs = 1;
    zero_out(C, n);
    for (int ii = 0; ii < n; ii += bs) {
        if (deadline > 0.0 && a2_time_sec() > deadline) return 0;
        int imax = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int kmax = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++) {
                    const double *Ai = A + (size_t)i * n;
                    double *Ci = C + (size_t)i * n;
                    for (int k = kk; k < kmax; k++) {
                        double aik = Ai[k];
                        const double *Bk = B + (size_t)k * n;
                        for (int j = jj; j < jmax; j++) {
                            Ci[j] += aik * Bk[j];
                        }
                    }
                }
            }
        }
    }
    return 1;
}

/* ---- Registry ----------------------------------------------------------- */
static const DenseKernel KERNELS[] = {
    {"dense_ijk_baseline", k_ijk, 0},
    {"dense_ikj_reorder",  k_ikj, 0},
    {"dense_blocked",      k_blocked, 1},
};

const DenseKernel *a2_dense_kernels(int *count) {
    *count = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));
    return KERNELS;
}

const DenseKernel *a2_find_kernel(const char *name) {
    int count;
    const DenseKernel *ks = a2_dense_kernels(&count);
    for (int i = 0; i < count; i++) {
        if (strcmp(ks[i].name, name) == 0) return &ks[i];
    }
    return NULL;
}
