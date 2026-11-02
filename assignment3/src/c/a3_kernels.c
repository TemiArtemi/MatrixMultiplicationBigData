/**
 * a3_kernels.c - Tiled i-k-j kernel (Assignment 2 cache blocking, reused).
 *
 * This file is compiled once per variant with -DA3_KERNEL_SUFFIX=<suffix>
 * (see a3_kernels.h and scripts/build.ps1). Without -fopenmp the pragmas
 * are ignored and the code is the sequential baseline; with -fopenmp the
 * row-tile loop is shared among threads.
 *
 * Change with respect to the Assignment 2 kernel: tiles are rectangular,
 * row_tile rows x block columns of k and j. With square 512 x 512 tiles a
 * 2000 x 2000 product has only 4 row tiles, which cannot keep 28 threads
 * busy; small row tiles provide enough independent work units while the
 * k/j tiling keeps the Assignment 2 cache behaviour. The loop nest order
 * (row tile, kk, jj, i, k, j) and therefore the per-element accumulation
 * order are the same as in Assignment 2.
 *
 * Race freedom: each row tile writes only its own rows of C, and A and B
 * are read-only, so no locks or atomics are needed in the computation.
 */

#include "a3_kernels.h"
#include "a3_timer.h"

#include <string.h>

#ifndef A3_KERNEL_SUFFIX
#error "A3_KERNEL_SUFFIX must be defined (e.g. -DA3_KERNEL_SUFFIX=_seq_vec)"
#endif

#define A3_CAT_(a, b) a##b
#define A3_CAT(a, b) A3_CAT_(a, b)
#define A3_KERNEL_NAME A3_CAT(k_tiled, A3_KERNEL_SUFFIX)

int A3_KERNEL_NAME(const double *A, const double *B, double *C, int n,
                   int row_tile, int block, double deadline) {
    int bi = row_tile < 1 ? 1 : row_tile;
    int bs = block < 1 ? 1 : block;
    int tiles = (n + bi - 1) / bi;
    int timed_out = 0;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < tiles; t++) {
        int stop;
        #pragma omp atomic read
        stop = timed_out;
        if (stop) continue;
        if (deadline > 0.0 && a3_time_sec() > deadline) {
            #pragma omp atomic write
            timed_out = 1;
            continue;
        }

        int ii = t * bi;
        int imax = ii + bi < n ? ii + bi : n;
        memset(C + (size_t)ii * n, 0,
               (size_t)(imax - ii) * (size_t)n * sizeof(double));

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
    return !timed_out;
}
