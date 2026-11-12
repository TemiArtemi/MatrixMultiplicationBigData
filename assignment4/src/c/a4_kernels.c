/**
 * a4_kernels.c - Tiled i-k-j kernel on a row block (Assignment 3 kernel
 * without OpenMP).
 *
 * Loop nest per row tile: (kk, jj, i, k, j), identical to Assignment 3, so
 * each C[i][j] accumulates its n products in ascending k order whatever the
 * row-block boundaries are.
 */

#include "a4_kernels.h"
#include "a4_timer.h"

#include <string.h>

int a4_tiled_rows(const double *A_block, const double *B, double *C_block,
                  int rows, int n, int row_tile, int block, double deadline) {
    int bi = row_tile < 1 ? 1 : row_tile;
    int bs = block < 1 ? 1 : block;

    for (int ii = 0; ii < rows; ii += bi) {
        if (deadline > 0.0 && a4_time_sec() > deadline) return 0;
        int imax = ii + bi < rows ? ii + bi : rows;
        memset(C_block + (size_t)ii * n, 0,
               (size_t)(imax - ii) * (size_t)n * sizeof(double));

        for (int kk = 0; kk < n; kk += bs) {
            int kmax = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++) {
                    const double *Ai = A_block + (size_t)i * n;
                    double *Ci = C_block + (size_t)i * n;
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
