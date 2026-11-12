/**
 * a4_kernels.h - Local multiplication kernel executed by every MPI rank.
 *
 * The kernel never calls MPI: it multiplies a block of `rows` consecutive
 * rows of A by the full matrix B (single thread). This is the Assignment 3
 * tiled i-k-j kernel, generalised from n x n to a row block, compiled once
 * (-O3 -march=native, vectorised, no OpenMP).
 *
 * Contract:
 *   - A_block is rows x n, B is n x n, C_block is rows x n (row-major).
 *   - The kernel zeroes C_block itself (inside the timed region).
 *   - Every output element accumulates its k terms in ascending k order, so
 *     the result does not depend on how the rows are split among ranks: the
 *     distributed C is bit-identical to the single-rank C.
 *   - rows may be 0 (a rank that received no rows does nothing).
 *   - If deadline > 0 and monotonic time exceeds it, the kernel aborts and
 *     returns 0 (C_block invalid); it returns 1 on completion. The deadline
 *     is checked once per row tile.
 */

#ifndef A4_KERNELS_H
#define A4_KERNELS_H

int a4_tiled_rows(const double *A_block, const double *B, double *C_block,
                  int rows, int n, int row_tile, int block, double deadline);

#endif /* A4_KERNELS_H */
