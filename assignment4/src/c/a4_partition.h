/**
 * a4_partition.h - Distribution strategies for the distributed product.
 *
 * A Partitioner decides how A, B and C are spread over the MPI ranks and how
 * the result is collected; it calls the MPI-free local kernel
 * (a4_kernels.h) for the computation. The harness only knows this
 * interface, so a new scheme (e.g. a 2D block grid) is added by registering
 * another Partitioner without touching the harness (open/closed).
 *
 * Contract (collective: every rank of `comm` must call each function):
 *   create(comm, n)      allocate the per-rank buffers OUTSIDE the timed
 *                        region; returns an opaque context (never NULL).
 *   multiply(ctx, ...)   one full product. On rank 0, A, B and C are the
 *                        full n x n row-major matrices; on every other rank
 *                        they are NULL. Returns 1 when every rank finished,
 *                        0 when the deadline expired on some rank (C is
 *                        then invalid on rank 0). Phase times are filled on
 *                        rank 0.
 *   destroy(ctx)         free the context.
 */

#ifndef A4_PARTITION_H
#define A4_PARTITION_H

#include <mpi.h>

typedef struct {
    /* Phases measured on rank 0 between barriers, in seconds. The four
     * phases are consecutive intervals of one clock, so they sum to
     * t_total (each barrier wait is charged to the phase it closes). */
    double t_scatter;   /* distribute the row blocks of A                  */
    double t_bcast;     /* broadcast B to every rank                       */
    double t_compute;   /* slowest rank's kernel (+ completion agreement)  */
    double t_gather;    /* collect the row blocks of C on rank 0           */
    double t_total;     /* scatter start -> gather end                     */
    /* Per-rank kernel time, gathered AFTER t_total (outside the boundary).
     * Their spread measures load imbalance (e.g. P-cores versus E-cores). */
    double t_kernel_min;
    double t_kernel_max;
} PhaseTimes;

typedef struct {
    const char *name;
    void *(*create)(MPI_Comm comm, int n);
    int (*multiply)(void *ctx, const double *A, const double *B, double *C,
                    int row_tile, int block, double deadline, PhaseTimes *t);
    void (*destroy)(void *ctx);
    /* Bytes of matrix storage this rank allocates for size n (inputs and
     * outputs of the strategy, excluding harness reference buffers). */
    double (*storage_bytes)(int rank, int size, int n);
} Partitioner;

/* Registry (OCP: new strategies are added here). */
const Partitioner *a4_partitioners(int *count);
const Partitioner *a4_find_partitioner(const char *name);

/* Row split shared with the validation suite: rows owned by `rank`. */
int a4_rows_of_rank(int rank, int size, int n);
int a4_first_row_of_rank(int rank, int size, int n);

#endif /* A4_PARTITION_H */
