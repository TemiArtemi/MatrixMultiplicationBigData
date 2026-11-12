/**
 * a4_partition.c - 1D row-block partitioning with B broadcast.
 *
 *   scatter  : rank r receives rows [first_r, first_r + rows_r) of A
 *              (MPI_Scatterv; the n mod p remainder rows go to the first
 *              ranks, so block sizes differ by at most one row);
 *   broadcast: every rank receives the whole of B (MPI_Bcast);
 *   compute  : rank r computes the same rows of C with the local kernel;
 *   gather   : rank 0 collects the row blocks of C (MPI_Gatherv).
 *
 * Rank 0 owns the full matrices and works in place: its own block is the
 * first rows of A and C (first_0 = 0), so MPI_IN_PLACE avoids a self-copy.
 * Ranks with zero rows (p > n) take part in every collective with empty
 * blocks.
 */

#include "a4_partition.h"
#include "a4_kernels.h"
#include "a4_timer.h"

#include <stdlib.h>
#include <string.h>

int a4_rows_of_rank(int rank, int size, int n) {
    int base = n / size, rem = n % size;
    return base + (rank < rem ? 1 : 0);
}

int a4_first_row_of_rank(int rank, int size, int n) {
    int base = n / size, rem = n % size;
    return rank * base + (rank < rem ? rank : rem);
}

typedef struct {
    MPI_Comm comm;
    int rank, size, n;
    int rows;            /* rows owned by this rank                       */
    double *A_block;     /* non-root: rows x n (at least 1 element)       */
    double *B_copy;      /* non-root: full B                              */
    double *C_block;     /* non-root: rows x n (at least 1 element)       */
    int *counts;         /* root: element counts per rank (rows * n)      */
    int *displs;         /* root: element displacements per rank          */
} RowBlockCtx;

static void *rb_create(MPI_Comm comm, int n) {
    RowBlockCtx *c = (RowBlockCtx *)calloc(1, sizeof(RowBlockCtx));
    c->comm = comm;
    MPI_Comm_rank(comm, &c->rank);
    MPI_Comm_size(comm, &c->size);
    c->n = n;
    c->rows = a4_rows_of_rank(c->rank, c->size, n);

    c->counts = (int *)malloc((size_t)c->size * sizeof(int));
    c->displs = (int *)malloc((size_t)c->size * sizeof(int));
    for (int r = 0; r < c->size; r++) {
        c->counts[r] = a4_rows_of_rank(r, c->size, n) * n;
        c->displs[r] = a4_first_row_of_rank(r, c->size, n) * n;
    }

    if (c->rank != 0) {
        size_t block = (size_t)c->rows * (size_t)n;
        c->A_block = (double *)malloc((block ? block : 1) * sizeof(double));
        c->C_block = (double *)malloc((block ? block : 1) * sizeof(double));
        c->B_copy = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
        /* Touch the pages now so first-touch faults are not timed. */
        memset(c->A_block, 0, (block ? block : 1) * sizeof(double));
        memset(c->C_block, 0, (block ? block : 1) * sizeof(double));
        memset(c->B_copy, 0, (size_t)n * (size_t)n * sizeof(double));
    }
    return c;
}

static void rb_destroy(void *ctx) {
    RowBlockCtx *c = (RowBlockCtx *)ctx;
    free(c->A_block);
    free(c->C_block);
    free(c->B_copy);
    free(c->counts);
    free(c->displs);
    free(c);
}

static int rb_multiply(void *ctx, const double *A, const double *B, double *C,
                       int row_tile, int block, double deadline,
                       PhaseTimes *t) {
    RowBlockCtx *c = (RowBlockCtx *)ctx;
    const int n = c->n;
    const int root = (c->rank == 0);

    /* ---- scatter A -------------------------------------------------- */
    MPI_Barrier(c->comm);
    double t0 = a4_time_sec();
    if (root) {
        MPI_Scatterv((void *)A, c->counts, c->displs, MPI_DOUBLE, MPI_IN_PLACE,
                     c->counts[0], MPI_DOUBLE, 0, c->comm);
    } else {
        MPI_Scatterv(NULL, c->counts, c->displs, MPI_DOUBLE, c->A_block,
                     c->rows * n, MPI_DOUBLE, 0, c->comm);
    }
    MPI_Barrier(c->comm);
    double t1 = a4_time_sec();

    /* ---- broadcast B ------------------------------------------------ */
    MPI_Bcast(root ? (void *)B : (void *)c->B_copy, n * n, MPI_DOUBLE, 0,
              c->comm);
    MPI_Barrier(c->comm);
    double t2 = a4_time_sec();

    /* ---- local compute ----------------------------------------------- */
    const double *a_loc = root ? A : c->A_block;
    const double *b_loc = root ? B : c->B_copy;
    double *c_loc = root ? C : c->C_block;
    double k0 = a4_time_sec();
    int ok = a4_tiled_rows(a_loc, b_loc, c_loc, c->rows, n, row_tile, block,
                           deadline);
    double kernel_time = a4_time_sec() - k0;
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, c->comm);
    double t3 = a4_time_sec();

    /* ---- gather C ----------------------------------------------------- */
    if (all_ok) {
        if (root) {
            MPI_Gatherv(MPI_IN_PLACE, c->counts[0], MPI_DOUBLE, C, c->counts,
                        c->displs, MPI_DOUBLE, 0, c->comm);
        } else {
            MPI_Gatherv(c->C_block, c->rows * n, MPI_DOUBLE, NULL, NULL, NULL,
                        MPI_DOUBLE, 0, c->comm);
        }
    }
    MPI_Barrier(c->comm);
    double t4 = a4_time_sec();

    /* Per-rank kernel spread: outside the end-to-end boundary. */
    double kmin = 0.0, kmax = 0.0;
    MPI_Reduce(&kernel_time, &kmin, 1, MPI_DOUBLE, MPI_MIN, 0, c->comm);
    MPI_Reduce(&kernel_time, &kmax, 1, MPI_DOUBLE, MPI_MAX, 0, c->comm);

    if (root && t) {
        t->t_scatter = t1 - t0;
        t->t_bcast = t2 - t1;
        t->t_compute = t3 - t2;
        t->t_gather = t4 - t3;
        t->t_total = t4 - t0;
        t->t_kernel_min = kmin;
        t->t_kernel_max = kmax;
    }
    return all_ok;
}

static double rb_storage_bytes(int rank, int size, int n) {
    double nn = (double)n * (double)n * sizeof(double);
    if (rank == 0) return 3.0 * nn;
    double block = (double)a4_rows_of_rank(rank, size, n) * (double)n *
                   sizeof(double);
    return nn + 2.0 * block;
}

static const Partitioner PARTITIONERS[] = {
    {"mpi_rowblock", rb_create, rb_multiply, rb_destroy, rb_storage_bytes},
};

const Partitioner *a4_partitioners(int *count) {
    *count = (int)(sizeof(PARTITIONERS) / sizeof(PARTITIONERS[0]));
    return PARTITIONERS;
}

const Partitioner *a4_find_partitioner(const char *name) {
    int count;
    const Partitioner *ps = a4_partitioners(&count);
    for (int i = 0; i < count; i++) {
        if (strcmp(ps[i].name, name) == 0) return &ps[i];
    }
    return NULL;
}
