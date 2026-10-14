/**
 * a2_sparse.c - Sparse x sparse matrix product in CSR form (student code).
 *
 * Algorithm: Gustavson's row-wise SpGEMM [1]. For every row i of A the
 * algorithm walks its non-zeros (i, k, a_ik) and accumulates a_ik * B(k, :)
 * into a dense workspace w(:) indexed by column. The columns touched in the
 * current row are tracked with a stamp array plus a singly-linked list, so
 * clearing the workspace costs O(touched columns) instead of O(n).
 * Each finished row is sorted by column index to produce canonical CSR
 * (deterministic output, directly comparable with the dense reference).
 *
 * [1] F. G. Gustavson, "Two algorithms for multiplying sparse matrices,"
 *     BIT, vol. 18, pp. 425-434, 1978.
 *
 * Workspace (allocated inside the timed kernel, freed before returning):
 *   wacc[n] doubles, stamp[n] ints, nxt[n] ints, rowbuf[n] {int,double},
 *   plus the output arrays which grow geometrically with the product nnz.
 */

#include "a2_kernels.h"
#include "a2_timer.h"

#include <stdlib.h>

typedef struct {
    int j;
    double v;
} JV;

static int jv_cmp(const void *a, const void *b) {
    const JV *ja = (const JV *)a;
    const JV *jb = (const JV *)b;
    return ja->j - jb->j;
}

int spgemm_gustavson(const CSRMatrix *A, const CSRMatrix *B,
                     CSRMatrix **C_out, double deadline) {
    *C_out = NULL;
    int n = A->n;
    if (B->n != n || A->rowptr[n] != A->nnz || B->rowptr[n] != B->nnz) {
        return -1; /* invalid input */
    }

    double *wacc = (double *)calloc((size_t)n, sizeof(double));
    int *stamp = (int *)calloc((size_t)n, sizeof(int));
    int *nxt = (int *)malloc((size_t)n * sizeof(int));
    JV *rowbuf = (JV *)malloc((size_t)n * sizeof(JV));
    int *orowptr = (int *)malloc((size_t)(n + 1) * sizeof(int));

    int cap = 2 * n > 16 ? 2 * n : 16;
    int *ocol = (int *)malloc((size_t)cap * sizeof(int));
    double *oval = (double *)malloc((size_t)cap * sizeof(double));

    if (!wacc || !stamp || !nxt || !rowbuf || !orowptr || !ocol || !oval) {
        free(wacc); free(stamp); free(nxt); free(rowbuf);
        free(orowptr); free(ocol); free(oval);
        return -1;
    }

    orowptr[0] = 0;
    long long nnz_total = 0;

    for (int i = 0; i < n; i++) {
        if (deadline > 0.0 && a2_time_sec() > deadline) {
            free(wacc); free(stamp); free(nxt); free(rowbuf);
            free(orowptr); free(ocol); free(oval);
            return 0;
        }

        int head = -1;
        for (int p = A->rowptr[i]; p < A->rowptr[i + 1]; p++) {
            int k = A->col[p];
            double a = A->val[p];
            for (int q = B->rowptr[k]; q < B->rowptr[k + 1]; q++) {
                int j = B->col[q];
                if (stamp[j] != i + 1) {
                    stamp[j] = i + 1;
                    wacc[j] = 0.0;
                    nxt[j] = head;
                    head = j;
                }
                wacc[j] += a * B->val[q];
            }
        }

        /* Collect, sort and append the row to the output arrays. */
        int m = 0;
        for (int j = head; j != -1; j = nxt[j]) {
            rowbuf[m].j = j;
            rowbuf[m].v = wacc[j];
            m++;
        }
        qsort(rowbuf, (size_t)m, sizeof(JV), jv_cmp);

        while (nnz_total + m > cap) {
            cap *= 2;
            int *ncol = (int *)realloc(ocol, (size_t)cap * sizeof(int));
            double *nval = (double *)realloc(oval, (size_t)cap * sizeof(double));
            if (!ncol || !nval) {
                free(ncol ? ncol : ocol);
                free(nval ? nval : oval);
                free(wacc); free(stamp); free(nxt); free(rowbuf); free(orowptr);
                return -1;
            }
            ocol = ncol;
            oval = nval;
        }
        for (int t = 0; t < m; t++) {
            ocol[nnz_total + t] = rowbuf[t].j;
            oval[nnz_total + t] = rowbuf[t].v;
        }
        nnz_total += m;
        orowptr[i + 1] = (int)nnz_total;
    }

    free(wacc);
    free(stamp);
    free(nxt);
    free(rowbuf);

    CSRMatrix *C = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!C) {
        free(orowptr); free(ocol); free(oval);
        return -1;
    }
    C->n = n;
    C->nnz = (int)nnz_total;
    C->rowptr = orowptr;
    C->col = ocol;
    C->val = oval;
    *C_out = C;
    return 1;
}
