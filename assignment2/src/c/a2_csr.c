/**
 * a2_csr.c - CSR representation and dense/sparse conversions (student code).
 */

#include "a2_csr.h"

#include <stdlib.h>
#include <string.h>

CSRMatrix *csr_alloc(int n, int nnz) {
    CSRMatrix *m = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!m) return NULL;
    m->n = n;
    m->nnz = nnz;
    m->rowptr = (int *)malloc((size_t)(n + 1) * sizeof(int));
    m->col = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    m->val = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    if (!m->rowptr || !m->col || !m->val) {
        csr_free(m);
        return NULL;
    }
    return m;
}

void csr_free(CSRMatrix *m) {
    if (m) {
        free(m->rowptr);
        free(m->col);
        free(m->val);
        free(m);
    }
}

size_t csr_storage_bytes(const CSRMatrix *m) {
    return (size_t)(m->n + 1) * sizeof(int) +
           (size_t)m->nnz * sizeof(int) +
           (size_t)m->nnz * sizeof(double);
}

long long dm_count_nnz(const DenseMatrix *m) {
    long long count = 0;
    size_t total = (size_t)m->n * (size_t)m->n;
    for (size_t k = 0; k < total; k++) {
        if (m->data[k] != 0.0) count++;
    }
    return count;
}

typedef struct {
    int r;
    int c;
    double v;
} Triplet;

static int triplet_cmp(const void *a, const void *b) {
    const Triplet *ta = (const Triplet *)a;
    const Triplet *tb = (const Triplet *)b;
    if (ta->r != tb->r) return ta->r - tb->r;
    return ta->c - tb->c;
}

CSRMatrix *csr_from_dense(const DenseMatrix *m) {
    int n = m->n;
    long long nnz_ll = dm_count_nnz(m);

    if (nnz_ll > 2000000000LL) {
        return NULL; /* refuse pathological allocations */
    }
    int nnz = (int)nnz_ll;

    CSRMatrix *c = csr_alloc(n, nnz);
    if (!c) return NULL;

    Triplet *trip = (Triplet *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(Triplet));
    if (!trip) {
        csr_free(c);
        return NULL;
    }

    long long t = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double v = dm_get(m, i, j);
            if (v != 0.0) {
                trip[t].r = i;
                trip[t].c = j;
                trip[t].v = v;
                t++;
            }
        }
    }

    qsort(trip, (size_t)nnz, sizeof(Triplet), triplet_cmp);

    int idx = 0;
    for (int i = 0; i <= n; i++) {
        c->rowptr[i] = idx;
        while (idx < nnz && trip[idx].r == i) {
            c->col[idx] = trip[idx].c;
            c->val[idx] = trip[idx].v;
            idx++;
        }
    }

    free(trip);
    return c;
}

DenseMatrix *csr_to_dense(const CSRMatrix *m) {
    DenseMatrix *d = dm_zero_matrix(m->n);
    if (!d) return NULL;
    for (int i = 0; i < m->n; i++) {
        for (int p = m->rowptr[i]; p < m->rowptr[i + 1]; p++) {
            dm_set(d, i, m->col[p], m->val[p]);
        }
    }
    return d;
}

double csr_trace(const CSRMatrix *m) {
    double t = 0.0;
    for (int i = 0; i < m->n; i++) {
        for (int p = m->rowptr[i]; p < m->rowptr[i + 1]; p++) {
            if (m->col[p] == i) {
                t += m->val[p];
            }
        }
    }
    return t;
}
