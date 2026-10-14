/**
 * a2_csr.h - Compressed Sparse Row (CSR) representation and conversions.
 *
 * CSR layout (custom student code, no external libraries):
 *   rowptr : int[n+1]   - start offset of row i in col/val (and row i ends
 *                         at rowptr[i+1]); rowptr[n] == nnz
 *   col    : int[nnz]   - column indices of the non-zero entries
 *   val    : double[nnz]- values of the non-zero entries
 *
 * Row indices are implicit; column indices inside each row are kept sorted
 * in ascending order (canonical CSR) after every product so that results are
 * deterministic and directly comparable across methods.
 */

#ifndef A2_CSR_H
#define A2_CSR_H

#include "a2_matrix.h"

typedef struct {
    int n;
    int nnz;
    int *rowptr;
    int *col;
    double *val;
} CSRMatrix;

/* Allocation for a matrix with a known nnz (arrays are NOT initialised). */
CSRMatrix *csr_alloc(int n, int nnz);
void csr_free(CSRMatrix *m);

/* Bytes stored by the CSR arrays (rowptr + col + val), excluding the struct. */
size_t csr_storage_bytes(const CSRMatrix *m);

/* Count of entries that are not exactly 0.0 (generated zeros are exact). */
long long dm_count_nnz(const DenseMatrix *m);

/* Dense -> CSR conversion (timed separately by the harness). */
CSRMatrix *csr_from_dense(const DenseMatrix *m);

/* CSR -> dense conversion (used for validation and cross-checks only). */
DenseMatrix *csr_to_dense(const CSRMatrix *m);

/* Trace of the product stored in CSR form (anti-DCE checksum). */
double csr_trace(const CSRMatrix *m);

#endif /* A2_CSR_H */
