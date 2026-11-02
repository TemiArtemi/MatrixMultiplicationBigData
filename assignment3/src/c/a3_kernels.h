/**
 * a3_kernels.h - Tiled dense kernel variants with a uniform contract.
 *
 * One source file (a3_kernels.c) is compiled four times so that the
 * sequential baseline and the parallel implementation are the SAME
 * algorithm by construction:
 *
 *   variant  simd   compiler flags
 *   seq      vec    -O3 -march=native
 *   par      vec    -O3 -march=native -fopenmp
 *   seq      novec  -O3 -march=native -fno-tree-vectorize
 *   par      novec  -O3 -march=native -fno-tree-vectorize -fopenmp
 *
 * Uniform kernel contract:
 *   - A and B are row-major n x n double arrays; C receives A x B.
 *   - The kernel zeroes C itself INSIDE the timed region (each row tile is
 *     zeroed by the thread that computes it); allocation is done by the
 *     harness outside the timer.
 *   - Every output element accumulates its k terms in ascending k order,
 *     whatever the thread count, so all thread counts give bit-identical C.
 *   - If deadline > 0 and monotonic time exceeds it, the kernel aborts,
 *     returns 0 (timeout) and C must be treated as invalid. Returns 1 on
 *     completion. The deadline is checked once per row tile.
 *   - The parallel variants use the thread count set by the caller with
 *     omp_set_num_threads(); the sequential variants ignore it.
 */

#ifndef A3_KERNELS_H
#define A3_KERNELS_H

typedef int (*TiledKernelFn)(const double *A, const double *B, double *C,
                             int n, int row_tile, int block, double deadline);

typedef struct {
    const char *name;    /* method label written to the results CSV */
    const char *variant; /* "seq" or "par"                           */
    const char *simd;    /* "vec" or "novec"                         */
    int threaded;        /* 1 when the variant honours the thread count */
    TiledKernelFn fn;
} KernelVariant;

/* The four compiled instances of the tiled kernel. */
int k_tiled_seq_vec(const double *A, const double *B, double *C, int n,
                    int row_tile, int block, double deadline);
int k_tiled_par_vec(const double *A, const double *B, double *C, int n,
                    int row_tile, int block, double deadline);
int k_tiled_seq_novec(const double *A, const double *B, double *C, int n,
                      int row_tile, int block, double deadline);
int k_tiled_par_novec(const double *A, const double *B, double *C, int n,
                      int row_tile, int block, double deadline);

/* Variant registry used by the harness (OCP: new variants are added here). */
const KernelVariant *a3_kernel_variants(int *count);
const KernelVariant *a3_find_variant(const char *variant, const char *simd);

#endif /* A3_KERNELS_H */
