/**
 * a3_registry.c - Registry of the compiled kernel variants.
 */

#include "a3_kernels.h"

#include <string.h>

static const KernelVariant VARIANTS[] = {
    {"tiled_seq_vec",   "seq", "vec",   0, k_tiled_seq_vec},
    {"tiled_par_vec",   "par", "vec",   1, k_tiled_par_vec},
    {"tiled_seq_novec", "seq", "novec", 0, k_tiled_seq_novec},
    {"tiled_par_novec", "par", "novec", 1, k_tiled_par_novec},
};

const KernelVariant *a3_kernel_variants(int *count) {
    *count = (int)(sizeof(VARIANTS) / sizeof(VARIANTS[0]));
    return VARIANTS;
}

const KernelVariant *a3_find_variant(const char *variant, const char *simd) {
    int count;
    const KernelVariant *vs = a3_kernel_variants(&count);
    for (int i = 0; i < count; i++) {
        if (strcmp(vs[i].variant, variant) == 0 && strcmp(vs[i].simd, simd) == 0) {
            return &vs[i];
        }
    }
    return NULL;
}
