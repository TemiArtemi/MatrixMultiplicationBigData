/**
 * a2_matrix.c - Dense matrix operations, CSV I/O and deterministic RNG.
 */

#include "a2_matrix.h"
#include "a2_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DenseMatrix *dm_alloc(int n) {
    DenseMatrix *m = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    if (!m) return NULL;
    m->n = n;
    m->data = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!m->data) {
        free(m);
        return NULL;
    }
    return m;
}

void dm_free(DenseMatrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void dm_zero(DenseMatrix *m) {
    memset(m->data, 0, (size_t)m->n * (size_t)m->n * sizeof(double));
}

DenseMatrix *dm_identity(int n) {
    DenseMatrix *m = dm_alloc(n);
    if (!m) return NULL;
    dm_zero(m);
    for (int i = 0; i < n; i++) {
        dm_set(m, i, i, 1.0);
    }
    return m;
}

DenseMatrix *dm_zero_matrix(int n) {
    DenseMatrix *m = dm_alloc(n);
    if (!m) return NULL;
    dm_zero(m);
    return m;
}

DenseMatrix *dm_load_csv(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    /* First pass: count rows and columns. */
    int n = 0, cols = 0;
    size_t cap = 2u * 1024u * 1024u; /* 2 MiB line buffer (n up to ~4000) */
    char *line = (char *)malloc(cap);
    if (!line) {
        fclose(f);
        return NULL;
    }

    while (fgets(line, (int)cap, f)) {
        if (n == 0) {
            /* Count fields of the first line: commas + 1 (trailing newline
             * and carriage return are stripped before testing emptiness). */
            int commas = 0;
            char *p = line;
            while (*p) {
                if (*p == ',') commas++;
                p++;
            }
            int len = (int)(p - line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                len--;
            }
            cols = (len > 0) ? commas + 1 : 0;
        }
        n++;
    }

    rewind(f);

    DenseMatrix *m = dm_alloc(n);
    if (!m) {
        free(line);
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < n && fgets(line, (int)cap, f); i++) {
        char *p = line;
        for (int j = 0; j < cols; j++) {
            dm_set(m, i, j, strtod(p, &p));
            while (*p == ',') p++;
        }
    }

    free(line);
    fclose(f);
    return m;
}

int dm_save_csv(const DenseMatrix *m, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    int n = m->n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%.17g%s", dm_get(m, i, j), (j + 1 < n) ? "," : "");
        }
        fputc('\n', f);
    }
    fclose(f);
    return 1;
}

double a2_rand_uniform01(unsigned int *state) {
    *state = (*state) * 1103515245u + 12345u;
    return (double)(*state) / 4294967296.0;
}

DenseMatrix *dm_generate_uniform(int n, unsigned int *state) {
    DenseMatrix *m = dm_alloc(n);
    if (!m) return NULL;
    double span = A2_VALUE_MAX - A2_VALUE_MIN;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dm_set(m, i, j, A2_VALUE_MIN + span * a2_rand_uniform01(state));
        }
    }
    return m;
}

DenseMatrix *dm_generate_sparse_pattern(int n, double density, unsigned int *state) {
    DenseMatrix *m = dm_alloc(n);
    if (!m) return NULL;
    double span = A2_VALUE_MAX - A2_VALUE_MIN;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            /* One position draw per entry; the value draw is only consumed
             * when the entry is non-zero, so zero patterns are nested
             * across densities when the same seed restarts the stream. */
            double u = a2_rand_uniform01(state);
            if (u < density) {
                dm_set(m, i, j, A2_VALUE_MIN + span * a2_rand_uniform01(state));
            } else {
                dm_set(m, i, j, 0.0);
            }
        }
    }
    return m;
}

double dm_trace(const DenseMatrix *m) {
    int n = m->n;
    double t = 0.0;
    for (int i = 0; i < n; i++) {
        t += dm_get(m, i, i);
    }
    return t;
}

double dm_frobenius_sq(const DenseMatrix *m) {
    int n = m->n;
    double s = 0.0;
    size_t total = (size_t)n * (size_t)n;
    for (size_t k = 0; k < total; k++) {
        s += m->data[k] * m->data[k];
    }
    return s;
}

double dm_max_abs_diff(const DenseMatrix *a, const DenseMatrix *b, int *ok,
                       double abs_tol, double rel_tol) {
    *ok = 0;
    if (a->n != b->n) return -1.0;
    size_t total = (size_t)a->n * (size_t)a->n;
    double max_diff = 0.0;
    int all_ok = 1;
    for (size_t k = 0; k < total; k++) {
        double va = a->data[k];
        double vb = b->data[k];
        double diff = va - vb;
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > abs_tol && diff > rel_tol * (vb < 0 ? -vb : vb)) {
            all_ok = 0;
        }
    }
    *ok = all_ok;
    return max_diff;
}
