/**
 * Matrix multiplication benchmark - C implementation.
 * Implements basic O(n³) triple-loop matrix multiplication.
 * Reads matrices from CSV, measures kernel time and memory.
 * Handles timeouts dynamically to avoid infinite execution on large matrices.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <psapi.h>

// Link with psapi.lib for memory measurement (MSVC only)
#ifdef _MSC_VER
#pragma comment(lib, "psapi.lib")
#endif

// Configuration
#define SEED 42
#define VALUE_MIN -10.0
#define VALUE_MAX 10.0
#define SIZES_COUNT 11
static const int SIZES[SIZES_COUNT] = {10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000};

// Updated directories
#define MATRICES_DIR "assignment1\\data\\matrices"
#define RESULTS_DIR "assignment1\\data\\results"

// Benchmark settings
#define WARMUP_RUNS 1
#define TIME_BUDGET_SECONDS 600 // 10 minutes

// Batch sizes for very small matrices
static const struct { int size; int batches; } BATCH_SIZES[] = {
    {10, 1000}, {50, 100}, {100, 10}
};

// Tolerances for correctness
#define ABS_TOL 1e-9
#define REL_TOL 1e-9

// Global RNG state
static unsigned int rng_state = SEED;

// Helper to get time in seconds as double
static double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Simple RNG (LCG) for reproducibility
static double rand_double() {
    rng_state = rng_state * 1103515245 + 12345;
    return VALUE_MIN + (VALUE_MAX - VALUE_MIN) * (rng_state / 4294967296.0);
}

// Matrix structure
typedef struct {
    int n;
    double *data;  // 1D array: row-major order [i*n + j]
} Matrix;

// Allocate matrix
static Matrix* matrix_alloc(int n) {
    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->n = n;
    m->data = malloc(n * n * sizeof(double));
    if (!m->data) {
        free(m);
        return NULL;
    }
    return m;
}

// Free matrix
static void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

// Access element (row-major)
static inline double matrix_get(const Matrix *m, int i, int j) {
    return m->data[i * m->n + j];
}

static inline void matrix_set(Matrix *m, int i, int j, double val) {
    m->data[i * m->n + j] = val;
}

// Load matrix from CSV
static Matrix* load_matrix_csv(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f) return NULL;
    
    // First pass: count rows and columns
    int n = 0, cols = 0;
    char line[1024 * 1024];  // Large buffer for big matrices
    while (fgets(line, sizeof(line), f)) {
        if (n == 0) {
            char *tok = strtok(line, ",");
            while (tok) { cols++; tok = strtok(NULL, ","); }
        }
        n++;
    }
    
    rewind(f);
    
    Matrix *m = matrix_alloc(n);
    if (!m) { fclose(f); return NULL; }
    
    for (int i = 0; i < n && fgets(line, sizeof(line), f); i++) {
        char *tok = strtok(line, ",");
        for (int j = 0; j < cols && tok; j++) {
            matrix_set(m, i, j, atof(tok));
            tok = strtok(NULL, ",");
        }
    }
    
    fclose(f);
    return m;
}

// Basic triple-loop matrix multiplication: C = A × B
// Returns 1 if completed, 0 if deadline exceeded
static int multiply(const Matrix *A, const Matrix *B, Matrix *C, double deadline) {
    int n = A->n;
    for (int i = 0; i < n; i++) {
        // Zero-overhead timeout check
        if (deadline > 0.0 && get_time_sec() > deadline) {
            return 0; // Timeout
        }
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += matrix_get(A, i, k) * matrix_get(B, k, j);
            }
            matrix_set(C, i, j, s);
        }
    }
    return 1;
}

// Run multiplication multiple times (for very small matrices)
static int multiply_batch(const Matrix *A, const Matrix *B, Matrix *C, int batches, double deadline) {
    for (int b = 0; b < batches; b++) {
        if (deadline > 0.0 && get_time_sec() > deadline) {
            return 0;
        }
        // Inner call uses 0.0 to ensure zero overhead
        multiply(A, B, C, 0.0);
    }
    return 1;
}

// Check if two matrices are equal within tolerances
static int matrices_equal(const Matrix *A, const Matrix *B, double abs_tol, double rel_tol) {
    if (A->n != B->n) return 0;
    int n = A->n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double a = matrix_get(A, i, j);
            double b = matrix_get(B, i, j);
            double diff = fabs(a - b);
            double max_ab = fmax(fabs(a), fabs(b));
            if (diff > abs_tol && diff > rel_tol * max_ab) {
                return 0;
            }
        }
    }
    return 1;
}

// Create identity matrix
static Matrix* identity_matrix(int n) {
    Matrix *m = matrix_alloc(n);
    if (!m) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_set(m, i, j, (i == j) ? 1.0 : 0.0);
        }
    }
    return m;
}

// Create zero matrix
static Matrix* zero_matrix(int n) {
    Matrix *m = matrix_alloc(n);
    if (!m) return NULL;
    memset(m->data, 0, n * n * sizeof(double));
    return m;
}

// Generate random matrix
static Matrix* generate_random_matrix(int n) {
    Matrix *m = matrix_alloc(n);
    if (!m) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_set(m, i, j, rand_double());
        }
    }
    return m;
}

// Reference implementation with different loop order (k, i, j)
static Matrix* multiply_reference(const Matrix *A, const Matrix *B) {
    int n = A->n;
    Matrix *C = matrix_alloc(n);
    if (!C) return NULL;
    memset(C->data, 0, n * n * sizeof(double));
    
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            double aik = matrix_get(A, i, k);
            for (int j = 0; j < n; j++) {
                double val = matrix_get(C, i, j) + aik * matrix_get(B, k, j);
                matrix_set(C, i, j, val);
            }
        }
    }
    return C;
}

// Get current process memory in MB (Windows)
static double get_memory_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}

// Get batch size for a given n
static int get_batch_size(int n) {
    for (size_t i = 0; i < sizeof(BATCH_SIZES)/sizeof(BATCH_SIZES[0]); i++) {
        if (BATCH_SIZES[i].size == n) return BATCH_SIZES[i].batches;
    }
    return 1;
}

// Benchmark a specific matrix size
static int benchmark_size(int n, FILE *csv_out) {
    // 1. Repeticiones dinámicas
    int reps_to_run = (n <= 800) ? 5 : 2;
    printf("\n  Benchmarking n=%d (reps=%d, budget=%ds)...\n", n, reps_to_run, TIME_BUDGET_SECONDS);
    
    char a_path[256], b_path[256];
    snprintf(a_path, sizeof(a_path), "%s\\A_%d.csv", MATRICES_DIR, n);
    snprintf(b_path, sizeof(b_path), "%s\\B_%d.csv", MATRICES_DIR, n);
    
    Matrix *A = load_matrix_csv(a_path);
    Matrix *B = load_matrix_csv(b_path);
    
    if (!A || !B) {
        printf("    Matrices not found for n=%d, skipping\n", n);
        if (A) matrix_free(A);
        if (B) matrix_free(B);
        return 0;
    }
    
    // Pre-allocate result matrix (outside timing)
    Matrix *C = matrix_alloc(n);
    if (!C) {
        printf("    Failed to allocate result matrix\n");
        matrix_free(A);
        matrix_free(B);
        return 0;
    }
    
    // 2. Warm-up runs with deadline
    printf("    Warm-up (%d runs)...\n", WARMUP_RUNS);
    double warmup_start = get_time_sec();
    double warmup_deadline = warmup_start + TIME_BUDGET_SECONDS;
    
    for (int i = 0; i < WARMUP_RUNS; i++) {
        if (!multiply(A, B, C, warmup_deadline)) {
            printf("    [TIMEOUT] Time limit exceeded during warm-up! Aborting size n=%d.\n", n);
            matrix_free(A); matrix_free(B); matrix_free(C);
            return 0;
        }
    }
    
    // Determine if we need batching
    int batches = get_batch_size(n);
    
    // Timed runs
    for (int rep = 0; rep < reps_to_run; rep++) {
        // Clear memory outside timer to guarantee valid result
        memset(C->data, 0, n * n * sizeof(double));
        
        // Measure memory before
        double mem_before = get_memory_mb();
        
        // Time the kernel
        double start_t = get_time_sec();
        
        if (batches > 1) {
            multiply_batch(A, B, C, batches, 0.0);
        } else {
            multiply(A, B, C, 0.0);
        }
        
        double end_t = get_time_sec();
        
        // Measure memory after
        double mem_after = get_memory_mb();
        
        double elapsed_ms = (end_t - start_t) * 1000.0;
        if (batches > 1) {
            elapsed_ms /= batches;
        }
        
        double memory_mb = fmax(mem_after - mem_before, 0);
        
        // Write to CSV
        fprintf(csv_out, "C,%d,float64,%d,%.2f,%.2f\n", n, rep + 1, elapsed_ms, memory_mb);
        
        printf("    Rep %d: %.2f ms, memory delta: %.2f MB\n", rep + 1, elapsed_ms, memory_mb);
        
        // Check time budget
        if ((end_t - start_t) > TIME_BUDGET_SECONDS) {
            printf("    Time budget exceeded, stopping further repetitions for this size\n");
            break;
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
    return 1;
}

// Correctness validation
static int validate_correctness() {
    printf("\n============================================================\n");
    printf("CORRECTNESS VALIDATION\n");
    printf("============================================================\n");
    
    int all_passed = 1;
    
    // Test 1: Identity matrix (A × I = A)
    printf("\n1. Testing with identity matrix...\n");
    rng_state = SEED;
    int n = 5;
    Matrix *A = generate_random_matrix(n);
    Matrix *I = identity_matrix(n);
    Matrix *C = matrix_alloc(n);
    
    // Use 0.0 to disable timeout during validation
    multiply(A, I, C, 0.0);
    if (matrices_equal(C, A, ABS_TOL, REL_TOL)) {
        printf("   PASS: A x I = A\n");
    } else {
        printf("   FAIL: A x I != A\n");
        all_passed = 0;
    }
    matrix_free(A);
    matrix_free(I);
    matrix_free(C);
    
    // Test 2: Zero matrix (A × 0 = 0)
    printf("\n2. Testing with zero matrix...\n");
    rng_state = SEED;
    A = generate_random_matrix(n);
    Matrix *Z = zero_matrix(n);
    C = matrix_alloc(n);
    
    multiply(A, Z, C, 0.0);
    Matrix *expected = zero_matrix(n);
    if (matrices_equal(C, expected, ABS_TOL, REL_TOL)) {
        printf("   PASS: A x 0 = 0\n");
    } else {
        printf("   FAIL: A x 0 != 0\n");
        all_passed = 0;
    }
    matrix_free(A);
    matrix_free(Z);
    matrix_free(C);
    matrix_free(expected);
    
    // Test 3: Small known matrices
    printf("\n3. Testing with known small matrices...\n");
    Matrix *A_test = matrix_alloc(2);
    Matrix *B_test = matrix_alloc(2);
    Matrix *expected_test = matrix_alloc(2);
    
    matrix_set(A_test, 0, 0, 1.0); matrix_set(A_test, 0, 1, 2.0);
    matrix_set(A_test, 1, 0, 3.0); matrix_set(A_test, 1, 1, 4.0);
    matrix_set(B_test, 0, 0, 5.0); matrix_set(B_test, 0, 1, 6.0);
    matrix_set(B_test, 1, 0, 7.0); matrix_set(B_test, 1, 1, 8.0);
    matrix_set(expected_test, 0, 0, 19.0); matrix_set(expected_test, 0, 1, 22.0);
    matrix_set(expected_test, 1, 0, 43.0); matrix_set(expected_test, 1, 1, 50.0);
    
    C = matrix_alloc(2);
    multiply(A_test, B_test, C, 0.0);
    if (matrices_equal(C, expected_test, ABS_TOL, REL_TOL)) {
        printf("   PASS: 2x2 known result\n");
    } else {
        printf("   FAIL: 2x2 known result differs\n");
        all_passed = 0;
    }
    matrix_free(A_test);
    matrix_free(B_test);
    matrix_free(expected_test);
    matrix_free(C);
    
    // Test 4: Reference comparison
    printf("\n4. Testing against reference implementation...\n");
    rng_state = SEED;
    n = 20;
    A = generate_random_matrix(n);
    Matrix *B = generate_random_matrix(n);
    C = matrix_alloc(n);
    multiply(A, B, C, 0.0);
    
    Matrix *C_ref = multiply_reference(A, B);
    if (matrices_equal(C, C_ref, ABS_TOL, REL_TOL)) {
        printf("   PASS: Matches reference implementation\n");
    } else {
        printf("   FAIL: Differs from reference implementation\n");
        all_passed = 0;
    }
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
    matrix_free(C_ref);
    
    printf("\n============================================================\n");
    if (all_passed) {
        printf("ALL CORRECTNESS TESTS PASSED\n");
    } else {
        printf("SOME CORRECTNESS TESTS FAILED\n");
    }
    printf("\n============================================================\n");
    
    return all_passed;
}

int main(int argc, char *argv[]) {
    int validate_only = 0;
    int *custom_sizes = NULL;
    int custom_count = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate-only") == 0) {
            validate_only = 1;
        } else if (strcmp(argv[i], "--sizes") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                custom_sizes = realloc(custom_sizes, (custom_count + 1) * sizeof(int));
                custom_sizes[custom_count++] = atoi(argv[++i]);
            }
        }
    }
    
    // Create results directory
    CreateDirectory(RESULTS_DIR, NULL);
    
    printf("C Matrix Multiplication Benchmark\n");
    printf("Matrices directory: %s\n", MATRICES_DIR);
    printf("Results directory: %s\n", RESULTS_DIR);
    printf("Sizes: ");
    for (int i = 0; i < SIZES_COUNT; i++) printf("%d ", SIZES[i]);
    printf("\n");
    printf("Warm-up runs: %d\n", WARMUP_RUNS);
    printf("Repetitions: 5 for n<=800, 2 for n>=1000\n");
    printf("Time budget: %ds per size\n", TIME_BUDGET_SECONDS);
    
    // Run correctness validation first
    if (!validate_correctness()) {
        printf("Correctness validation failed. Aborting benchmark.\n");
        free(custom_sizes);
        return 1;
    }
    
    if (validate_only) {
        free(custom_sizes);
        return 0;
    }
    
    // Open results CSV
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
    
    char output_file[512];
    snprintf(output_file, sizeof(output_file), "%s\\results_c_%s.csv", RESULTS_DIR, timestamp);
    
    FILE *csv_out = fopen(output_file, "w");
    if (!csv_out) {
        printf("Failed to open output file\n");
        free(custom_sizes);
        return 1;
    }
    
    fprintf(csv_out, "language,size,numeric_type,repetition,time_ms,memory_mb\n");
    
    // Run benchmarks
    const int *sizes = custom_sizes ? custom_sizes : SIZES;
    int count = custom_sizes ? custom_count : SIZES_COUNT;
    
    for (int i = 0; i < count; i++) {
        benchmark_size(sizes[i], csv_out);
    }
    
    fclose(csv_out);
    free(custom_sizes);
    
    printf("\nResults saved to: %s\n", output_file);
    printf("\nBenchmark complete.\n");
    
    return 0;
}