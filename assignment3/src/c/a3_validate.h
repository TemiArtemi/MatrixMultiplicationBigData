/**
 * a3_validate.h - Correctness gate for Assignment 3.
 */

#ifndef A3_VALIDATE_H
#define A3_VALIDATE_H

/* Runs every correctness test for every kernel variant and, for the
 * parallel variants, every thread count in threads[0..thread_count).
 * Returns 1 when all tests pass, 0 otherwise. The harness refuses to start
 * benchmarking when this gate fails. */
int a3_validate_all(const int *threads, int thread_count);

#endif /* A3_VALIDATE_H */
