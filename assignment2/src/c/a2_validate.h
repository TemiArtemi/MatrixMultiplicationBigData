/**
 * a2_validate.h - Correctness gate for Assignment 2.
 */

#ifndef A2_VALIDATE_H
#define A2_VALIDATE_H

/* Runs every correctness test (Part A dense kernels, Part B CSR product).
 * Returns 1 when all tests pass, 0 otherwise. The harness refuses to start
 * benchmarking when this gate fails. */
int a2_validate_all(void);

#endif /* A2_VALIDATE_H */
