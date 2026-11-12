/**
 * a4_validate.h - Correctness gate for Assignment 4.
 */

#ifndef A4_VALIDATE_H
#define A4_VALIDATE_H

#include "a4_partition.h"

#include <mpi.h>

/* Collective. Runs every correctness test with the given partitioner on the
 * ranks of `comm` (the rank count is fixed by the launch, so the driver
 * script launches the gate with several rank counts, including counts larger
 * than the matrix size). Returns the same value (1 = all passed) on every
 * rank; the harness refuses to benchmark when it is 0. */
int a4_validate_all(MPI_Comm comm, const Partitioner *part);

#endif /* A4_VALIDATE_H */
