/**
 * a3_sysinfo.h - Runtime fingerprint of the processor and OpenMP settings.
 *
 * Core topology is read from the operating system at run time
 * (GetLogicalProcessorInformationEx) instead of being typed by hand, so the
 * reported P-core / E-core counts are measured facts of the test machine.
 */

#ifndef A3_SYSINFO_H
#define A3_SYSINFO_H

#include <stdio.h>

typedef struct {
    int physical_cores;          /* processor cores                         */
    int logical_processors;      /* hardware execution contexts             */
    int perf_cores;              /* cores in the highest efficiency class   */
    int perf_logical;            /* logical processors on those cores       */
    int eff_cores;               /* cores in lower efficiency classes       */
    int eff_logical;             /* logical processors on those cores       */
} CpuTopology;

/* Fill *t from the OS; returns 1 on success, 0 otherwise. */
int a3_cpu_topology(CpuTopology *t);

/* Write the topology, OpenMP runtime settings and SIMD target macros as
 * "key,value" lines to f (and nothing else). */
void a3_write_sysinfo(FILE *f);

#endif /* A3_SYSINFO_H */
