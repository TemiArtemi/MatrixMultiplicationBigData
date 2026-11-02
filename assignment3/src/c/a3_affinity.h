/**
 * a3_affinity.h - Process placement controls for the hybrid (P/E-core) CPU.
 *
 * libgomp does not support OMP_PROC_BIND / OMP_PLACES on Windows, so the
 * harness controls placement through the Windows API instead:
 *   - a3_disable_power_throttling() opts the process out of EcoQoS, which
 *     otherwise lets Windows steer a non-foreground process to E-cores;
 *   - a3_restrict_to_pcores() / a3_allow_all_cores() set the process
 *     affinity mask (inherited by every OpenMP worker thread).
 */

#ifndef A3_AFFINITY_H
#define A3_AFFINITY_H

/* Returns 1 when the execution-speed throttling opt-out succeeded. */
int a3_disable_power_throttling(void);

/* Restrict the process to the logical processors of the performance cores
 * (highest efficiency class). Returns the number of allowed logical
 * processors, or 0 on failure. */
int a3_restrict_to_pcores(void);

/* Allow every logical processor. Returns the number allowed, 0 on failure. */
int a3_allow_all_cores(void);

#endif /* A3_AFFINITY_H */
