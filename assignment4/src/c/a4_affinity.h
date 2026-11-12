/**
 * a4_affinity.h - Process placement controls for the hybrid (P/E-core) CPU.
 *
 * Placement is controlled through the Windows API (the same policy as
 * Assignment 3, where libgomp could not pin threads on this platform):
 *   - a4_disable_power_throttling() opts the process out of EcoQoS, which
 *     otherwise lets Windows steer a non-foreground process to E-cores;
 *   - a4_restrict_to_pcores() / a4_allow_all_cores() set the process
 *     affinity mask of the calling MPI rank.
 */

#ifndef A4_AFFINITY_H
#define A4_AFFINITY_H

/* Returns 1 when the execution-speed throttling opt-out succeeded. */
int a4_disable_power_throttling(void);

/* Restrict the process to the logical processors of the performance cores
 * (highest efficiency class). Returns the number of allowed logical
 * processors, or 0 on failure. */
int a4_restrict_to_pcores(void);

/* Allow every logical processor. Returns the number allowed, 0 on failure. */
int a4_allow_all_cores(void);

#endif /* A4_AFFINITY_H */
