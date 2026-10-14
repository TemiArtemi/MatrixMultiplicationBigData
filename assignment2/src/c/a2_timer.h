/**
 * a2_timer.h - Monotonic timer and peak resident-memory sampler.
 *
 * Timing uses clock_gettime(CLOCK_MONOTONIC), the same clock as the
 * Assignment 1 C baseline, so kernel times remain comparable across stages.
 * Memory is sampled from a lightweight background thread that polls the
 * process working set (resident memory) every few milliseconds.
 */

#ifndef A2_TIMER_H
#define A2_TIMER_H

/* Monotonic time in seconds. */
double a2_time_sec(void);

/* Background sampler of the process peak working set (Windows RSS). */
typedef struct {
    void *thread_handle; /* internal HANDLE, kept opaque here */
    volatile long stop;  /* non-zero requests the sampler to finish  */
    double peak_mb;      /* peak working set observed while running  */
} MemSampler;

/* Start sampling; returns 0 on failure (sampling is then a no-op). */
int mem_sampler_start(MemSampler *s);

/* Stop sampling and return the peak resident memory in MB. */
double mem_sampler_stop(MemSampler *s);

/* One-shot reading of the current process working set in MB. */
double mem_current_rss_mb(void);

#endif /* A2_TIMER_H */
