/**
 * a2_timer.c - Implementation of the monotonic timer and memory sampler.
 */

#include "a2_timer.h"

#include <time.h>
#include <windows.h>
#include <psapi.h>

double a2_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

double mem_current_rss_mb(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}

#define A2_SAMPLER_INTERVAL_MS 5

static void record_peak(MemSampler *s) {
    double rss = mem_current_rss_mb();
    if (rss > s->peak_mb) {
        s->peak_mb = rss;
    }
}

static DWORD WINAPI sampler_thread(LPVOID arg) {
    MemSampler *s = (MemSampler *)arg;
    while (!s->stop) {
        record_peak(s);
        Sleep(A2_SAMPLER_INTERVAL_MS);
    }
    /* One final reading in case the last interval was missed. */
    record_peak(s);
    return 0;
}

int mem_sampler_start(MemSampler *s) {
    s->stop = 0;
    s->peak_mb = 0.0;
    s->thread_handle = CreateThread(NULL, 0, sampler_thread, s, 0, NULL);
    if (!s->thread_handle) {
        s->thread_handle = NULL;
        return 0;
    }
    return 1;
}

double mem_sampler_stop(MemSampler *s) {
    if (!s->thread_handle) {
        return mem_current_rss_mb();
    }
    InterlockedExchange(&s->stop, 1);
    WaitForSingleObject((HANDLE)s->thread_handle, 5000);
    CloseHandle((HANDLE)s->thread_handle);
    s->thread_handle = NULL;
    return s->peak_mb;
}
