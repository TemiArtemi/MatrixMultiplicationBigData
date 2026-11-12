/**
 * a4_sysinfo.c - Implementation of the runtime fingerprint.
 */

#include "a4_sysinfo.h"

#include <mpi.h>
#include <windows.h>
#include <stdlib.h>
#include <string.h>

static int popcount_mask(KAFFINITY mask) {
    int count = 0;
    while (mask) {
        count += (int)(mask & 1);
        mask >>= 1;
    }
    return count;
}

int a4_cpu_topology(CpuTopology *t) {
    t->physical_cores = t->logical_processors = 0;
    t->perf_cores = t->perf_logical = t->eff_cores = t->eff_logical = 0;

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len == 0) return 0;
    char *buf = (char *)malloc(len);
    if (!buf) return 0;
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        free(buf);
        return 0;
    }

    int max_class = 0;
    for (DWORD off = 0; off < len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
        if (info->Processor.EfficiencyClass > max_class) {
            max_class = info->Processor.EfficiencyClass;
        }
        off += info->Size;
    }

    for (DWORD off = 0; off < len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
        int logical = 0;
        for (WORD g = 0; g < info->Processor.GroupCount; g++) {
            logical += popcount_mask(info->Processor.GroupMask[g].Mask);
        }
        t->physical_cores++;
        t->logical_processors += logical;
        if (info->Processor.EfficiencyClass == max_class) {
            t->perf_cores++;
            t->perf_logical += logical;
        } else {
            t->eff_cores++;
            t->eff_logical += logical;
        }
        off += info->Size;
    }
    free(buf);
    return 1;
}

static void write_sanitised(FILE *f, const char *key, const char *value) {
    /* Keep the key,value format: one line, no commas. */
    fprintf(f, "%s,", key);
    for (const char *p = value; *p && *p != '\n' && *p != '\r'; p++) {
        fputc(*p == ',' ? ';' : *p, f);
    }
    fputc('\n', f);
}

void a4_write_sysinfo(FILE *f) {
    CpuTopology t;
    if (a4_cpu_topology(&t)) {
        fprintf(f, "physical_cores,%d\n", t.physical_cores);
        fprintf(f, "logical_processors,%d\n", t.logical_processors);
        fprintf(f, "performance_cores,%d\n", t.perf_cores);
        fprintf(f, "performance_logical_processors,%d\n", t.perf_logical);
        fprintf(f, "efficiency_cores,%d\n", t.eff_cores);
        fprintf(f, "efficiency_logical_processors,%d\n", t.eff_logical);
    } else {
        fprintf(f, "cpu_topology,unavailable\n");
    }

    int major = 0, minor = 0;
    MPI_Get_version(&major, &minor);
    fprintf(f, "mpi_standard_version,%d.%d\n", major, minor);
    char lib[MPI_MAX_LIBRARY_VERSION_STRING];
    int len = 0;
    if (MPI_Get_library_version(lib, &len) == MPI_SUCCESS) {
        write_sanitised(f, "mpi_library_version", lib);
    }

    const char *threads = getenv("OMP_NUM_THREADS");
    fprintf(f, "env_OMP_NUM_THREADS,%s\n", (threads && *threads) ? threads : "(unset)");
    fprintf(f, "openmp_linked,0\n");

#ifdef __AVX2__
    fprintf(f, "target_avx2,1\n");
#else
    fprintf(f, "target_avx2,0\n");
#endif
#ifdef __FMA__
    fprintf(f, "target_fma,1\n");
#else
    fprintf(f, "target_fma,0\n");
#endif
#ifdef __AVX512F__
    fprintf(f, "target_avx512f,1\n");
#else
    fprintf(f, "target_avx512f,0\n");
#endif
    fprintf(f, "compiler,gcc %d.%d.%d\n", __GNUC__, __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
}
