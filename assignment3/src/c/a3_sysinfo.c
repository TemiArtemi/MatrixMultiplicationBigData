/**
 * a3_sysinfo.c - Implementation of the runtime fingerprint.
 */

#include "a3_sysinfo.h"

#include <windows.h>
#include <omp.h>
#include <stdlib.h>

static int popcount_mask(KAFFINITY mask) {
    int count = 0;
    while (mask) {
        count += (int)(mask & 1);
        mask >>= 1;
    }
    return count;
}

int a3_cpu_topology(CpuTopology *t) {
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

    /* First pass: highest efficiency class (= the most performant cores). */
    int max_class = 0;
    for (DWORD off = 0; off < len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
        if (info->Processor.EfficiencyClass > max_class) {
            max_class = info->Processor.EfficiencyClass;
        }
        off += info->Size;
    }

    /* Second pass: count cores and logical processors per class. */
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

static const char *env_or_unset(const char *name) {
    const char *v = getenv(name);
    return (v && *v) ? v : "(unset)";
}

static const char *proc_bind_name(omp_proc_bind_t b) {
    switch (b) {
        case omp_proc_bind_false:  return "false";
        case omp_proc_bind_true:   return "true";
        case omp_proc_bind_primary: return "primary";
        case omp_proc_bind_close:  return "close";
        case omp_proc_bind_spread: return "spread";
        default:                   return "unknown";
    }
}

void a3_write_sysinfo(FILE *f) {
    CpuTopology t;
    if (a3_cpu_topology(&t)) {
        fprintf(f, "physical_cores,%d\n", t.physical_cores);
        fprintf(f, "logical_processors,%d\n", t.logical_processors);
        fprintf(f, "performance_cores,%d\n", t.perf_cores);
        fprintf(f, "performance_logical_processors,%d\n", t.perf_logical);
        fprintf(f, "efficiency_cores,%d\n", t.eff_cores);
        fprintf(f, "efficiency_logical_processors,%d\n", t.eff_logical);
    } else {
        fprintf(f, "cpu_topology,unavailable\n");
    }

    fprintf(f, "openmp_version,%d\n", _OPENMP);
    fprintf(f, "omp_num_procs,%d\n", omp_get_num_procs());
    fprintf(f, "omp_max_threads_default,%d\n", omp_get_max_threads());
    fprintf(f, "omp_dynamic,%d\n", omp_get_dynamic());
    fprintf(f, "omp_proc_bind,%s\n", proc_bind_name(omp_get_proc_bind()));
    fprintf(f, "omp_num_places,%d\n", omp_get_num_places());
    fprintf(f, "env_OMP_NUM_THREADS,%s\n", env_or_unset("OMP_NUM_THREADS"));
    fprintf(f, "env_OMP_DYNAMIC,%s\n", env_or_unset("OMP_DYNAMIC"));
    fprintf(f, "env_OMP_PROC_BIND,%s\n", env_or_unset("OMP_PROC_BIND"));
    fprintf(f, "env_OMP_PLACES,%s\n", env_or_unset("OMP_PLACES"));
    fprintf(f, "env_OMP_WAIT_POLICY,%s\n", env_or_unset("OMP_WAIT_POLICY"));
    fprintf(f, "env_OPENBLAS_NUM_THREADS,%s\n", env_or_unset("OPENBLAS_NUM_THREADS"));
    fprintf(f, "env_MKL_NUM_THREADS,%s\n", env_or_unset("MKL_NUM_THREADS"));

    /* SIMD capabilities the kernels were allowed to target (-march=native). */
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
