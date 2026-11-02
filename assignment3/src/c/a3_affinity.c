/**
 * a3_affinity.c - Implementation of the placement controls (single
 * processor group, which covers up to 64 logical processors).
 */

#include "a3_affinity.h"

#include <windows.h>
#include <stdlib.h>

static int popcount_mask(KAFFINITY mask) {
    int count = 0;
    while (mask) {
        count += (int)(mask & 1);
        mask >>= 1;
    }
    return count;
}

int a3_disable_power_throttling(void) {
    PROCESS_POWER_THROTTLING_STATE state;
    ZeroMemory(&state, sizeof(state));
    state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0; /* control the policy and switch throttling OFF */
    return SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                 &state, sizeof(state)) ? 1 : 0;
}

/* Affinity mask of the logical processors on performance cores. */
static KAFFINITY pcore_mask(void) {
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
    KAFFINITY mask = 0;
    for (DWORD off = 0; off < len;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
        if (info->Processor.EfficiencyClass == max_class &&
            info->Processor.GroupMask[0].Group == 0) {
            mask |= info->Processor.GroupMask[0].Mask;
        }
        off += info->Size;
    }
    free(buf);
    return mask;
}

int a3_restrict_to_pcores(void) {
    KAFFINITY mask = pcore_mask();
    if (!mask || !SetProcessAffinityMask(GetCurrentProcess(), mask)) return 0;
    return popcount_mask(mask);
}

int a3_allow_all_cores(void) {
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        return 0;
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), system_mask)) return 0;
    return popcount_mask((KAFFINITY)system_mask);
}
