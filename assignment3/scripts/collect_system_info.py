#!/usr/bin/env python3
"""
Collect the system fingerprint required by the common experimental protocol.

Combines operating-system queries (CPU model, RAM, cache, OS, compiler) with
the runtime fingerprint written by the C harness (sysinfo_*.csv: P-core /
E-core topology read with GetLogicalProcessorInformationEx, OpenMP runtime
settings and SIMD target macros). Writes report/environment.md and
report/tables/table_env.tex. Run it on the benchmarking machine.
"""

import csv
import glob
import platform
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE / "report"
RESULTS_DIR = BASE / "data" / "results"


def run_cmd(cmd: list[str]) -> str:
    """Run a command and return its first output line (empty on failure)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        first = (out.stdout or out.stderr).strip().splitlines()
        return first[0] if first else "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def powershell(query: str) -> str:
    """Evaluate a small PowerShell expression and return its output."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() or "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def harness_sysinfo() -> dict[str, str]:
    """Most recent sysinfo_*.csv written by a3_bench (key,value rows)."""
    files = sorted(glob.glob(str(RESULTS_DIR / "sysinfo_*.csv")))
    if not files:
        sys.exit("No sysinfo_*.csv found; run a3_bench once first.")
    with open(files[-1], newline="", encoding="utf-8") as f:
        return {row["key"]: row["value"] for row in csv.DictReader(f)}


def main() -> None:
    info = harness_sysinfo()
    cpu_model = powershell("(Get-CimInstance Win32_Processor).Name")
    total_ram_gb = powershell(
        "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"
    )
    llc_mb = powershell(
        "[math]::Round(((Get-CimInstance Win32_CacheMemory | "
        "Where-Object Level -EQ 5 | "
        "Measure-Object -Property MaxCacheSize -Maximum).Maximum)/1KB,0)"
    )
    # Power plans are reported by GUID so the value does not depend on the
    # Windows display language (the scheme names below are the built-in ones).
    plan_guid = powershell(
        "(powercfg /getactivescheme) -replace '^.*:\\s*([0-9a-fA-F-]{36}).*$','$1'"
    ).lower()
    builtin_plans = {
        "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
        "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
        "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate performance",
    }
    power_plan = f"{builtin_plans.get(plan_guid, 'custom')} ({plan_guid})"
    gcc_version = run_cmd(["gcc", "--version"])
    python_version = platform.python_version()
    # platform.release() reports "10" on Windows 11; builds >= 22000 are Windows 11.
    build = platform.version()
    release = platform.release()
    try:
        if platform.system() == "Windows" and int(build.split(".")[-1]) >= 22000:
            release = "11"
    except ValueError:
        pass
    os_version = f"{platform.system()} {release} (build {build})"

    topo = (f"{info['physical_cores']} cores / {info['logical_processors']} logical: "
            f"{info['performance_cores']} P-cores ({info['performance_logical_processors']} "
            f"logical, SMT) + {info['efficiency_cores']} E-cores "
            f"({info['efficiency_logical_processors']} logical)")
    simd = (f"AVX2={info['target_avx2']}, FMA={info['target_fma']}, "
            f"AVX-512F={info['target_avx512f']}")
    omp = (f"OpenMP {info['openmp_version']} (libgomp), dynamic adjustment off, "
           f"thread count set per run with omp_set_num_threads; OMP_PROC_BIND/"
           f"OMP_PLACES unsupported by libgomp on this platform; EcoQoS opt-out "
           f"(ecoqos_optout={info.get('ecoqos_optout', 'n/a')}); 1-thread runs "
           f"restricted to P-cores, p>=2 runs on all logical processors")

    rows = [
        ("CPU model", cpu_model),
        ("Core topology", topo),
        ("Last-level cache (L3)", f"{llc_mb} MB"),
        ("Total RAM", f"{total_ram_gb} GB"),
        ("Operating system", os_version),
        ("Power plan", power_plan),
        ("C compiler", gcc_version),
        ("Kernel flags", "-O3 -march=native [-fopenmp] [-fno-tree-vectorize]"),
        ("SIMD target", simd),
        ("Threading", omp),
        ("Library threads", "no numerical library linked; OPENBLAS/MKL variables unset"),
        ("Analysis runtime", f"CPython {python_version}"),
    ]

    lines = ["## System fingerprint (benchmarking machine)", "",
             "| Component | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = REPORT_DIR / "environment.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"(written to {md})")

    def esc(s: str) -> str:
        return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
                .replace("_", r"\_").replace("%", r"\%"))

    tex = [
        r"% Auto-generated by scripts/collect_system_info.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Benchmarking machine and toolchain.}",
        r"\label{tab:env}",
        r"\begin{tabular}{lp{0.68\linewidth}}",
        r"\toprule",
        r"Component & Value \\",
        r"\midrule",
    ]
    tex += [f"{esc(k)} & {esc(v)} \\\\" for k, v in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tables = REPORT_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "table_env.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"(written to {tables / 'table_env.tex'})")


if __name__ == "__main__":
    sys.exit(main())
