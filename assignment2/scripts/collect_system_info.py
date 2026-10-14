#!/usr/bin/env python3
"""
Collect the system fingerprint required by the common experimental protocol.

Writes a Markdown fragment to report/environment.md and prints it to stdout.
Run it on the benchmarking machine before the measurements so the report can
cite the exact hardware and toolchain versions.
"""

import platform
import subprocess
import sys
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"


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


def main() -> None:
    cpu_model = powershell("(Get-CimInstance Win32_Processor).Name")
    physical_cores = powershell("(Get-CimInstance Win32_Processor).NumberOfCores")
    logical_cpus = powershell("(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors")
    total_ram_gb = powershell(
        "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"
    )

    gcc_version = run_cmd(["gcc", "--version"])
    python_version = platform.python_version()
    os_version = f"{platform.system()} {platform.release()} (build {platform.version()})"

    lines = [
        "## System fingerprint (benchmarking machine)",
        "",
        "| Component | Value |",
        "| --- | --- |",
        f"| CPU model | {cpu_model} |",
        f"| Physical cores | {physical_cores} |",
        f"| Logical processors | {logical_cpus} |",
        f"| Total RAM | {total_ram_gb} GB |",
        f"| Operating system | {os_version} |",
        f"| C compiler | {gcc_version} |",
        f"| Compile flags | `-O2 -march=native -Wall -Wextra` (single-threaded, no OpenMP) |",
        f"| Analysis runtime | CPython {python_version} |",
        "| Timer | `clock_gettime(CLOCK_MONOTONIC)` (kernel time) |",
        "| Memory metric | Peak process working set (RSS), sampled every 5 ms (`GetProcessMemoryInfo`) |",
        "| Thread settings | 1 thread for every benchmark; sampler thread only reads RSS |",
        "",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "environment.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"(written to {out_path})")


if __name__ == "__main__":
    sys.exit(main())
