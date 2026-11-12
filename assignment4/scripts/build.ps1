# build.ps1 - Build the Assignment 4 MPI harness (run from the repository root).
#
# Requirements: MSYS2 GCC (ucrt64) with gendef/dlltool, and the Microsoft MPI
# runtime + SDK (winget: Microsoft.msmpi, Microsoft.msmpisdk).
#
# GCC cannot link the Visual-Studio import library msmpi.lib, so an MinGW
# import library (libmsmpi.a) is generated once from the installed msmpi.dll
# with gendef + dlltool into assignment4\build.

$ErrorActionPreference = "Stop"

$src = "assignment4\src\c"
$build = "assignment4\build"
New-Item -ItemType Directory -Force $build | Out-Null

$sdk = "C:\Program Files (x86)\Microsoft SDKs\MPI"
$incMpi = "$sdk\Include"
$incMpiArch = "$sdk\Include\x64"
$dll = "$env:SystemRoot\System32\msmpi.dll"
if (-not (Test-Path "$incMpi\mpi.h")) { throw "MS-MPI SDK not found at $sdk" }
if (-not (Test-Path $dll)) { throw "msmpi.dll not found (MS-MPI runtime missing)" }

$implib = "$build\libmsmpi.a"
if (-not (Test-Path $implib)) {
    Push-Location $build
    & gendef $dll
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "gendef failed" }
    & dlltool -d msmpi.def -l libmsmpi.a -D msmpi.dll
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "dlltool failed" }
    Pop-Location
    Write-Host "Generated $implib"
}

# Kernel: performance critical, vectorised, no OpenMP (one thread per rank).
$kernelFlags = @("-O3", "-march=native", "-Wall", "-Wextra",
                 "-fopt-info-vec-optimized=assignment4\data\results\vec_report_kernel.txt")
& gcc @kernelFlags -c "$src\a4_kernels.c" -o "$build\a4_kernels.o"
if ($LASTEXITCODE -ne 0) { throw "kernel build failed" }

# Everything else: -O2, MPI headers.
$common = @("-O2", "-march=native", "-Wall", "-Wextra", "-I$incMpi", "-I$incMpiArch")
$support = @("a4_timer", "a4_matrix", "a4_affinity", "a4_partition", "a4_sysinfo",
             "a4_validate", "a4_bench")
foreach ($m in $support) {
    & gcc @common -c "$src\$m.c" -o "$build\$m.o"
    if ($LASTEXITCODE -ne 0) { throw "build failed ($m)" }
}

$objects = @("$build\a4_kernels.o") + ($support | ForEach-Object { "$build\$_.o" })
$exe = "$src\a4_bench.exe"
& gcc -o $exe @objects "-L$build" -lmsmpi -lpsapi
if ($LASTEXITCODE -ne 0) { throw "link failed" }
Write-Host "Built $exe"
