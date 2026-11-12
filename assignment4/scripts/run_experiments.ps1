# run_experiments.ps1 - Full Assignment 4 measurement session (repository root).
#
# Order (everything in ONE session so thermal and OS conditions are shared):
#   1. build the harness;
#   2. baselines: the Assignment 3 sequential kernel and OpenMP kernel at the
#      same sizes and worker counts (a3_bench.exe, built by
#      assignment3\scripts\build.ps1);
#   3. MPI start-up cost: launcher wall clock of "mpiexec -n p a4_bench --sysinfo"
#      (process creation + MPI_Init + MPI_Finalize), 5 repetitions per p;
#   4. the distributed experiment: one mpiexec launch per worker count p, each
#      running every size, appending to results_a4_scaling.csv.
# Keep the laptop idle and plugged in (High performance power plan).

param(
    [int[]]$Sizes = @(1000, 2000, 3000, 4000),
    [int[]]$Workers = @(1, 2, 4, 8, 16, 28),
    [int]$Reps = 5,
    [string]$Tag = "scaling"
)

$ErrorActionPreference = "Stop"
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\msys64\ucrt64\bin"

$results = "assignment4\data\results"
$exe = "assignment4\src\c\a4_bench.exe"
$a3exe = "assignment3\src\c\a3_bench.exe"
New-Item -ItemType Directory -Force $results | Out-Null

$plan = (powercfg /getactivescheme) -join " "
Write-Host "Active power plan: $plan"

& powershell -NoProfile -ExecutionPolicy Bypass -File assignment4\scripts\build.ps1
if ($LASTEXITCODE -ne 0) { throw "build failed" }
if (-not (Test-Path $a3exe)) {
    throw "$a3exe not found: run assignment3\scripts\build.ps1 first"
}

# ---- 2. Baselines (Assignment 3 harness, same session) ----------------------
$a3results = "assignment3\data\results"
$before = @(Get-ChildItem $a3results -File | ForEach-Object { $_.Name })
$baselineCsv = "$results\baseline_a3_threads.csv"
if (Test-Path $baselineCsv) { Remove-Item $baselineCsv }
$a3args = @("--experiment", "threads", "--sizes") + $Sizes + @("--threads") + $Workers +
          @("--reps", $Reps, "--out", $baselineCsv)
Write-Host "Baselines: $a3exe $($a3args -join ' ')"
& $a3exe @a3args *> "$results\run_log_baseline.txt"
if ($LASTEXITCODE -ne 0) { throw "baseline run failed" }
# The A3 harness writes its cross-check and sysinfo logs next to its own
# results: move the new ones here so Assignment 3's folder is left untouched.
$logs = "$results\baseline_a3_logs"
New-Item -ItemType Directory -Force $logs | Out-Null
Get-ChildItem $a3results -File | Where-Object { $before -notcontains $_.Name } |
    ForEach-Object { Move-Item $_.FullName "$logs\$($_.Name)" }

# ---- 3. MPI start-up cost ----------------------------------------------------
$startupCsv = "$results\results_a4_startup.csv"
"assignment,workers,repetition_id,wall_ms" | Set-Content -Encoding ascii $startupCsv
foreach ($p in $Workers) {
    for ($rep = 0; $rep -le $Reps; $rep++) {   # rep 0 = warm-up (excluded)
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & mpiexec -n $p $exe --sysinfo *> $null
        $sw.Stop()
        "4,$p,$rep,$([math]::Round($sw.Elapsed.TotalMilliseconds, 1))" |
            Add-Content -Encoding ascii $startupCsv
    }
    Write-Host "start-up p=${p} done"
}

# ---- 4. Distributed experiment ----------------------------------------------
$scalingCsv = "$results\results_a4_$Tag.csv"
$crossCsv = "$results\results_a4_${Tag}_crosscheck.csv"
foreach ($f in @($scalingCsv, $crossCsv)) { if (Test-Path $f) { Remove-Item $f } }
"workers,wall_seconds" | Set-Content -Encoding ascii "$results\launcher_wall_$Tag.csv"
foreach ($p in $Workers) {
    Write-Host "=== p = $p ranks ==="
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & mpiexec -n $p $exe --sizes @Sizes --reps $Reps --experiment $Tag --out $scalingCsv *> "$results\run_log_${Tag}_p$p.txt"
    $rc = $LASTEXITCODE
    $sw.Stop()
    "$p,$([math]::Round($sw.Elapsed.TotalSeconds, 1))" |
        Add-Content -Encoding ascii "$results\launcher_wall_$Tag.csv"
    if ($rc -ne 0) { throw "mpiexec failed for p=$p (exit $rc); see run_log_${Tag}_p$p.txt" }
}

# ---- Fingerprint --------------------------------------------------------------
"key,value" | Set-Content -Encoding ascii "$results\sysinfo_$Tag.csv"
& mpiexec -n 1 $exe --sysinfo | Add-Content -Encoding ascii "$results\sysinfo_$Tag.csv"
("powerplan," + ($plan -replace ",", ";")) |
    Add-Content -Encoding ascii "$results\sysinfo_$Tag.csv"
Write-Host "Done. Raw results: $scalingCsv"
