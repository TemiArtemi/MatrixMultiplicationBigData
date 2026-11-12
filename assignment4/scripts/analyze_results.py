#!/usr/bin/env python3
"""
Analysis pipeline for Assignment 4.

Reads the raw measurement CSVs of the distributed (MPI) experiment and of the
same-session Assignment 3 baselines, computes the protocol statistics (median
and interquartile range of the measured repetitions, warm-ups excluded),
derives strong-scaling speedup S(p) = T_seq(n) / T_p(n) and efficiency
E(p) = S(p) / p from MEDIAN times, and regenerates every table and figure
used in the report:

  report/summary_statistics.csv   - full statistics table (machine readable)
  report/scaling_statistics.csv   - speedup / efficiency / communication share
  report/tables/*.tex             - booktabs tables included by the report
  report/figures/*.png            - all report figures

Inputs (assignment4/data/results/):
  results_a4_scaling.csv     MPI row-block runs, one launch per worker count
  baseline_a3_threads.csv    A3 sequential kernel and OpenMP kernel, same session
  results_a4_startup.csv     launcher wall clock of mpiexec start-up
  launcher_wall_scaling.csv  launcher wall clock of each full launch

Run from anywhere:  python assignment4/scripts/analyze_results.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "data" / "results"
REPORT_DIR = BASE / "report"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.loc": "best",
})

SEQ = "tiled_seq_vec"        # Assignment 3 sequential baseline
OMP = "tiled_par_vec"        # Assignment 3 OpenMP kernel
MPI = "mpi_rowblock"
SIZE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
PHASES = [("t_scatter_ms", "Scatter A", "#1f77b4"),
          ("t_bcast_ms", "Broadcast B", "#ff7f0e"),
          ("t_compute_ms", "Compute (slowest rank)", "#2ca02c"),
          ("t_gather_ms", "Gather C", "#d62728")]


# ---------------------------------------------------------------------------
# Loading and statistics
# ---------------------------------------------------------------------------

def load(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        sys.exit(f"{path} not found; run scripts/run_experiments.ps1 first.")
    df = pd.read_csv(path)
    print(f"loaded {name}: {len(df)} rows")
    return df


def q1(s: pd.Series) -> float:
    return s.quantile(0.25)


def q3(s: pd.Series) -> float:
    return s.quantile(0.75)


def summarize_mpi(raw: pd.DataFrame) -> pd.DataFrame:
    """Median / IQR per (n, p) for the total time and every phase."""
    rows = []
    for (n, p), g in raw[raw["repetition_id"] >= 1].groupby(
            ["n", "threads_or_workers"]):
        ok = g[g["status"] == "OK"]
        row = {"n": int(n), "workers": int(p), "reps_ok": len(ok),
               "reps_failed": int((g["status"] != "OK").sum())}
        t = ok["execution_time_ms"]
        row.update({
            "median_ms": t.median() if len(ok) else np.nan,
            "q1_ms": q1(t) if len(ok) else np.nan,
            "q3_ms": q3(t) if len(ok) else np.nan,
            "iqr_ms": (q3(t) - q1(t)) if len(ok) else np.nan,
        })
        for col in ["t_scatter_ms", "t_bcast_ms", "t_compute_ms", "t_gather_ms",
                    "t_kernel_min_ms", "t_kernel_max_ms"]:
            row[f"med_{col}"] = ok[col].median() if len(ok) else np.nan
            row[f"iqr_{col}"] = (q3(ok[col]) - q1(ok[col])) if len(ok) else np.nan
        row["med_mem_sum_mb"] = ok["memory_value_mb"].median() if len(ok) else np.nan
        row["med_mem_rank0_mb"] = ok["mem_rank0_mb"].median() if len(ok) else np.nan
        row["med_mem_nonroot_mb"] = ok["mem_max_nonroot_mb"].median() if len(ok) else np.nan
        row["theoretical_storage_mb"] = g["theoretical_storage_mb"].iloc[0]
        row["t_load_ms"] = g["t_load_ms"].median()
        rows.append(row)
    s = pd.DataFrame(rows).sort_values(["n", "workers"]).reset_index(drop=True)
    comm = s["med_t_scatter_ms"] + s["med_t_bcast_ms"] + s["med_t_gather_ms"]
    s["comm_ms"] = comm
    s["comm_share"] = comm / s["median_ms"]
    return s


def summarize_baseline(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, n, p), g in base[base["repetition_id"] >= 1].groupby(
            ["method", "n", "threads_or_workers"]):
        ok = g[g["status"] == "OK"]["execution_time_ms"]
        rows.append({"method": m, "n": int(n), "workers": int(p),
                     "reps_ok": len(ok),
                     "median_ms": ok.median() if len(ok) else np.nan,
                     "q1_ms": q1(ok) if len(ok) else np.nan,
                     "q3_ms": q3(ok) if len(ok) else np.nan,
                     "iqr_ms": (q3(ok) - q1(ok)) if len(ok) else np.nan})
    return pd.DataFrame(rows).sort_values(["method", "n", "workers"]).reset_index(drop=True)


def scaling_table(mpi: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Speedup and efficiency (medians) of MPI and of the A3 OpenMP kernel."""
    seq = base[base["method"] == SEQ].set_index("n")["median_ms"]
    omp = base[base["method"] == OMP].set_index(["n", "workers"])["median_ms"]
    out = mpi.copy()
    out["t_seq_ms"] = out["n"].map(seq)
    out["speedup"] = out["t_seq_ms"] / out["median_ms"]
    out["efficiency"] = out["speedup"] / out["workers"]
    out["speedup_lo"] = out["t_seq_ms"] / out["q3_ms"]
    out["speedup_hi"] = out["t_seq_ms"] / out["q1_ms"]
    out["omp_median_ms"] = [omp.get((n, p), np.nan)
                            for n, p in zip(out["n"], out["workers"])]
    out["omp_speedup"] = out["t_seq_ms"] / out["omp_median_ms"]
    out["omp_efficiency"] = out["omp_speedup"] / out["workers"]
    out["mpi_over_omp"] = out["median_ms"] / out["omp_median_ms"]
    # Overhead of the distributed pipeline at fixed worker count versus its
    # own compute phase: how much of the end-to-end time is not computation.
    out["overhead_ms"] = out["median_ms"] - out["med_t_compute_ms"]
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def fmt_ms(median: float, iqr: float) -> str:
    if np.isnan(median):
        return "--"
    if median >= 1000:
        return f"{median:,.0f} [{iqr:,.0f}]".replace(",", "{,}")
    return f"{median:.1f} [{iqr:.1f}]"


def write_tex(name: str, lines: list[str]) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    path = TAB_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def table_time(mpi: pd.DataFrame, base_s: pd.DataFrame) -> None:
    sizes = sorted(mpi["n"].unique())
    reps = int(mpi["reps_ok"].max())
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{End-to-end time of the distributed product in ms (median "
        r"[IQR], $N=%d$ measured repetitions, 1 warm-up excluded) for $p$ MPI "
        r"ranks on one host, versus the sequential kernel measured in the same "
        r"session. End-to-end = $A,B$ in rank-0 memory $\rightarrow$ $C$ in "
        r"rank-0 memory.}" % reps,
        r"\label{tab:time}",
        r"\begin{tabular}{l" + "r" * len(sizes) + "}",
        r"\toprule",
        "Configuration & " + " & ".join(f"$n={n}$" for n in sizes) + r" \\",
        r"\midrule",
    ]
    seq = base_s[base_s["method"] == SEQ].set_index("n")
    lines.append("Sequential (A3 kernel) & " + " & ".join(
        fmt_ms(seq.loc[n, "median_ms"], seq.loc[n, "iqr_ms"]) for n in sizes) + r" \\")
    lines.append(r"\midrule")
    for p in sorted(mpi["workers"].unique()):
        d = mpi[mpi["workers"] == p].set_index("n")
        lines.append(f"MPI $p={p}$ & " + " & ".join(
            fmt_ms(d.loc[n, "median_ms"], d.loc[n, "iqr_ms"]) if n in d.index else "--"
            for n in sizes) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_time.tex", lines)


def table_speedup(scal: pd.DataFrame) -> None:
    sizes = sorted(scal["n"].unique())
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Strong-scaling speedup $S(p)=T_{\mathrm{seq}}(n)/T_p(n)$ "
        r"(efficiency $E(p)=S(p)/p$ in parentheses), from median times. Each "
        r"cell: MPI row-block / Assignment~3 OpenMP kernel measured in the "
        r"same session.}",
        r"\label{tab:speedup}",
        r"\begin{tabular}{l" + "c" * len(sizes) + "}",
        r"\toprule",
        "$p$ & " + " & ".join(f"$n={n}$" for n in sizes) + r" \\",
        r"\midrule",
    ]
    for p in sorted(scal["workers"].unique()):
        d = scal[scal["workers"] == p].set_index("n")
        cells = []
        for n in sizes:
            if n in d.index and not np.isnan(d.loc[n, "speedup"]):
                r = d.loc[n]
                cells.append(f"{r['speedup']:.2f} ({r['efficiency']:.2f}) / "
                             f"{r['omp_speedup']:.2f}")
            else:
                cells.append("--")
        lines.append(f"{p} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_speedup.tex", lines)


def table_phases(mpi: pd.DataFrame, n: int, label: str) -> None:
    d = mpi[mpi["n"] == n].sort_values("workers")
    if d.empty:
        return
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Phase decomposition at $n=%d$ (median ms over $N=%d$ "
        r"repetitions). The phases are consecutive barrier-delimited intervals "
        r"of one clock, so within each run they add up to the end-to-end time "
        r"(medians of the phases need not add up exactly to the median total); "
        r"the barriers remove any overlap between phases. Kernel min/max are "
        r"the fastest and slowest rank's compute time, measured outside the "
        r"boundary.}"
        % (n, int(d["reps_ok"].max())),
        r"\label{%s}" % label,
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"$p$ & Scatter & Bcast & Compute & Gather & Total & Comm.\ share & "
        r"Kernel min & Kernel max \\",
        r"\midrule",
    ]
    for r in d.itertuples(index=False):
        lines.append(
            f"{r.workers} & {r.med_t_scatter_ms:.1f} & {r.med_t_bcast_ms:.1f} & "
            f"{r.med_t_compute_ms:.1f} & {r.med_t_gather_ms:.1f} & "
            f"{r.median_ms:.1f} & {100 * r.comm_share:.1f}\\% & "
            f"{r.med_t_kernel_min_ms:.1f} & {r.med_t_kernel_max_ms:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(f"table_phases_n{n}.tex", lines)


def table_comm_share(mpi: pd.DataFrame) -> None:
    sizes = sorted(mpi["n"].unique())
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Share of the end-to-end time spent in data movement "
        r"(scatter $+$ broadcast $+$ gather), in percent of the median total.}",
        r"\label{tab:commshare}",
        r"\begin{tabular}{l" + "r" * len(sizes) + "}",
        r"\toprule",
        "$p$ & " + " & ".join(f"$n={n}$" for n in sizes) + r" \\",
        r"\midrule",
    ]
    for p in sorted(mpi["workers"].unique()):
        d = mpi[mpi["workers"] == p].set_index("n")
        lines.append(f"{p} & " + " & ".join(
            f"{100 * d.loc[n, 'comm_share']:.1f}" if n in d.index else "--"
            for n in sizes) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_comm_share.tex", lines)


def table_memory(mpi: pd.DataFrame) -> None:
    n = int(mpi["n"].max())
    d = mpi[mpi["n"] == n].sort_values("workers")
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Memory at $n=%d$. Metric: peak process working set (resident "
        r"memory) of each MPI rank, sampled every 5\,ms. Sum over ranks, rank 0 "
        r"(owns $A$, $B$, $C$ and the harness' reference result) and the largest "
        r"non-root rank (its $B$ copy plus its row blocks of $A$ and $C$), "
        r"versus the theoretical matrix storage allocated by the strategy "
        r"(all values in MB, median).}" % n,
        r"\label{tab:memory}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$p$ & Sum over ranks & Rank 0 & Max non-root rank & Theoretical sum \\",
        r"\midrule",
    ]
    for r in d.itertuples(index=False):
        nonroot = "--" if r.workers == 1 else f"{r.med_mem_nonroot_mb:.0f}"
        lines.append(f"{r.workers} & {r.med_mem_sum_mb:.0f} & "
                     f"{r.med_mem_rank0_mb:.0f} & {nonroot} & "
                     f"{r.theoretical_storage_mb:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_memory.tex", lines)


def table_probe(mpi: pd.DataFrame) -> None:
    """Probe: the same MPI runs with the 256-wide k/j tile (diagnostic, N=3)."""
    path = RESULTS_DIR / "results_a4_probe_block256.csv"
    if not path.exists():
        print("no block-256 probe; skipping table_probe.tex")
        return
    raw = pd.read_csv(path)
    pr = summarize_mpi(raw)
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Probe of the large-$p$ slowdown: end-to-end time in ms "
        r"(median [IQR]) with the Assignment~2/3 tuned $512$-wide $k/j$ tile "
        r"(main experiment, $N=%d$) and with a $256$-wide tile (diagnostic "
        r"probe, $N=%d$), same code and inputs. Gain $=T_{512}/T_{256}$.}"
        % (int(mpi["reps_ok"].max()), int(pr["reps_ok"].max())),
        r"\label{tab:probe}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$n$ & $p$ & $512$-wide tile & $256$-wide tile & Gain \\",
        r"\midrule",
    ]
    for n in sorted(pr["n"].unique()):
        for p in sorted(pr["workers"].unique()):
            a = mpi[(mpi["n"] == n) & (mpi["workers"] == p)]
            b = pr[(pr["n"] == n) & (pr["workers"] == p)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            lines.append(f"{n} & {p} & {fmt_ms(a['median_ms'], a['iqr_ms'])} & "
                         f"{fmt_ms(b['median_ms'], b['iqr_ms'])} & "
                         f"{a['median_ms'] / b['median_ms']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_probe.tex", lines)


def table_affinity(mpi: pd.DataFrame) -> None:
    """Diagnostic: default scheduling versus `mpiexec -affinity` (N=3)."""
    path = RESULTS_DIR / "results_a4_probe_affinity.csv"
    if not path.exists():
        print("no affinity probe; skipping table_affinity.tex")
        return
    pr = summarize_mpi(pd.read_csv(path))
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Diagnostic of \texttt{mpiexec -affinity} (MS-MPI default "
        r"pinning layout, $N=%d$) against the default Windows scheduling of the "
        r"main experiment ($N=%d$): end-to-end time in ms (median [IQR]) and "
        r"ratio of the slowest to the fastest rank's kernel time.}"
        % (int(pr["reps_ok"].max()), int(mpi["reps_ok"].max())),
        r"\label{tab:affinity}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$n$ & $p$ & Default & Imbalance & \texttt{-affinity} & Imbalance \\",
        r"\midrule",
    ]
    for n in sorted(pr["n"].unique()):
        for p in sorted(pr["workers"].unique()):
            a = mpi[(mpi["n"] == n) & (mpi["workers"] == p)]
            b = pr[(pr["n"] == n) & (pr["workers"] == p)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            lines.append(
                f"{n} & {p} & {fmt_ms(a['median_ms'], a['iqr_ms'])} & "
                f"{a['med_t_kernel_max_ms'] / a['med_t_kernel_min_ms']:.2f} & "
                f"{fmt_ms(b['median_ms'], b['iqr_ms'])} & "
                f"{b['med_t_kernel_max_ms'] / b['med_t_kernel_min_ms']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_affinity.tex", lines)


def table_overheads(mpi: pd.DataFrame, startup: pd.DataFrame) -> None:
    st = startup[startup["repetition_id"] >= 1].groupby("workers")["wall_ms"]
    sizes = sorted(mpi["n"].unique())
    load = mpi.groupby("n")["t_load_ms"].median()
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Costs outside the end-to-end boundary. Left: launcher wall "
        r"clock of \texttt{mpiexec -n p} for a program that only initialises "
        r"and finalises MPI (median [IQR] of $N=%d$ launches, ms). Right: rank-0 "
        r"time to read $A_n$ and $B_n$ from CSV (median ms).}" % int(st.size().max()),
        r"\label{tab:overheads}",
        r"\begin{tabular}{rr@{\qquad}rr}",
        r"\toprule",
        r"$p$ & MPI start-up & $n$ & CSV load \\",
        r"\midrule",
    ]
    ps = sorted(st.groups.keys())
    for i in range(max(len(ps), len(sizes))):
        left = ("", "")
        if i < len(ps):
            g = st.get_group(ps[i])
            left = (str(ps[i]), fmt_ms(g.median(), q3(g) - q1(g)))
        right = ("", "")
        if i < len(sizes):
            right = (str(sizes[i]), f"{load[sizes[i]]:,.0f}".replace(",", "{,}"))
        lines.append(f"{left[0]} & {left[1]} & {right[0]} & {right[1]}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_overheads.tex", lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def tick_label(p: int) -> str:
    return "" if p in (12, 20, 24) else str(p)


def mark_topology(ax) -> None:
    """Vertical guides: 8 P-cores, 16 P-core threads, 28 logical processors."""
    for x, label in ((8, "8 P-cores"), (16, "16 P-threads"), (28, "28 logical")):
        ax.axvline(x, color="grey", ls=":", lw=1)
        ax.text(x, ax.get_ylim()[1], label, rotation=90, va="top", ha="right",
                fontsize=7, color="grey")


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_time(scal: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.3))
    pvals = sorted(scal["workers"].unique())
    for i, n in enumerate(sorted(scal["n"].unique())):
        c = SIZE_COLORS[i % len(SIZE_COLORS)]
        d = scal[scal["n"] == n].sort_values("workers")
        ax.plot(d["workers"], d["median_ms"], "o-", color=c, label=f"MPI, $n={n}$")
        ax.fill_between(d["workers"], d["q1_ms"], d["q3_ms"], color=c, alpha=0.2)
        ax.plot(d["workers"], d["omp_median_ms"], "s--", color=c, ms=3, lw=0.9,
                alpha=0.8)
        ax.axhline(d["t_seq_ms"].iloc[0], color=c, ls=":", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(pvals)
    ax.set_xticklabels([str(p) for p in pvals])
    ax.set_xlabel("Number of workers $p$ (MPI ranks / OpenMP threads)")
    ax.set_ylabel("End-to-end / kernel time [ms]")
    ax.set_title("Distributed (MPI, solid) vs OpenMP (dashed) vs sequential (dotted)\n"
                 "median, shaded IQR of the MPI total")
    mark_topology(ax)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig_time_vs_workers.png")


def fig_speedup_efficiency(scal: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    pvals = sorted(scal["workers"].unique())
    for i, n in enumerate(sorted(scal["n"].unique())):
        c = SIZE_COLORS[i % len(SIZE_COLORS)]
        d = scal[scal["n"] == n].sort_values("workers")
        yerr = np.vstack([d["speedup"] - d["speedup_lo"], d["speedup_hi"] - d["speedup"]])
        ax1.errorbar(d["workers"], d["speedup"], yerr=yerr, fmt="o-", color=c,
                     capsize=2, ms=4, label=f"MPI, $n={n}$")
        ax1.plot(d["workers"], d["omp_speedup"], "s--", color=c, ms=3, lw=0.9,
                 alpha=0.7)
        ax2.plot(d["workers"], d["efficiency"], "o-", color=c, ms=4, label=f"$n={n}$")
        ax2.plot(d["workers"], d["omp_efficiency"], "s--", color=c, ms=3, lw=0.9,
                 alpha=0.7)
    ax1.plot(pvals, pvals, "k--", lw=1, label="Ideal $S=p$")
    ax1.set_ylim(0, max(30, scal["speedup_hi"].max() * 1.1))
    ax1.set_xlabel("Number of workers $p$")
    ax1.set_ylabel("Speedup $S(p)$ (from medians)")
    ax1.set_title("Speedup: MPI (solid) vs A3 OpenMP (dashed)")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("Number of workers $p$")
    ax2.set_ylabel("Efficiency $E(p)=S(p)/p$")
    ax2.set_title("Parallel efficiency")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(pvals)
        ax.set_xticklabels([tick_label(p) for p in pvals])
        mark_topology(ax)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_speedup_efficiency.png")


def fig_phases(mpi: pd.DataFrame) -> None:
    sizes = sorted(mpi["n"].unique())
    picks = [sizes[0], sizes[-1]] if len(sizes) > 1 else sizes
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 4.2),
                             squeeze=False)
    for ax, n in zip(axes[0], picks):
        d = mpi[mpi["n"] == n].sort_values("workers")
        x = np.arange(len(d))
        bottom = np.zeros(len(d))
        for col, label, color in PHASES:
            vals = d[f"med_{col}"].to_numpy()
            ax.bar(x, vals, bottom=bottom, color=color, label=label)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in d["workers"]])
        ax.set_xlabel("Number of ranks $p$")
        ax.set_ylabel("Time [ms] (median)")
        ax.set_title(f"Phase decomposition, $n={n}$")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_phases.png")


def fig_comm_share(mpi: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    pvals = sorted(mpi["workers"].unique())
    for i, n in enumerate(sorted(mpi["n"].unique())):
        d = mpi[mpi["n"] == n].sort_values("workers")
        ax.plot(d["workers"], 100 * d["comm_share"], "o-",
                color=SIZE_COLORS[i % len(SIZE_COLORS)], label=f"$n={n}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(pvals)
    ax.set_xticklabels([str(p) for p in pvals])
    ax.set_xlabel("Number of ranks $p$")
    ax.set_ylabel("Data movement share of end-to-end time [%]")
    ax.set_title("Share of scatter + broadcast + gather in the total time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_comm_share.png")


# ---------------------------------------------------------------------------

def main() -> None:
    raw = load("results_a4_scaling.csv")
    base_raw = load("baseline_a3_threads.csv")
    startup = load("results_a4_startup.csv")

    failed = raw[raw["status"] != "OK"]
    if not failed.empty:
        print("Non-OK MPI rows (reported, not timed):")
        print(failed[["n", "threads_or_workers", "repetition_id", "status"]]
              .to_string(index=False))

    mpi = summarize_mpi(raw)
    base_s = summarize_baseline(base_raw)
    scal = scaling_table(mpi, base_s)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mpi.to_csv(REPORT_DIR / "summary_statistics.csv", index=False, float_format="%.6f")
    scal.to_csv(REPORT_DIR / "scaling_statistics.csv", index=False, float_format="%.6f")
    base_s.to_csv(REPORT_DIR / "baseline_statistics.csv", index=False, float_format="%.6f")

    table_time(mpi, base_s)
    table_speedup(scal)
    table_comm_share(mpi)
    table_phases(mpi, int(mpi["n"].max()), "tab:phases-large")
    table_phases(mpi, int(mpi["n"].min()), "tab:phases-small")
    table_memory(mpi)
    table_overheads(mpi, startup)
    table_probe(mpi)
    table_affinity(mpi)
    fig_time(scal)
    fig_speedup_efficiency(scal)
    fig_phases(mpi)
    fig_comm_share(mpi)


if __name__ == "__main__":
    main()
