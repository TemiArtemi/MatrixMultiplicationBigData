#!/usr/bin/env python3
"""
Analysis pipeline for Assignment 2.

Reads the raw measurement CSVs produced by the C harness
(assignment2/data/results/results_a2_*.csv), computes the protocol
statistics (median and interquartile range of the measured repetitions,
warm-ups excluded) and regenerates every table and figure used in the
report:

  report/summary_statistics.csv   - full statistics table (machine readable)
  report/tables/*.tex             - booktabs tables included by the report
  report/figures/*.png            - all report figures

Run from anywhere:  python assignment2/scripts/analyze_results.py
"""

import glob
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

# Style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.legendloc": "best",
})

DENSE_METHODS = ["dense_ijk_baseline", "dense_ikj_reorder", "dense_blocked"]
DENSE_LABELS = {
    "dense_ijk_baseline": "Baseline i-j-k",
    "dense_ikj_reorder": "Opt. i-k-j reorder",
    "dense_blocked": "Opt. cache blocking",
}
COLORS = {
    "dense_ijk_baseline": "#d62728",
    "dense_ikj_reorder": "#1f77b4",
    "dense_blocked": "#2ca02c",
}
SPARSE_COLOR = "#9467bd"
CONV_COLOR = "#8c564b"


def load_raw() -> pd.DataFrame:
    """Load every results_a2_*.csv; later files override earlier duplicates."""
    files = sorted(glob.glob(str(RESULTS_DIR / "results_a2_*.csv")))
    if not files:
        sys.exit(f"No results_a2_*.csv found in {RESULTS_DIR}")
    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        df["_order"] = i
        frames.append(df)
        print(f"loaded {Path(f).name}: {len(df)} rows")
    raw = pd.concat(frames, ignore_index=True)
    keys = ["method", "n", "density", "repetition_id"]
    raw = raw.sort_values("_order").drop_duplicates(keys, keep="last")
    return raw


def measured(df: pd.DataFrame) -> pd.DataFrame:
    """Measured (non-warm-up) repetitions that completed successfully."""
    return df[(df["repetition_id"] >= 1) & (df["status"] == "OK")].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Median / IQR per configuration (method, n, density)."""
    m = measured(df)
    group_cols = ["method", "n", "density"]
    rows = []
    for keys, g in m.groupby(group_cols):
        method, n, density = keys
        times = g["execution_time_ms"]
        mem = g["memory_value_mb"]
        rows.append({
            "method": method,
            "n": int(n),
            "density": float(density),
            "reps": len(g),
            "median_ms": times.median(),
            "iqr_ms": times.quantile(0.75) - times.quantile(0.25),
            "median_peak_mb": mem.median(),
            "nnz_A": int(g["nnz_A"].iloc[0]),
            "nnz_B": int(g["nnz_B"].iloc[0]),
            "nnz_C": int(g["nnz_C"].iloc[-1]),
            "theoretical_storage_mb": float(g["theoretical_storage_mb"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(["method", "density", "n"]).reset_index(drop=True)


def write_summary_csv(summary: pd.DataFrame) -> None:
    out = REPORT_DIR / "summary_statistics.csv"
    summary.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


# ---------------------------------------------------------------------------
# Part A figures
# ---------------------------------------------------------------------------

def fig_dense_time(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pa = summary[summary["method"].isin(DENSE_METHODS)]
    for method in DENSE_METHODS:
        d = pa[pa["method"] == method].sort_values("n")
        ax.plot(d["n"], d["median_ms"], "o-", color=COLORS[method],
                label=DENSE_LABELS[method])
        ax.fill_between(d["n"], d["median_ms"] - d["iqr_ms"],
                        d["median_ms"] + d["iqr_ms"], color=COLORS[method],
                        alpha=0.18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix dimension $n$ (elements per side)")
    ax.set_ylabel("Kernel execution time [ms] (median, IQR band)")
    ax.set_title("Part A: dense methods on the Assignment 1 matrices\n"
                 "(median of 5 repetitions, shaded IQR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dense_time.png")
    plt.close(fig)


def fig_dense_speedup(summary: pd.DataFrame) -> None:
    base = summary[summary["method"] == "dense_ijk_baseline"][["n", "median_ms"]]
    base = base.rename(columns={"median_ms": "t_base"})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method in ["dense_ikj_reorder", "dense_blocked"]:
        d = summary[summary["method"] == method].merge(base, on="n")
        d = d[d["t_base"] > 0]
        speedup = d["t_base"] / d["median_ms"]
        ax.plot(d["n"], speedup, "s-", color=COLORS[method],
                label=f"{DENSE_LABELS[method]} vs baseline")
        # IQR-based uncertainty band on the speedup
        hi = d["t_base"] / (d["median_ms"] - d["iqr_ms"] / 2).clip(lower=1e-12)
        lo = d["t_base"] / (d["median_ms"] + d["iqr_ms"] / 2)
        ax.fill_between(d["n"], lo, hi, color=COLORS[method], alpha=0.15)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Matrix dimension $n$")
    ax.set_ylabel("Speedup $T_{ijk}(n)\\;/\\;T_{method}(n)$ (median)")
    ax.set_title("Part A: speedup of the optimised dense methods over the "
                 "validated baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dense_speedup.png")
    plt.close(fig)


def fig_memory_dense(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pa = summary[summary["method"].isin(DENSE_METHODS)]
    for method in DENSE_METHODS:
        d = pa[pa["method"] == method].sort_values("n")
        ax.plot(d["n"], d["median_peak_mb"], "o-", color=COLORS[method],
                label=DENSE_LABELS[method])
    ns = np.array(sorted(pa["n"].unique()), dtype=float)
    theo = 3.0 * 8.0 * ns ** 2 / (1024 ** 2)
    ax.plot(ns, theo, "k--", lw=1.0, label="Theoretical $3\\times 8n^2$ bytes")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix dimension $n$")
    ax.set_ylabel("Peak process working set [MB]")
    ax.set_title("Part A: peak resident memory (sampled RSS) vs. the "
                 "theoretical array storage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_memory_dense.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part B figures
# ---------------------------------------------------------------------------

def part_b_tables(summary: pd.DataFrame):
    pb = summary[summary["method"].isin(
        ["dense_blocked", "sparse_csr_gustavson", "dense_to_csr_conversion"])]
    dense = pb[pb["method"] == "dense_blocked"]
    sparse = pb[pb["method"] == "sparse_csr_gustavson"]
    conv = pb[pb["method"] == "dense_to_csr_conversion"]
    return dense, sparse, conv


def fig_sparse_time(summary: pd.DataFrame) -> None:
    dense, sparse, conv = part_b_tables(summary)
    ns = sorted(set(sparse["n"]))
    ncols = 3
    nrows = int(np.ceil(len(ns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, n in zip(axes, ns):
        s = sparse[sparse["n"] == n].sort_values("density", ascending=False)
        d = dense[dense["n"] == n]
        c = conv[conv["n"] == n].sort_values("density", ascending=False)
        ax.plot(s["density"] * 100, s["median_ms"], "o-", color=SPARSE_COLOR,
                label="Sparse CSR (Gustavson)")
        ax.plot(c["density"] * 100, c["median_ms"], "^--", color=CONV_COLOR,
                label="Dense-to-CSR conversion")
        if len(d):
            ax.axhline(d["median_ms"].iloc[0], color=COLORS["dense_blocked"],
                       lw=1.6, label="Dense cache blocking")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"$n = {n}$")
        ax.set_xlabel("Input density [%]")
        ax.set_ylabel("Kernel time [ms] (median)")
    for ax in axes[len(ns):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Part B: sparse vs. dense kernel time versus input density",
                 y=1.06)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sparse_time.png")
    plt.close(fig)


def fig_sparse_speedup(summary: pd.DataFrame) -> None:
    dense, sparse, _ = part_b_tables(summary)
    merged = sparse.merge(dense, on=["n", "density"], suffixes=("_sp", "_de"))
    merged = merged[merged["median_ms_sp"] > 0]
    merged["speedup"] = merged["median_ms_de"] / merged["median_ms_sp"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for n in sorted(merged["n"].unique()):
        d = merged[merged["n"] == n].sort_values("density", ascending=False)
        ax.plot(d["density"] * 100, d["speedup"], "o-", label=f"$n = {n}$")
    ax.axhline(1.0, color="black", lw=0.9, ls="--",
               label="Break-even (sparse = dense)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input density [%]")
    ax.set_ylabel("Speedup $T_{dense}\\;/\\;T_{sparse}$ (median)")
    ax.set_title("Part B: sparse advantage versus input density\n"
                 "(dense baseline: best dense method, cache blocking)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sparse_speedup.png")
    plt.close(fig)


def fig_fill_in(summary: pd.DataFrame) -> None:
    _, sparse, _ = part_b_tables(summary)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for n in sorted(sparse["n"].unique()):
        d = sparse[sparse["n"] == n].sort_values("density")
        out_density = d["nnz_C"] / float(n * n)
        ax.plot(d["density"] * 100, out_density, "o-", label=f"$n = {n}$ (measured)")
        # Independent-Bernoulli model: P(C_ij != 0) = 1 - (1 - d^2)^n
        dd = d["density"].to_numpy()
        model = 1.0 - (1.0 - dd ** 2) ** n
        ax.plot(d["density"] * 100, model, "--", lw=0.9,
                color=ax.lines[-1].get_color(), alpha=0.6,
                label=f"$n = {n}$ (model)")
    ax.set_xscale("log")
    ax.set_xlabel("Input density [%]")
    ax.set_ylabel("Output density $nnz_C / n^2$")
    ax.set_title("Part B: fill-in of the product (dashed: independent\n"
                 "Bernoulli model $1-(1-d^2)^n$)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fill_in.png")
    plt.close(fig)


def fig_memory_sparse(summary: pd.DataFrame) -> None:
    _, sparse, _ = part_b_tables(summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for n in sorted(sparse["n"].unique()):
        d = sparse[sparse["n"] == n].sort_values("density", ascending=False)
        ax.plot(d["density"] * 100, d["theoretical_storage_mb"], "o-",
                label=f"CSR: $A$, $B$, $C$")
    for n in sorted(sparse["n"].unique()):
        d = sparse[sparse["n"] == n].sort_values("density", ascending=False)
        dense_mb = 3.0 * 8.0 * n * n / (1024 ** 2)
        ax.plot(d["density"] * 100, np.full(len(d), dense_mb), "--", lw=1.0,
                color=COLORS["dense_blocked"],
                label="Dense $3\\times 8n^2$" if n == sorted(sparse["n"].unique())[0] else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input density [%]")
    ax.set_ylabel("Theoretical storage [MB]")
    ax.set_title("Storage: CSR ($A,B,C$) vs. dense arrays")
    ax.legend(fontsize=8)

    ax = axes[1]
    for n in sorted(sparse["n"].unique()):
        d = sparse[sparse["n"] == n].sort_values("density", ascending=False)
        ax.plot(d["density"] * 100, d["median_peak_mb"], "o-",
                label=f"$n = {n}$")
    ax.set_xscale("log")
    ax.set_xlabel("Input density [%]")
    ax.set_ylabel("Peak process working set [MB]")
    ax.set_title("Sampled peak RSS of the sparse kernel")
    ax.legend(fontsize=8)
    fig.suptitle("Part B: memory behaviour of the sparse representation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_memory_sparse.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def table_dense_time(summary: pd.DataFrame) -> None:
    pa = summary[summary["method"].isin(DENSE_METHODS)]
    ns = sorted(pa["n"].unique())
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Part A kernel execution time in ms (median [IQR] of the",
        r"measured repetitions; $N=5$; batch factor 100 applied for $n\leq50$).}",
        r"\label{tab:dense-time}",
        r"\begin{tabular}{l" + "c" * len(DENSE_METHODS) + "}",
        r"\toprule",
        r"$n$ & " + " & ".join(DENSE_LABELS[m] for m in DENSE_METHODS) + r" \\",
        r"\midrule",
    ]
    for n in ns:
        cells = []
        for m in DENSE_METHODS:
            row = pa[(pa["method"] == m) & (pa["n"] == n)]
            if len(row):
                r = row.iloc[0]
                cells.append(f"{r['median_ms']:.2f} [{r['iqr_ms']:.2f}]")
            else:
                cells.append("--")
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB_DIR / "table_dense_time.tex").write_text("\n".join(lines), encoding="utf-8")


def table_dense_speedup(summary: pd.DataFrame) -> None:
    base = summary[summary["method"] == "dense_ijk_baseline"][["n", "median_ms"]]
    base = base.rename(columns={"median_ms": "t_base"})
    ns = sorted(base["n"])
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Speedup of the optimised dense methods over the validated",
        r"$i$-$j$-$k$ baseline (ratio of medians on identical inputs).}",
        r"\label{tab:dense-speedup}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"$n$ & $i$-$k$-$j$ reorder & Cache blocking \\",
        r"\midrule",
    ]
    for n in ns:
        tb = base.loc[base["n"] == n, "t_base"].iloc[0]
        cells = []
        for m in ["dense_ikj_reorder", "dense_blocked"]:
            row = summary[(summary["method"] == m) & (summary["n"] == n)]
            cells.append(f"{tb / row['median_ms'].iloc[0]:.2f}\\x" if len(row) else "--")
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB_DIR / "table_dense_speedup.tex").write_text("\n".join(lines), encoding="utf-8")


def table_block_tuning() -> None:
    files = sorted(glob.glob(str(RESULTS_DIR / "tune_block_*.csv")))
    if not files:
        return
    rows = {}
    for f in files:
        name = Path(f).name
        block = int(name.replace("tune_block_", "").split("_")[0])
        df = pd.read_csv(f)
        df = df[(df["repetition_id"] >= 1) & (df["status"] == "OK")]
        for n, g in df.groupby("n"):
            rows.setdefault(block, {})[int(n)] = g["execution_time_ms"].median()
    ns = sorted({n for d in rows.values() for n in d})
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cache-block size tuning sweep: kernel time in ms (median of",
        r"3 repetitions) for the blocked kernel at several $n$. The default is",
        r"$b=512$.}",
        r"\label{tab:block-tuning}",
        r"\begin{tabular}{l" + "c" * len(ns) + "}",
        r"\toprule",
        r"Block $b$ & " + " & ".join(f"$n={n}$" for n in ns) + r" \\",
        r"\midrule",
    ]
    for block in sorted(rows):
        cells = [f"{rows[block][n]:.1f}" if n in rows[block] else "--" for n in ns]
        mark = " (default)" if block == 512 else ""
        lines.append(f"{block}{mark} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB_DIR / "table_block_tuning.tex").write_text("\n".join(lines), encoding="utf-8")


def table_sparse(summary: pd.DataFrame) -> None:
    dense, sparse, conv = part_b_tables(summary)
    merged = sparse.merge(dense, on=["n", "density"], suffixes=("_sp", "_de"))
    merged = merged.merge(conv[["n", "density", "median_ms"]], on=["n", "density"],
                          how="left")
    merged = merged.rename(columns={"median_ms": "median_ms_conv"})
    merged = merged.sort_values(["n", "density"], ascending=[True, False])
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Part B: sparse CSR product (Gustavson) versus the best dense",
        r"method (cache blocking) on identical logical inputs. Times are medians",
        r"of $N=5$ repetitions in ms. `Speedup' is",
        r"$T_{\mathrm{dense}}/T_{\mathrm{sparse}}$.}",
        r"\label{tab:sparse-time}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrcrrrrr}",
        r"\toprule",
        r"$n$ & Density [\%] & $nnz_A$ & $nnz_C$ & Dense [ms] & Sparse [ms] & Conv. [ms] & Speedup \\",
        r"\midrule",
    ]
    for _, r in merged.iterrows():
        conv_ms = f"{r['median_ms_conv']:.2f}" if pd.notna(r["median_ms_conv"]) else "--"
        sp = r["median_ms_de"] / r["median_ms_sp"] if r["median_ms_sp"] > 0 else float("inf")
        lines.append(
            f"{int(r['n'])} & {r['density'] * 100:.3g} & {int(r['nnz_A_sp']):,}".replace(",", r"\,")
            + f" & {int(r['nnz_C_sp']):,}".replace(",", r"\,")
            + f" & {r['median_ms_de']:.2f} & {r['median_ms_sp']:.3f} & {conv_ms} & {sp:.1f}\\x \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    (TAB_DIR / "table_sparse.tex").write_text("\n".join(lines), encoding="utf-8")


def table_fill_in(summary: pd.DataFrame) -> None:
    _, sparse, _ = part_b_tables(summary)
    merged = sparse.sort_values(["n", "density"], ascending=[True, False])
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Part B fill-in: measured output non-zeros versus the",
        r"independent-Bernoulli model $nnz_C \approx n^2\left[1-(1-d^2)^n\right]$.}",
        r"\label{tab:fill-in}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$n$ & Density $d$ [\%] & $nnz_C$ measured & Model & Output density [\%] \\",
        r"\midrule",
    ]
    for _, r in merged.iterrows():
        n = int(r["n"])
        d = r["density"]
        model = n * n * (1.0 - (1.0 - d * d) ** n)
        lines.append(
            f"{n} & {d * 100:.3g} & {int(r['nnz_C']):,}".replace(",", r"\,")
            + f" & {model:,.0f}".replace(",", r"\,")
            + f" & {100.0 * r['nnz_C'] / (n * n):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB_DIR / "table_fill_in.tex").write_text("\n".join(lines), encoding="utf-8")


def table_memory_partb(summary: pd.DataFrame) -> None:
    dense, sparse, _ = part_b_tables(summary)
    merged = sparse.merge(dense, on=["n", "density"], suffixes=("_sp", "_de"))
    merged = merged.sort_values(["n", "density"], ascending=[True, False])
    lines = [
        r"% Auto-generated by scripts/analyze_results.py -- do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Part B memory: sampled peak RSS of each kernel and theoretical",
        r"storage of the representations (dense: $3\times 8n^2$; sparse: CSR of",
        r"$A$, $B$ and $C$ plus the $O(n)$ Gustavson workspace).}",
        r"\label{tab:memory-partb}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$n$ & Density [\%] & Peak RSS dense [MB] & Peak RSS sparse [MB] & Storage dense/sparse [MB] \\",
        r"\midrule",
    ]
    for _, r in merged.iterrows():
        lines.append(
            f"{int(r['n'])} & {r['density'] * 100:.3g}"
            f" & {r['median_peak_mb_de']:.1f} & {r['median_peak_mb_sp']:.1f}"
            f" & {r['theoretical_storage_mb_de']:.1f} / {r['theoretical_storage_mb_sp']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB_DIR / "table_memory_partb.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    summary = summarize(raw)
    write_summary_csv(summary)

    # Timeouts and other non-OK outcomes are reported, never silently dropped.
    bad = raw[raw["status"] != "OK"]
    if len(bad):
        print("\nNon-completed runs (recorded with status, no invented time):")
        print(bad[["method", "n", "repetition_id", "status"]].to_string(index=False))

    fig_dense_time(summary)
    fig_dense_speedup(summary)
    fig_memory_dense(summary)
    fig_sparse_time(summary)
    fig_sparse_speedup(summary)
    fig_fill_in(summary)
    fig_memory_sparse(summary)

    table_dense_time(summary)
    table_dense_speedup(summary)
    table_block_tuning()
    table_sparse(summary)
    table_fill_in(summary)
    table_memory_partb(summary)

    print(f"\nFigures in {FIG_DIR}")
    print(f"Tables  in {TAB_DIR}")


if __name__ == "__main__":
    main()
