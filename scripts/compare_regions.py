#!/usr/bin/env python
# compare_regions.py
#
# Compare yearly marine heatwave metrics between two regions.
#
# Usage (CLI):
#   python scripts/compare_regions.py summary1.csv summary2.csv [--out OUT_DIR]
#
# Example:
#   python scripts/compare_regions.py \
#       mhw_yearly_summary_Eastern_Mediterranean.csv \
#       mhw_yearly_summary_Western_Mediterranean.csv \
#       --out compare_outputs
#
# Usage (no arguments):
#   python scripts/compare_regions.py
#   → opens file dialogs to pick the two summary CSVs and an output folder.
#
# Expected input CSV format:
#   year, n_events, total_mhw_days, max_intensity_degC,
#   mean_intensity_degC, longest_event_days, region
#
# Outputs:
#   - <out_dir>/mhw_compare_<REG1>_vs_<REG2>.csv
#   - <out_dir>/mhw_compare_total_mhw_days_<REG1>_vs_<REG2>.png
#   - <out_dir>/mhw_compare_max_intensity_degC_<REG1>_vs_<REG2>.png
#   - <out_dir>/mhw_compare_longest_event_days_<REG1>_vs_<REG2>.png

import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

# GUI fallback
import tkinter as tk
from tkinter import filedialog, messagebox

# Project root (parent of scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _safe_region_name(name: str) -> str:
    """Convert region label into a filesystem-safe token."""
    return name.strip().replace(" ", "_").replace("/", "-")


def _ensure_out_dir(path: str) -> str:
    """Create output directory if needed and return its absolute path."""
    out_dir = os.path.abspath(path)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_summary(path: str) -> pd.DataFrame:
    """Load a yearly summary CSV and validate minimum required columns."""
    df = pd.read_csv(path)

    required = {
        "year",
        "n_events",
        "total_mhw_days",
        "max_intensity_degC",
        "mean_intensity_degC",
        "longest_event_days",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"File '{path}' is missing columns: {', '.join(sorted(missing))}"
        )

    # If region column missing, create from filename as a fallback
    if "region" not in df.columns:
        base = os.path.basename(path)
        region_guess = os.path.splitext(base)[0]
        df["region"] = region_guess

    # Ensure numeric types
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def _plot_metric_comparison(
    df_merged: pd.DataFrame,
    region1: str,
    region2: str,
    metric: str,
    out_path: str,
):
    """
    Create a simple time-series comparison plot for a given metric:
    metric_region1 vs metric_region2 as separate lines.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    col1 = f"{metric}_{_safe_region_name(region1)}"
    col2 = f"{metric}_{_safe_region_name(region2)}"

    # Drop rows where both are NaN
    tmp = df_merged.dropna(subset=[col1, col2], how="all").copy()
    years = tmp["year"].values

    ax.plot(
        years,
        tmp[col1].values,
        "-o",
        label=region1,
        linewidth=1.5,
    )
    ax.plot(
        years,
        tmp[col2].values,
        "-o",
        label=region2,
        linewidth=1.5,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')} – {region1} vs {region2}")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison plot → {out_path}")


# ---------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------

def compare_summaries(path1: str, path2: str, out_dir: str):
    """Main function to compare two yearly summary tables."""
    out_dir = _ensure_out_dir(out_dir)

    print(f"\n=== Comparing regions ===")
    print(f"Summary 1: {path1}")
    print(f"Summary 2: {path2}")
    print(f"Output to: {out_dir}\n")

    df1 = _load_summary(path1)
    df2 = _load_summary(path2)

    region1 = str(df1["region"].iloc[0])
    region2 = str(df2["region"].iloc[0])

    safe1 = _safe_region_name(region1)
    safe2 = _safe_region_name(region2)

    print(f"Detected region 1: {region1}")
    print(f"Detected region 2: {region2}")

    # Select subset of columns to merge
    keep_cols = [
        "year",
        "n_events",
        "total_mhw_days",
        "max_intensity_degC",
        "mean_intensity_degC",
        "longest_event_days",
    ]

    df1_sub = df1[keep_cols].copy()
    df2_sub = df2[keep_cols].copy()

    df1_sub = df1_sub.rename(
        columns={c: f"{c}_{safe1}" for c in keep_cols if c != "year"}
    )
    df2_sub = df2_sub.rename(
        columns={c: f"{c}_{safe2}" for c in keep_cols if c != "year"}
    )

    merged = pd.merge(
        df1_sub,
        df2_sub,
        on="year",
        how="outer",
        sort=True,
    )

    # Save merged comparison table
    out_csv = os.path.join(
        out_dir,
        f"mhw_compare_{safe1}_vs_{safe2}.csv",
    )
    merged.to_csv(out_csv, index=False)
    print(f"Saved comparison table → {out_csv}")

    # Create a few standard comparison plots
    metrics_to_plot = [
        "total_mhw_days",
        "max_intensity_degC",
        "longest_event_days",
    ]

    for metric in metrics_to_plot:
        out_png = os.path.join(
            out_dir,
            f"mhw_compare_{metric}_{safe1}_vs_{safe2}.png",
        )
        _plot_metric_comparison(
            merged,
            region1=region1,
            region2=region2,
            metric=metric,
            out_path=out_png,
        )

    print("\nDone comparing regions.\n")


# ---------------------------------------------------------------------
# CLI + GUI entry point
# ---------------------------------------------------------------------

def _run_with_gui():
    """
    Fallback mode: ask the user for the two summary CSVs and an output folder
    using Tkinter file dialogs (starting in project root).
    """
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Compare regions",
        "Please select the FIRST yearly summary CSV (e.g.\n"
        "'mhw_yearly_summary_Eastern_Mediterranean.csv').",
    )
    path1 = filedialog.askopenfilename(
        title="Select first yearly summary CSV",
        initialdir=BASE_DIR,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path1:
        messagebox.showerror("Compare regions", "No file selected. Aborting.")
        return

    messagebox.showinfo(
        "Compare regions",
        "Now select the SECOND yearly summary CSV (e.g.\n"
        "'mhw_yearly_summary_Western_Mediterranean.csv').",
    )
    path2 = filedialog.askopenfilename(
        title="Select second yearly summary CSV",
        initialdir=BASE_DIR,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path2:
        messagebox.showerror("Compare regions", "No file selected. Aborting.")
        return

    messagebox.showinfo(
        "Compare regions",
        "Finally, select the folder where results should be saved.",
    )
    out_dir = filedialog.askdirectory(
        title="Select output folder for comparison products",
        initialdir=BASE_DIR,
    )
    if not out_dir:
        messagebox.showerror("Compare regions", "No output folder selected. Aborting.")
        return

    try:
        compare_summaries(path1, path2, out_dir)
        messagebox.showinfo(
            "Compare regions",
            f"Comparison finished.\n\nResults saved in:\n{os.path.abspath(out_dir)}",
        )
    except Exception as e:
        messagebox.showerror("Compare regions", f"Error during comparison:\n{e}")


def main():
    # If no extra args: use GUI mode
    if len(sys.argv) == 1:
        _run_with_gui()
        return

    # CLI mode with argparse
    parser = argparse.ArgumentParser(
        description="Compare yearly marine heatwave metrics between two regions."
    )
    parser.add_argument(
        "summary1",
        help="Path to first yearly summary CSV "
             "(e.g. mhw_yearly_summary_Eastern_Mediterranean.csv)",
    )
    parser.add_argument(
        "summary2",
        help="Path to second yearly summary CSV "
             "(e.g. mhw_yearly_summary_Western_Mediterranean.csv)",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        default=os.path.join(BASE_DIR, "compare_outputs"),
        help="Output directory for comparison tables and plots "
             "(default: '<project_root>/compare_outputs')",
    )

    args = parser.parse_args()

    compare_summaries(args.summary1, args.summary2, args.out_dir)


if __name__ == "__main__":
    main()
