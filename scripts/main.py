# main.py
# Tkinter GUI + main pipeline for the MHW–OISST toolkit.
# NOTE: interactive anomaly map removed (use GEE/QGIS instead).

import os
import time
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from tqdm.auto import tqdm

from data_io import (
    ensure_all_years_downloaded,
    load_sst_timeseries_for_region,
)
from mhw_core import (
    compute_climatology_and_threshold,
    detect_marine_heatwaves,
    summarize_events_by_year,
)
from stats_tool import run_significance_tests
from plotting import plot_time_series_with_events, plot_seasonal_cycle_by_year

# -------------------------------------------------
# Project root (parent of scripts/)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# REGION PRESETS
# ============================================================

REGION_PRESETS = {
    "Mediterranean Sea": {"lat_min": 30.0, "lat_max": 46.0, "lon_min": -6.0, "lon_max": 36.0},
    "Western Mediterranean": {"lat_min": 35.0, "lat_max": 45.0, "lon_min": -6.0, "lon_max": 10.0},
    "Eastern Mediterranean": {"lat_min": 32.0, "lat_max": 40.0, "lon_min": 20.0, "lon_max": 36.0},
    "Adriatic Sea": {"lat_min": 40.0, "lat_max": 46.5, "lon_min": 12.0, "lon_max": 20.0},
    "North Sea": {"lat_min": 50.0, "lat_max": 62.0, "lon_min": -4.0, "lon_max": 9.0},
    "Bay of Biscay": {"lat_min": 43.0, "lat_max": 50.0, "lon_min": -10.0, "lon_max": -1.0},
    "Baltic Sea": {"lat_min": 53.0, "lat_max": 66.0, "lon_min": 10.0, "lon_max": 30.0},
    "Gulf of Naples": {"lat_min": 40.70, "lat_max": 40.95, "lon_min": 13.85, "lon_max": 14.35},
    "Custom region": None,
}


# ============================================================
# Helper parsing
# ============================================================

def _parse_date(text: str, field: str) -> str:
    """Parse a YYYY-MM-DD string; raise ValueError if invalid."""
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return text
    except ValueError:
        raise ValueError(f"{field}: invalid date, use YYYY-MM-DD")


def _parse_float(text: str, field: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{field}: must be a number")


def _parse_int(text: str, field: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{field}: must be an integer")


# ============================================================
# Tkinter GUI
# ============================================================

def interactive_setup_gui() -> dict:
    """
    Build Tkinter configuration window and return a config dict
    used by main().
    """
    cfg: dict = {}

    root = tk.Tk()
    root.title("Marine Heatwave Detector – Configuration")

    region_names = list(REGION_PRESETS.keys())
    region_var = tk.StringVar(value="Gulf of Naples")

    lat_min_var = tk.StringVar(value="40.70")
    lat_max_var = tk.StringVar(value="40.95")
    lon_min_var = tk.StringVar(value="13.85")
    lon_max_var = tk.StringVar(value="14.35")

    # Optional custom label when "Custom region" is selected
    custom_name_var = tk.StringVar(value="")

    analysis_start_var = tk.StringVar(value="1984-01-01")
    analysis_end_var = tk.StringVar(value="2024-12-31")
    baseline_start_var = tk.StringVar(value="1984-01-01")
    baseline_end_var = tk.StringVar(value="2013-12-31")

    percentile_var = tk.StringVar(value="0.9")
    min_length_var = tk.StringVar(value="5")

    folder_var = tk.StringVar(value="")

    # ------------ Callbacks ------------

    def on_region_change(*_):
        name = region_var.get()

        # Show/hide custom-name entry
        if name == "Custom region":
            custom_name_entry.grid(row=1, column=1, columnspan=3, sticky="we", padx=5)
        else:
            custom_name_entry.grid_remove()

        # For presets, auto-fill bounding box
        if name in REGION_PRESETS and name != "Custom region":
            r = REGION_PRESETS[name]
            lat_min_var.set(str(r["lat_min"]))
            lat_max_var.set(str(r["lat_max"]))
            lon_min_var.set(str(r["lon_min"]))
            lon_max_var.set(str(r["lon_max"]))

    region_var.trace_add("write", on_region_change)

    def on_browse_folder():
        path = filedialog.askdirectory(title="Select folder for OISST NetCDF files")
        if path:
            folder_var.set(path)

    def on_submit():
        nonlocal cfg
        try:
            region_name = region_var.get()

            # If preset: use stored bounds
            if region_name in REGION_PRESETS and region_name != "Custom region":
                preset = REGION_PRESETS[region_name]
                lat_min = preset["lat_min"]
                lat_max = preset["lat_max"]
                lon_min = preset["lon_min"]
                lon_max = preset["lon_max"]
            else:
                # Custom region: user-defined name + bounds
                user_label = custom_name_var.get().strip()
                if not user_label:
                    raise ValueError("Please provide a name for your custom region.")
                region_name = user_label
                lat_min = _parse_float(lat_min_var.get(), "lat_min")
                lat_max = _parse_float(lat_max_var.get(), "lat_max")
                lon_min = _parse_float(lon_min_var.get(), "lon_min")
                lon_max = _parse_float(lon_max_var.get(), "lon_max")

            analysis_start = _parse_date(analysis_start_var.get(), "Analysis start")
            analysis_end = _parse_date(analysis_end_var.get(), "Analysis end")
            baseline_start = _parse_date(baseline_start_var.get(), "Baseline start")
            baseline_end = _parse_date(baseline_end_var.get(), "Baseline end")

            percentile = _parse_float(percentile_var.get(), "Percentile")
            min_length = _parse_int(min_length_var.get(), "Minimum duration")

            raw_data_dir = folder_var.get().strip()
            if not raw_data_dir:
                raise ValueError("Please select the OISST data folder.")
            if not os.path.isdir(raw_data_dir):
                raise ValueError(f"Folder does not exist: {raw_data_dir}")

            # Absolute paths for plots/tables at project root
            plots_dir = os.path.join(BASE_DIR, "plots")
            tables_dir = os.path.join(BASE_DIR, "tables")

            cfg = {
                "region_name": region_name,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "analysis_start": analysis_start,
                "analysis_end": analysis_end,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "percentile": percentile,
                "min_event_length": min_length,
                "raw_data_dir": raw_data_dir,
                "plots_dir": plots_dir,
                "tables_dir": tables_dir,
            }

            root.destroy()

        except Exception as e:
            messagebox.showerror("Invalid configuration", str(e))

    # ------------ Layout ------------

    pad = {"padx": 5, "pady": 3}
    row = 0

    tk.Label(root, text="Region:").grid(row=row, column=0, sticky="e", **pad)
    region_menu = ttk.Combobox(root, textvariable=region_var, values=region_names, state="readonly")
    region_menu.grid(row=row, column=1, columnspan=3, sticky="we", **pad)

    row += 1
    tk.Label(root, text="Custom name:").grid(row=row, column=0, sticky="e", **pad)
    custom_name_entry = tk.Entry(root, textvariable=custom_name_var)
    custom_name_entry.grid(row=row, column=1, columnspan=3, sticky="we", padx=5)
    custom_name_entry.grid_remove()  # only visible for "Custom region"

    row += 1
    tk.Label(root, text="lat_min:").grid(row=row, column=0, sticky="e", **pad)
    tk.Entry(root, textvariable=lat_min_var, width=10).grid(row=row, column=1, **pad)
    tk.Label(root, text="lat_max:").grid(row=row, column=2, sticky="e", **pad)
    tk.Entry(root, textvariable=lat_max_var, width=10).grid(row=row, column=3, **pad)

    row += 1
    tk.Label(root, text="lon_min:").grid(row=row, column=0, sticky="e", **pad)
    tk.Entry(root, textvariable=lon_min_var, width=10).grid(row=row, column=1, **pad)
    tk.Label(root, text="lon_max:").grid(row=row, column=2, sticky="e", **pad)
    tk.Entry(root, textvariable=lon_max_var, width=10).grid(row=row, column=3, **pad)

    row += 1
    tk.Label(root, text="Analysis start (YYYY-MM-DD):").grid(row=row, column=0, sticky="e", **pad)
    tk.Entry(root, textvariable=analysis_start_var, width=12).grid(row=row, column=1, **pad)
    tk.Label(root, text="Analysis end:").grid(row=row, column=2, sticky="e", **pad)
    tk.Entry(root, textvariable=analysis_end_var, width=12).grid(row=row, column=3, **pad)

    row += 1
    tk.Label(root, text="Baseline start:").grid(row=row, column=0, sticky="e", **pad)
    tk.Entry(root, textvariable=baseline_start_var, width=12).grid(row=row, column=1, **pad)
    tk.Label(root, text="Baseline end:").grid(row=row, column=2, sticky="e", **pad)
    tk.Entry(root, textvariable=baseline_end_var, width=12).grid(row=row, column=3, **pad)

    row += 1
    tk.Label(root, text="Percentile (0–1):").grid(row=row, column=0, sticky="e", **pad)
    tk.Entry(root, textvariable=percentile_var, width=8).grid(row=row, column=1, **pad)
    tk.Label(root, text="Min duration (days):").grid(row=row, column=2, sticky="e", **pad)
    tk.Entry(root, textvariable=min_length_var, width=8).grid(row=row, column=3, **pad)

    row += 1
    tk.Label(root, text="OISST folder:").grid(row=row, column=0, sticky="e", **pad)
    tk.Entry(root, textvariable=folder_var, width=35).grid(row=row, column=1, columnspan=2, **pad)
    tk.Button(root, text="Browse…", command=on_browse_folder).grid(row=row, column=3, **pad)

    row += 1
    tk.Button(root, text="Run", command=on_submit).grid(
        row=row, column=0, columnspan=4, pady=10
    )

    root.resizable(False, False)
    root.mainloop()

    if not cfg:
        raise RuntimeError("Configuration window closed without confirmation.")

    return cfg


# ============================================================
# Main pipeline
# ============================================================

def main():
    cfg = interactive_setup_gui()

    print("\nConfiguration selected:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("")

    # Ensure output dirs at project root
    os.makedirs(cfg["plots_dir"], exist_ok=True)
    os.makedirs(cfg["tables_dir"], exist_ok=True)

    safe_region = cfg["region_name"].replace(" ", "_")

    year_a0 = int(cfg["analysis_start"][:4])
    year_a1 = int(cfg["analysis_end"][:4])
    year_b0 = int(cfg["baseline_start"][:4])
    year_b1 = int(cfg["baseline_end"][:4])

    start_year = min(year_a0, year_b0)
    end_year = max(year_a1, year_b1)

    total_steps = 8
    timings: dict[str, float] = {}

    with tqdm(total=total_steps, desc="Overall progress", unit="step") as pbar:

        # 1) Ensure data are present
        t0 = time.perf_counter()
        ensure_all_years_downloaded(start_year, end_year, cfg["raw_data_dir"])
        timings["Download check"] = time.perf_counter() - t0
        pbar.update(1)

        # 2) Load SST time series for region
        t0 = time.perf_counter()
        sst_ts = load_sst_timeseries_for_region(
            cfg["raw_data_dir"],
            cfg["lat_min"], cfg["lat_max"],
            cfg["lon_min"], cfg["lon_max"],
            start_year=start_year,
            end_year=end_year,
        )
        timings["Load SST"] = time.perf_counter() - t0
        pbar.update(1)

        # 3) Climatology & threshold
        t0 = time.perf_counter()
        clim_ts_full, thresh_ts_full = compute_climatology_and_threshold(
            sst_ts,
            cfg["baseline_start"],
            cfg["baseline_end"],
            cfg["percentile"],
        )
        timings["Climatology + threshold"] = time.perf_counter() - t0
        pbar.update(1)

        # 4) Restrict to analysis period
        t0 = time.perf_counter()
        sst_analysis = sst_ts.sel(time=slice(cfg["analysis_start"], cfg["analysis_end"]))
        clim_analysis = clim_ts_full.sel(time=slice(cfg["analysis_start"], cfg["analysis_end"]))
        thresh_analysis = thresh_ts_full.sel(time=slice(cfg["analysis_start"], cfg["analysis_end"]))
        timings["Restrict period"] = time.perf_counter() - t0
        pbar.update(1)

        # 5) Detect MHWs
        t0 = time.perf_counter()
        events_df = detect_marine_heatwaves(
            sst_analysis,
            thresh_analysis,
            cfg["min_event_length"],
        )
        events_out = os.path.join(cfg["tables_dir"], f"mhw_events_{safe_region}.csv")
        events_df.to_csv(events_out, index=False)
        timings["Detect MHWs"] = time.perf_counter() - t0
        pbar.update(1)

        # 6) Yearly summary
        t0 = time.perf_counter()
        summary_df = summarize_events_by_year(events_df)
        summary_df["region"] = cfg["region_name"]
        summary_out = os.path.join(cfg["tables_dir"], f"mhw_yearly_summary_{safe_region}.csv")
        summary_df.to_csv(summary_out, index=False)
        print(f"Saved yearly summary → {summary_out}")
        timings["Yearly summary"] = time.perf_counter() - t0
        pbar.update(1)

        # 7) Time-series + seasonal cycle plots
        t0 = time.perf_counter()
        ts_out = os.path.join(cfg["plots_dir"], f"mhw_timeseries_{safe_region}.png")
        plot_time_series_with_events(
            sst_analysis,
            clim_analysis,
            thresh_analysis,
            events_df,
            ts_out,
            region_name=cfg["region_name"],
        )

        seasonal_out = os.path.join(cfg["plots_dir"], f"seasonal_cycle_{safe_region}.png")
        plot_seasonal_cycle_by_year(
            sst_analysis,
            seasonal_out,
            region_name=cfg["region_name"],
        )

        timings["Timeseries + seasonal"] = time.perf_counter() - t0
        pbar.update(1)

        # 8) Significance tests (trends on SST and MHW metrics)
        t0 = time.perf_counter()
        run_significance_tests(
            sst_analysis,
            summary_df,
            plots_dir=cfg["plots_dir"],
            tables_dir=cfg["tables_dir"],
        )
        timings["Significance tests"] = time.perf_counter() - t0
        pbar.update(1)

    print("\n=== PERFORMANCE REPORT ===")
    for step, sec in timings.items():
        print(f"{step:30s} {sec:6.2f} sec")

    print("\nResults available in:")
    print(f"  Plots:  {cfg['plots_dir']}")
    print(f"  Tables: {cfg['tables_dir']}")


if __name__ == "__main__":
    main()
