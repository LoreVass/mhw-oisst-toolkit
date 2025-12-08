# plotting.py
#
# Plotting utilities:
# - SST time series with climatology, threshold, and shaded MHW events.
# - Seasonal cycle plot (SST vs day-of-year colored by year).

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np


def plot_time_series_with_events(
    sst_ts: xr.DataArray,
    clim_ts: xr.DataArray,
    thresh_ts: xr.DataArray,
    events_df: pd.DataFrame,
    out_path: str,
    region_name: str = "Selected Region",
):
    """
    Plot SST time series, climatology, threshold, and shading
    for detected MHW events.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series (analysis window).
    clim_ts : xr.DataArray
        Climatological mean SST for the same time window.
    thresh_ts : xr.DataArray
        Threshold SST (e.g. 90th percentile) for the same time window.
    events_df : pd.DataFrame
        Detected MHW events (from detect_marine_heatwaves).
    out_path : str
        Output PNG path.
    region_name : str
        Name of the region (for plot title).
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    time_vals = sst_ts.time.values

    ax.plot(time_vals, sst_ts, label="SST (°C)", linewidth=1)
    ax.plot(time_vals, clim_ts, label="Climatology (baseline mean)", linestyle="--")
    ax.plot(
        time_vals,
        thresh_ts,
        label="Threshold (percentile)",
        linestyle=":",
    )

    # Shade MHW events
    for i, ev in events_df.iterrows():
        ax.axvspan(
            ev["start"],
            ev["end"],
            alpha=0.2,
            color="tab:blue",
            label="MHW event" if i == 0 else None,
        )

    ax.set_title(f"Marine Heatwaves – {region_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (°C)")
    ax.legend()
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved SST & MHW time-series plot to {out_path}")


def plot_seasonal_cycle_by_year(
    sst_ts: xr.DataArray,
    out_path: str,
    region_name: str = "Selected Region",
):
    """
    Plot the seasonal cycle (day-of-year vs SST) for each year
    in the analysis period, colored by year.

    Each line corresponds to one year, x-axis = day of year,
    y-axis = SST (°C). The most recent year is highlighted with
    a thicker line.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series (analysis window).
    out_path : str
        Output PNG path.
    region_name : str
        Name of the region (for plot title).
    """
    # Convert to tabular form
    df = sst_ts.to_dataframe(name="sst").reset_index()

    # Drop Feb 29 so all years have 365 days
    is_feb29 = (df["time"].dt.month == 2) & (df["time"].dt.day == 29)
    df = df[~is_feb29].copy()

    df["year"] = df["time"].dt.year
    df["doy"] = df["time"].dt.dayofyear

    years = np.sort(df["year"].unique())
    if len(years) == 0:
        print("No data available for seasonal cycle plot.")
        return

    cmap = plt.get_cmap("inferno")
    norm = plt.Normalize(years.min(), years.max())

    fig, ax = plt.subplots(figsize=(10, 5))

    # Thin lines for all years
    for y in years:
        sub = df[df["year"] == y]
        ax.plot(
            sub["doy"],
            sub["sst"],
            color=cmap(norm(y)),
            linewidth=0.8,
        )

    # Highlight the most recent year
    latest_year = years.max()
    sub_latest = df[df["year"] == latest_year]
    ax.plot(
        sub_latest["doy"],
        sub_latest["sst"],
        color=cmap(norm(latest_year)),
        linewidth=2.0,
        label=f"{latest_year}",
    )

    ax.set_xlabel("Day of year")
    ax.set_ylabel("SST (°C)")
    ax.set_title(f"Seasonal SST cycle by year – {region_name}")

    # Colorbar keyed to years
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Year")

    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved seasonal cycle plot to {out_path}")
