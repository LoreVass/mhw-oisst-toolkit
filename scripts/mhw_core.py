# mhw_core.py
#
# Core marine heatwave logic:
# - climatology & threshold computation for a chosen baseline period
# - event detection using a percentile-based threshold
# - yearly summaries of marine heatwave characteristics

import numpy as np
import pandas as pd
import xarray as xr


def compute_climatology_and_threshold(
    sst_ts: xr.DataArray,
    baseline_start: str,
    baseline_end: str,
    percentile: float = 0.9,
):
    """
    Compute daily climatological mean and percentile-based threshold
    for a given baseline period.

    The function is designed to be robust to missing data:
      - NaN days within the baseline are entirely removed before statistics
      - Any missing day-of-year values in the climatology are interpolated
        along the day-of-year axis.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series (at least covering the baseline period).
        The DataArray must have a 'time' coordinate.
    baseline_start : str
        Start date of the baseline period in 'YYYY-MM-DD' format.
    baseline_end : str
        End date of the baseline period in 'YYYY-MM-DD' format.
    percentile : float, optional
        Percentile used to define the marine heatwave threshold
        (e.g. 0.9 for the 90th percentile).

    Returns
    -------
    clim_full : xr.DataArray
        Daily climatological mean mapped onto the full time axis of `sst_ts`.
    thresh_full : xr.DataArray
        Daily climatological percentile threshold mapped onto the full time axis.

    Notes
    -----
    The core idea follows the Hobday et al. (2016) framework:
      - Build a day-of-year climatology from the baseline window.
      - For each day of the year, compute a mean and a percentile-based threshold.
      - Map those day-of-year values back onto the full daily time series.
    """
    # 1) Select baseline period
    sst_base = sst_ts.sel(time=slice(baseline_start, baseline_end))

    # Basic diagnostic: warn if the baseline is very short
    if sst_base.time.size < 365 * 10:
        print("Warning: baseline period is quite short; "
              "climatology/threshold may be noisy.")

    # 2) Drop NaN days entirely from the baseline
    n_total = sst_base.time.size
    sst_base_clean = sst_base.where(~np.isnan(sst_base), drop=True)
    n_valid = sst_base_clean.time.size

    if n_valid < n_total:
        print(
            f"Note: dropped {n_total - n_valid} baseline days with NaN SST "
            f"({n_valid} valid / {n_total} total)."
        )

    if n_valid == 0:
        raise ValueError(
            "Baseline period contains only NaNs after cleaning. "
            "Check your data or baseline dates."
        )

    # 3) Group by day-of-year on the CLEAN baseline series
    doy = sst_base_clean["time"].dt.dayofyear

    # Daily climatological mean
    clim_mean = sst_base_clean.groupby(doy).mean("time")
    # Daily percentile threshold (e.g. 90th percentile)
    clim_thresh = sst_base_clean.groupby(doy).quantile(percentile, dim="time")

    # Use a consistent 'doy' name for the day-of-year coordinate
    clim_mean = clim_mean.rename({"dayofyear": "doy"})
    clim_thresh = clim_thresh.rename({"dayofyear": "doy"})

    # Ensure DOY is sorted (important for interpolation and indexing)
    clim_mean = clim_mean.sortby("doy")
    clim_thresh = clim_thresh.sortby("doy")

    # 4) Interpolate along DOY if any NaNs remain (e.g. due to sparse sampling)
    if clim_mean.isnull().any():
        print("Interpolating missing climatological mean values along DOY...")
        clim_mean = clim_mean.interpolate_na(
            dim="doy",
            method="linear",
            fill_value="extrapolate",
        )

    if clim_thresh.isnull().any():
        print("Interpolating missing climatological threshold values along DOY...")
        clim_thresh = clim_thresh.interpolate_na(
            dim="doy",
            method="linear",
            fill_value="extrapolate",
        )

    # 5) Map the day-of-year climatology back to the full analysis time axis
    full_doy = sst_ts["time"].dt.dayofyear
    clim_full = clim_mean.sel(doy=full_doy)
    thresh_full = clim_thresh.sel(doy=full_doy)

    return clim_full, thresh_full


def detect_marine_heatwaves(
    sst_ts: xr.DataArray,
    threshold_ts: xr.DataArray,
    min_length: int = 5,
) -> pd.DataFrame:
    """
    Detect marine heatwaves where SST exceeds a threshold for at least
    `min_length` consecutive days.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series (analysis window). Must have a 'time' coordinate.
    threshold_ts : xr.DataArray
        Daily threshold time series (same shape and time dimension as `sst_ts`).
    min_length : int, optional
        Minimum event duration in days (default: 5).

    Returns
    -------
    events_df : pd.DataFrame
        Table of detected events with columns:
        - 'start'               : event start date (timestamp)
        - 'end'                 : event end date (timestamp)
        - 'duration_days'       : length of the event in days
        - 'max_intensity_degC'  : maximum (SST - threshold) during the event
        - 'mean_intensity_degC' : mean (SST - threshold) during the event

    Notes
    -----
    The time series are internally converted to NumPy arrays for speed,
    and a simple run-length encoding logic is used to identify contiguous
    sequences where SST > threshold.
    """
    sst_vals = sst_ts.values
    thr_vals = threshold_ts.values
    time_vals = pd.to_datetime(sst_ts.time.values)

    if sst_vals.shape != thr_vals.shape:
        raise ValueError("sst_ts and threshold_ts must have the same shape.")

    # Boolean mask indicating where SST exceeds the threshold
    above = sst_vals > thr_vals

    events = []
    in_event = False
    start_idx = None

    # Loop over the time axis and track start and end of each "above" run
    for i, is_above in enumerate(above):
        if is_above and not in_event:
            # Entering a new candidate event
            in_event = True
            start_idx = i
        elif not is_above and in_event:
            # Exiting an event; finalize and check duration
            end_idx = i - 1
            duration = end_idx - start_idx + 1

            if duration >= min_length:
                event_sst = sst_vals[start_idx:end_idx + 1]
                event_thr = thr_vals[start_idx:end_idx + 1]
                intensity = event_sst - event_thr

                events.append({
                    "start": time_vals[start_idx],
                    "end": time_vals[end_idx],
                    "duration_days": duration,
                    "max_intensity_degC": float(intensity.max()),
                    "mean_intensity_degC": float(intensity.mean()),
                })

            in_event = False
            start_idx = None

    # If the time series ends while still inside an event
    if in_event:
        end_idx = len(above) - 1
        duration = end_idx - start_idx + 1
        if duration >= min_length:
            event_sst = sst_vals[start_idx:end_idx + 1]
            event_thr = thr_vals[start_idx:end_idx + 1]
            intensity = event_sst - event_thr
            events.append({
                "start": time_vals[start_idx],
                "end": time_vals[end_idx],
                "duration_days": duration,
                "max_intensity_degC": float(intensity.max()),
                "mean_intensity_degC": float(intensity.mean()),
            })

    return pd.DataFrame(events)


def summarize_events_by_year(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate marine heatwave events at yearly resolution, using the
    event start date to define the event's year.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of detect_marine_heatwaves().

    Returns
    -------
    summary : pd.DataFrame
        Yearly metrics with columns:
        - 'year'
        - 'n_events'              : number of events starting in that year
        - 'total_mhw_days'        : sum of event durations (days)
        - 'max_intensity_degC'    : maximum event intensity observed that year
        - 'mean_intensity_degC'   : mean intensity across events in that year
        - 'longest_event_days'    : duration of the longest event (days)

    Notes
    -----
    If `events_df` is empty, an empty summary frame with the expected
    columns is returned. This makes downstream plotting and analysis
    easier to handle without special-case checks.
    """
    if events_df.empty:
        return pd.DataFrame(columns=[
            "year", "n_events", "total_mhw_days",
            "max_intensity_degC", "mean_intensity_degC",
            "longest_event_days",
        ])

    df = events_df.copy()
    df["year"] = df["start"].dt.year

    summary = df.groupby("year").agg(
        n_events=("duration_days", "count"),
        total_mhw_days=("duration_days", "sum"),
        max_intensity_degC=("max_intensity_degC", "max"),
        mean_intensity_degC=("mean_intensity_degC", "mean"),
        longest_event_days=("duration_days", "max"),
    ).reset_index()

    return summary
