# data_io.py
#
# I/O utilities for NOAA OISST v2 high-res data:
# - download missing years from PSL
# - load area-averaged SST time series for a region (fast, lon-safe, with coarsening for large regions)
# - load SST cubes for mapping (with coarsening for large regions)
#
# This version:
# - avoids open_mfdataset (too heavy for 40+ years)
# - loads one year at a time
# - slices spatially BEFORE doing heavy operations
# - handles 0–360° longitude grids (NOAA OISST) and regions crossing 0°
# - coarsens big regions to keep runtime reasonable, without biasing the area mean

import os
import glob
import time
from typing import List

import numpy as np
import xarray as xr
import requests
from tqdm.auto import tqdm

PSL_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"


# ============================================================
# Downloader
# ============================================================

def download_year_from_psl(year: int, raw_dir: str, overwrite: bool = False) -> str:
    """
    Download one year of daily OISST from NOAA PSL (high-res v2).

    Parameters
    ----------
    year : int
        Year to download (e.g., 1984).
    raw_dir : str
        Directory where NetCDF file will be stored.
    overwrite : bool
        If True, re-download even if file already exists.

    Returns
    -------
    out_path : str
        Path to the downloaded NetCDF file.
    """
    os.makedirs(raw_dir, exist_ok=True)
    fname = f"sst.day.mean.{year}.nc"
    url = f"{PSL_BASE_URL}/{fname}"
    out_path = os.path.join(raw_dir, fname)

    if os.path.exists(out_path) and not overwrite:
        print(f"✓ {year} already exists → skipping download")
        return out_path

    print(f"\n↓ Downloading {year} from PSL:\n   {url}")

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024

        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"   {fname}",
            leave=True,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"   → saved to {out_path}\n")
    return out_path


def ensure_all_years_downloaded(start: int, end: int, raw_dir: str):
    """
    Check which yearly OISST files are present in raw_dir and
    download any missing years from start to end (inclusive).
    """
    os.makedirs(raw_dir, exist_ok=True)
    expected = set(range(start, end + 1))
    existing = set()

    for f in os.listdir(raw_dir):
        if f.startswith("sst.day.mean.") and f.endswith(".nc"):
            parts = f.split(".")
            if len(parts) >= 4:
                try:
                    year = int(parts[3])
                    existing.add(year)
                except ValueError:
                    continue

    missing = sorted(expected - existing)

    print("\n=== OISST DATA CHECK ===")
    print(f"Requested years: {start}–{end}")
    print(f"Existing years:  {sorted(existing) if existing else '(none)'}")
    print(f"Missing years:   {missing if missing else 'none'}")
    print("========================\n")

    if not missing:
        print("✓ All requested years already downloaded — nothing to do.\n")
        return

    for y in tqdm(missing, desc="Downloading missing years", unit="year"):
        download_year_from_psl(y, raw_dir, overwrite=False)


# ============================================================
# Helper functions for xarray
# ============================================================

def _find_lon_name(ds: xr.Dataset) -> str:
    """Guess longitude coordinate name."""
    for cand in ["lon", "longitude", "x"]:
        if cand in ds.coords:
            return cand
    raise ValueError("Could not find longitude coordinate (lon/longitude/x) in dataset.")


def _find_lat_name(ds: xr.Dataset) -> str:
    """Guess latitude coordinate name."""
    for cand in ["lat", "latitude", "y"]:
        if cand in ds.coords:
            return cand
    raise ValueError("Could not find latitude coordinate (lat/latitude/y) in dataset.")


def _find_sst_var(ds: xr.Dataset) -> str:
    """Guess SST variable name."""
    for cand in ["sst", "SST", "sea_surface_temperature", "analysed_sst"]:
        if cand in ds.data_vars:
            return cand
    raise ValueError("Could not find SST variable in dataset.")


def _lon_to_dataset_domain(lon: float, lon_vals: np.ndarray) -> float:
    """
    Convert a user longitude (possibly -180..180) into the dataset
    longitude domain (either -180..180 or 0..360).

    This does NOT reorder the dataset; it just projects the user
    lon onto the same numeric range as lon_vals.
    """
    lo = float(np.nanmin(lon_vals))
    hi = float(np.nanmax(lon_vals))

    if lo >= 0.0 and hi > 180.0:
        # Dataset uses 0..360 (OISST case) → project with modulo
        return lon % 360.0
    else:
        # Dataset already -180..180 or similar
        return lon


def _slice_lon_region(
    da: xr.DataArray,
    lon_name: str,
    user_lon_min: float,
    user_lon_max: float,
) -> xr.DataArray:
    """
    Slice a DataArray in longitude, handling:
    - dataset lon in 0..360 or -180..180
    - regions crossing lon=0 in 0..360 domain (two slices merged)
    """
    lon_vals = da[lon_name].values
    lo = float(np.nanmin(lon_vals))
    hi = float(np.nanmax(lon_vals))
    is_0360 = (lo >= 0.0 and hi > 180.0)

    lon1 = _lon_to_dataset_domain(user_lon_min, lon_vals)
    lon2 = _lon_to_dataset_domain(user_lon_max, lon_vals)

    # Ensure increasing user bounds in geographic sense
    # but we must handle wrap-around separately for 0..360
    if not is_0360:
        # Normal case: dataset already ~-180..180
        lon_low, lon_high = sorted([lon1, lon2])
        return da.sel({lon_name: slice(lon_low, lon_high)})

    # 0..360 domain: need to consider wrap around 0°
    # Example: user [-6, 10] → [354, 10] in 0..360, which wraps.
    if lon1 <= lon2:
        # Simple case: does not wrap in 0..360
        return da.sel({lon_name: slice(lon1, lon2)})
    else:
        # Wrap case: [lon1..360] U [0..lon2]
        part1 = da.sel({lon_name: slice(lon1, 360.0)})
        part2 = da.sel({lon_name: slice(0.0, lon2)})
        # Concatenate along longitude dimension
        return xr.concat([part1, part2], dim=lon_name)


# ============================================================
# AREA-MEAN SST TIME SERIES (FAST + COARSENING)
# ============================================================

def load_sst_timeseries_for_region(
    raw_dir: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_year: int | None = None,
    end_year: int | None = None,
) -> xr.DataArray:
    """
    Load an area-averaged SST time series for a region.

    Strategy
    --------
    - Avoid open_mfdataset (too heavy for 40+ years).
    - Loop year-by-year, slicing only the requested region.
    - Handle longitude in 0..360 or -180..180 without reordering coords.
    - For very large regions, spatially coarsen before averaging to keep
      computation manageable.

    Returns
    -------
    mean_ts : xr.DataArray
        Daily SST (°C) averaged over the selected region.
    """

    pattern = os.path.join(raw_dir, "sst.day.mean.*.nc")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    # Filter by requested years
    filtered_files: List[str] = []
    for fpath in all_files:
        parts = os.path.basename(fpath).split(".")
        try:
            year = int(parts[3])
        except Exception:
            continue

        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        filtered_files.append(fpath)

    if not filtered_files:
        raise RuntimeError("No files match requested years.")

    print("\n[OPT] Loading SST year by year (optimized lon-safe mode)…")
    print(f"Files: {len(filtered_files)}\n")

    ts_list = []
    lat1, lat2 = sorted([lat_min, lat_max])

    for fpath in tqdm(filtered_files, desc="Optimized loader", unit="file"):
        t0 = time.perf_counter()
        fname = os.path.basename(fpath)
        print(f"[OPT] Reading {fname}…")

        ds = xr.open_dataset(fpath, engine="netcdf4")

        lon_name = _find_lon_name(ds)
        lat_name = _find_lat_name(ds)
        sst_var = _find_sst_var(ds)

        # 1. Slice in latitude immediately (reduces array size)
        sub = ds[sst_var].sel({lat_name: slice(lat1, lat2)})

        # 2. Slice in longitude using lon-domain aware helper (0..360 or -180..180)
        sub = _slice_lon_region(sub, lon_name, lon_min, lon_max)

        # 3. Optionally coarsen for very large regions
        nlat = sub.sizes[lat_name]
        nlon = sub.sizes[lon_name]
        n_cells = nlat * nlon

        # Threshold: if region has > 4000 grid cells, coarsen (2x2, possibly twice)
        if n_cells > 4000:
            print(
                f"[OPT] Large region for time series ({nlat}x{nlon} = {n_cells} cells), "
                f"coarsening before averaging…"
            )
            sub = sub.coarsen({lat_name: 2}, boundary="trim").mean()
            sub = sub.coarsen({lon_name: 2}, boundary="trim").mean()
            nlat = sub.sizes[lat_name]
            nlon = sub.sizes[lon_name]
            n_cells = nlat * nlon
            print(f"[OPT]  After coarsen(2x2): {nlat}x{nlon} = {n_cells} cells")

            if n_cells > 4000:
                print("[OPT]  Still large, applying second coarsen(2x2)…")
                sub = sub.coarsen({lat_name: 2}, boundary="trim").mean()
                sub = sub.coarsen({lon_name: 2}, boundary="trim").mean()
                nlat = sub.sizes[lat_name]
                nlon = sub.sizes[lon_name]
                n_cells = nlat * nlon
                print(f"[OPT]  After second coarsen: {nlat}x{nlon} = {n_cells} cells")

        # 4. Area mean for this year (lat, lon → mean)
        ts_year = sub.mean(dim=(lat_name, lon_name), skipna=True).load()
        ts_list.append(ts_year)
        ds.close()

        dt = time.perf_counter() - t0
        print(
            f"[OPT] {fname} processed in {dt:.2f} sec "
            f"(time steps: {ts_year.sizes['time']})"
        )

    # Concatenate all years
    mean_ts = xr.concat(ts_list, dim="time")

    # Kelvin → °C if needed
    if float(mean_ts.mean(skipna=True)) > 100.0:
        mean_ts -= 273.15

    mean_ts = mean_ts.sortby("time")
    print(f"\n[OPT] Final TS: {mean_ts.time.size} days\n")

    return mean_ts


# ============================================================
# SST CUBE FOR REGION (for maps, with coarsening)
# ============================================================

def load_sst_cube_for_region(
    raw_dir: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    """
    Load an SST cube (time, lat, lon) for the selected region,
    suitable for interactive mapping.

    Notes
    -----
    - Reads one year at a time.
    - Handles 0..360 or -180..180 longitude grids.
    - Coarsens very large regions to keep the cube reasonable for plotting.
    """

    pattern = os.path.join(raw_dir, "sst.day.mean.*.nc")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {raw_dir}")

    cube_list = []

    print("\nLoading SST cube year by year (map mode, lon-safe)...\n")

    lat1, lat2 = sorted([lat_min, lat_max])

    for fpath in tqdm(files, desc="Cube loader: yearly files", unit="file"):
        t0 = time.perf_counter()
        fname = os.path.basename(fpath)
        print(f"[CUBE] Reading {fname} …")

        ds = xr.open_dataset(fpath)

        lon_name = _find_lon_name(ds)
        lat_name = _find_lat_name(ds)
        sst_var = _find_sst_var(ds)

        # 1. Lat slice first
        sub = ds[sst_var].sel({lat_name: slice(lat1, lat2)})

        # 2. Lon slice with wrap-aware logic
        sub = _slice_lon_region(sub, lon_name, lon_min, lon_max)

        # 3. Coarsen if region is very large (map only needs moderate detail)
        nlat = sub.sizes[lat_name]
        nlon = sub.sizes[lon_name]
        n_cells = nlat * nlon

        if n_cells > 4000:
            print(f"[CUBE] Large region ({nlat}x{nlon} = {n_cells} cells), "
                  f"coarsening for map...")
            sub = sub.coarsen({lat_name: 2}, boundary="trim").mean()
            sub = sub.coarsen({lon_name: 2}, boundary="trim").mean()
            nlat = sub.sizes[lat_name]
            nlon = sub.sizes[lon_name]
            n_cells = nlat * nlon
            print(f"[CUBE] After coarsen(2x2): {nlat}x{nlon} = {n_cells} cells")

        # 4. Load into memory (after coarsening)
        sub = sub.load()

        cube_list.append(sub)
        ds.close()

        dt = time.perf_counter() - t0
        print(
            f"[CUBE] {fname} loaded in {dt:.2f} sec "
            f"(time={sub.sizes['time']}, lat={sub.sizes[lat_name]}, lon={sub.sizes[lon_name]})"
        )

    cube = xr.concat(cube_list, dim="time")

    # Kelvin → °C if needed
    if float(cube.mean(skipna=True)) > 100.0:
        cube = cube - 273.15

    cube = cube.sortby("time")

    print(
        "Final SST cube for map:",
        str(cube.time.values[0]),
        "→",
        str(cube.time.values[-1]),
        f"({cube.time.size} days)",
        f"lat={cube.sizes[_find_lat_name(cube)]}, lon={cube.sizes[_find_lon_name(cube)]}\n",
    )

    return cube
