# 🌊 Marine Heatwave (MHW) Detection Toolkit – OISST v2.1

A Python toolkit for detecting and analysing **marine heatwaves (MHWs)** using **NOAA OISST v2.1 daily sea surface temperature**.

---

## 🔥 Features

- Automated NOAA OISST year-check system  
- Fast SST loading over any custom or preset region  
- Daily climatology & percentile threshold (Hobday et al., 2016)  
- Full marine heatwave event detection  
- Annual summaries of MHW metrics  
- Trend analysis (linear regression + Mann–Kendall test)  
- Simple Tkinter graphical interface  

> **Note**  
> The interactive anomaly-map system was removed for stability.  
> For anomaly maps, use **Google Earth Engine** or **QGIS**.

---

## 📁 Project Structure

    mhw-oisst-toolkit/
    │
    ├── scripts/
    │   ├── main.py          # GUI + full analysis pipeline
    │   ├── data_io.py       # OISST loading utilities
    │   ├── mhw_core.py      # Climatology + threshold + MHW detection
    │   ├── stats_tool.py    # Trend + MK significance testing
    │   ├── plotting.py      # Plot generation (PNG)
    │
    ├── plots/               # Auto-generated figures
    ├── tables/              # Auto-generated CSV analysis tables
    │
    ├── README.md
    ├── requirements.txt
    └── LICENSE

---

## 🚀 Installation

1. **Clone the repository**

        git clone https://github.com/Lorevass/mhw-oisst-toolkit.git
        cd mhw-oisst-toolkit

2. **Create a virtual environment**

        python -m venv .venv
        # macOS / Linux
        source .venv/bin/activate
        # Windows
        .venv\Scripts\activate

3. **Install dependencies**

        pip install -r requirements.txt

---

## 🌐 NOAA OISST v2.1 Data

Download daily files (`sst.day.mean.YYYY.nc`) from:

https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html

Place them in a folder such as:

    C:/Users/<YOU>/Data/OISST/

The toolkit will automatically check that all required years are present.

---

## 🖥 Running the Toolkit (GUI)

Run:

    python scripts/main.py

You will be asked to select:

- Region (preset or custom)
- Latitude/longitude bounds
- Analysis period
- Baseline climatology period
- Percentile threshold (default: 0.9)
- Minimum heatwave duration (default: 5 days)
- Folder containing the NOAA OISST NetCDF files

Outputs are saved into:

- `plots/`
- `tables/`

---

## 📊 Output Files

### Figures (PNG)

- `mhw_timeseries_<REGION>.png`  
- `seasonal_cycle_<REGION>.png`  
- `trend_sst_annual_mean_<REGION>.png`  
- `trend_sst_annual_min_<REGION>.png`  
- `trend_sst_annual_max_<REGION>.png`  
- `trend_mhw_n_events_<REGION>.png`  
- `trend_mhw_total_mhw_days_<REGION>.png`  
- `trend_mhw_max_intensity_<REGION>.png`  
- `trend_mhw_mean_intensity_<REGION>.png`  
- `trend_mhw_longest_event_<REGION>.png`  

### Tables (CSV)

- `mhw_events_<REGION>.csv`  
- `mhw_yearly_summary_<REGION>.csv`  
- `trend_significance_sst_<REGION>.csv`  
- `trend_significance_mhw_metrics_<REGION>.csv`  

---

## 🧠 Methodology Summary

Implements the **Hobday et al. (2016)** Marine Heatwave definition:

- Daily climatology computed over a user-defined baseline period  
- Threshold = daily percentile (default: 90th)  
- A marine heatwave (MHW) is detected when:
  - SST > threshold  
  - for **N** consecutive days (default: 5)  

For each event, the toolkit computes:

- Duration  
- Mean intensity  
- Maximum intensity  
- Cumulative intensity  

Annual summary metrics are then extracted, and trends are estimated using linear regression, with significance evaluated using the **Mann–Kendall** test.

---

## 📄 Requirements (requirements.txt)

The project expects a `requirements.txt` similar to:

    numpy
    pandas
    xarray
    dask
    netCDF4
    scipy
    matplotlib
    statsmodels
    tqdm
    tk

(You can extend this list if you add extra functionality.)

---

## 📄 License

This project is licensed under the **MIT License**.  
See the file `LICENSE` for the full text.

---

## 🙌 Acknowledgements

- NOAA Physical Sciences Laboratory – OISST v2.1  
- Hobday et al. (2016) Marine Heatwave framework  
- Xarray, Dask, NumPy, SciPy, Matplotlib  

If you use this toolkit in research, please cite **NOAA OISST v2.1** and **Hobday et al. (2016)**.
