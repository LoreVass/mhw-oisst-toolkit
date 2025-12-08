# 🌊 Marine Heatwave (MHW) Detection Toolkit – OISST v2.1
A complete Python toolkit for detecting and analyzing **marine heatwaves (MHWs)** using **NOAA OISST v2.1 daily sea surface temperature**.

---

## 🔥 Features
- Automated NOAA OISST year-check system
- Fast SST loading over any custom or preset region
- Daily climatology & percentile threshold (Hobday et al., 2016)
- Full marine heatwave event detection
- Annual summaries of MHW metrics
- Trend analysis (linear regression + Mann–Kendall test)
- Clean Tkinter graphical interface

❗ *Note:* The interactive anomaly-map system was removed for stability.
Use **Google Earth Engine** or **QGIS** for anomaly visualization.

---

## 📁 Project Structure
mhw-oisst-toolkit/
│
├── scripts/
│ ├── main.py # GUI + full analysis pipeline
│ ├── data_io.py # OISST loading utilities
│ ├── mhw_core.py # Climatology + threshold + MHW detection
│ ├── stats_tool.py # Trend + MK significance testing
│ ├── plotting.py # Plot generation (PNG)
│
├── plots/ # Auto-generated figures
├── tables/ # Auto-generated CSV analysis tables
│
├── README.md
├── requirements.txt
└── LICENSE

yaml
Copia codice

---

## 🚀 Installation

### 1. Clone repository
```bash
git clone https://github.com/<YOUR_USERNAME>/mhw-oisst-toolkit.git
cd mhw-oisst-toolkit
2. Create virtual environment
bash
Copia codice
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
3. Install dependencies
bash
Copia codice
pip install -r requirements.txt
🌐 NOAA OISST v2.1 Data
Download daily files (sst.day.mean.YYYY.nc) from:
https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html

Place them in a folder such as:

javascript
Copia codice
C:/Users/<YOU>/Data/OISST/
The toolkit verifies all required years automatically.

🖥 Running the Toolkit (GUI)
Launch:

bash
Copia codice
python scripts/main.py
You will select:

Region (preset or custom)

Latitude/longitude bounds

Analysis period

Baseline climatology period

Percentile threshold (default 0.9)

Minimum heatwave duration (default 5 days)

Folder containing the NOAA OISST files

Outputs are saved to:

plots/

tables/

📊 Output Files
Figures (PNG)
mhw_timeseries_<REGION>.png

seasonal_cycle_<REGION>.png

trend_sst_annual_mean_<REGION>.png

trend_sst_annual_min_<REGION>.png

trend_sst_annual_max_<REGION>.png

trend_mhw_n_events_<REGION>.png

trend_mhw_total_mhw_days_<REGION>.png

trend_mhw_max_intensity_<REGION>.png

trend_mhw_mean_intensity_<REGION>.png

trend_mhw_longest_event_<REGION>.png

Tables (CSV)
mhw_events_<REGION>.csv

mhw_yearly_summary_<REGION>.csv

trend_significance_sst_<REGION>.csv

trend_significance_mhw_metrics_<REGION>.csv

🧠 Methodology Summary
Implements the official Hobday et al. (2016) Marine Heatwave definition:

Daily climatology computed over user-defined baseline

Threshold = daily percentile (default: 90th)

MHW = SST exceeds threshold for ≥ N consecutive days

Event metrics computed:

Duration

Mean intensity

Maximum intensity

Cumulative intensity

Annual summary metrics automatically extracted

Trend analysis via linear regression

Significance via Mann–Kendall

📄 requirements.txt
nginx
Copia codice
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
📄 LICENSE (MIT)
sql
Copia codice
MIT License

Copyright (c) 2025 Lorenzo Vassura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
🙌 Acknowledgements
NOAA Physical Sciences Laboratory

Hobday et al. (2016)

Xarray, Dask, NumPy, SciPy, Matplotlib

If used in research, please cite NOAA OISST v2.1 and Hobday et al. (2016).

yaml
Copia codice

---
