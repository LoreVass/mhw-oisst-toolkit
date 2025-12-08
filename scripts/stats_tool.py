# stats_tool.py
#
# Trend and change-point analysis:
# - Mann–Kendall test
# - Sen's slope
# - OLS linear trend
# - Pettitt test
# - Helper to run tests on SST and yearly MHW metrics

import math
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_kendall_test(series: pd.Series) -> dict:
    """
    Non-parametric Mann–Kendall trend test + Kendall's tau.
    Returns dict with tau, S, varS, Z, p.
    """
    s = series.dropna()
    n = len(s)
    if n < 3:
        return {
            "n": n,
            "mk_tau": float("nan"),
            "mk_S": float("nan"),
            "mk_varS": float("nan"),
            "mk_Z": float("nan"),
            "mk_p": float("nan"),
        }

    y = s.values

    # S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[j] > y[i]:
                S += 1
            elif y[j] < y[i]:
                S -= 1

    # Tie correction
    unique, counts = np.unique(y, return_counts=True)
    tie_term = 0
    for c in counts:
        if c > 1:
            tie_term += c * (c - 1) * (2 * c + 5)

    varS = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if S > 0:
        Z = (S - 1) / math.sqrt(varS)
    elif S < 0:
        Z = (S + 1) / math.sqrt(varS)
    else:
        Z = 0.0

    p = 2.0 * (1.0 - _normal_cdf(abs(Z)))
    tau = S / (0.5 * n * (n - 1))

    return {
        "n": n,
        "mk_tau": float(tau),
        "mk_S": float(S),
        "mk_varS": float(varS),
        "mk_Z": float(Z),
        "mk_p": float(p),
    }


def sen_slope(series: pd.Series, x_years: pd.Series) -> float:
    """
    Sen's slope (median of all pairwise slopes).
    Slope units: same as series units per year
    if x_years is in years.
    """
    s = series.dropna()
    x = x_years.loc[s.index].values
    y = s.values
    n = len(s)
    if n < 2:
        return float("nan")

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)

    if not slopes:
        return float("nan")
    return float(np.median(slopes))


def linear_trend(series: pd.Series, x_years: pd.Series) -> dict:
    """
    Simple OLS linear trend y = a + b * x.

    Returns
    -------
    {'lin_slope', 'lin_intercept', 'lin_R2', 'lin_p'}
    """
    s = series.dropna()
    x = x_years.loc[s.index].values.astype(float)
    y = s.values.astype(float)
    n = len(s)
    if n < 3:
        return {
            "lin_slope": float("nan"),
            "lin_intercept": float("nan"),
            "lin_R2": float("nan"),
            "lin_p": float("nan"),
        }

    # Fit y = b*x + a
    b, a = np.polyfit(x, y, 1)
    y_hat = a + b * x

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    R2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    # Standard error of slope
    df = n - 2
    if df <= 0:
        return {
            "lin_slope": float(b),
            "lin_intercept": float(a),
            "lin_R2": float(R2),
            "lin_p": float("nan"),
        }

    s_err2 = ss_res / df
    x_mean = x.mean()
    ssx = ((x - x_mean) ** 2).sum()
    if ssx == 0:
        se_b = float("nan")
        p = float("nan")
    else:
        se_b = math.sqrt(s_err2 / ssx)
        if se_b == 0:
            p = float("nan")
        else:
            t_stat = b / se_b
            p = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    return {
        "lin_slope": float(b),
        "lin_intercept": float(a),
        "lin_R2": float(R2),
        "lin_p": float(p),
    }


def pettitt_test(series: pd.Series) -> dict:
    """
    Pettitt change-point test.
    Returns change_index (int), change_value (year), and p-value.
    """
    s = series.dropna()
    n = len(s)
    if n < 2:
        return {
            "pettitt_index": None,
            "pettitt_x": None,
            "pettitt_p": float("nan"),
        }

    y = s.values
    K = [0] * n
    for t in range(n):
        sum_sign = 0
        for i in range(t + 1):
            for j in range(t + 1, n):
                if y[j] > y[i]:
                    sum_sign += 1
                elif y[j] < y[i]:
                    sum_sign -= 1
        K[t] = sum_sign

    absK = [abs(val) for val in K]
    K_max = max(absK)
    t_index = absK.index(K_max)

    # Approximate p-value
    p = 2.0 * math.exp((-6.0 * (K_max ** 2)) / (n**3 + n**2))

    # Get corresponding x-coordinate (e.g. year)
    x_vals = s.index.values
    x_change = x_vals[t_index]

    return {
        "pettitt_index": int(t_index),
        "pettitt_x": x_change,
        "pettitt_p": float(p),
    }


def _plot_trend_series(
    years: np.ndarray,
    values: np.ndarray,
    metric_label: str,
    out_path: str,
    lin_slope: float,
    lin_intercept: float,
    lin_p: float,
    mk_p: float,
    pettitt_x,
):
    """
    Generic plot for a yearly series with:
    - scatter + line
    - linear trend line
    - vertical Pettitt change-point (if any)
    - annotation with slope and p-values
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(years, values, "-o", label=metric_label)

    # Trend line
    if not math.isnan(lin_slope) and not math.isnan(lin_intercept):
        x_line = np.array([years.min(), years.max()])
        y_line = lin_intercept + lin_slope * x_line
        ax.plot(x_line, y_line, "--", label="Linear trend")

    # Pettitt change-point
    if pettitt_x is not None:
        try:
            cx = int(pettitt_x)
            ax.axvline(cx, linestyle=":", label=f"Pettitt change ({cx})")
        except Exception:
            pass

    ax.set_xlabel("Year")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} – Trend & Change-point")

    txt = (
        f"slope = {lin_slope:.4g} /yr\n"
        f"lin p = {lin_p:.3g}\n"
        f"MK p  = {mk_p:.3g}"
    )
    ax.text(
        0.01,
        0.99,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved trend plot to {out_path}")


def run_significance_tests(
    sst_ts: xr.DataArray,
    summary_df: pd.DataFrame,
    plots_dir: str = "plots",
    tables_dir: str = "tables",
):
    """
    1) Trend tests on annual mean/min/max SST (with plots).
    2) Trend tests on yearly MHW metrics (summary_df, with plots).
    3) Pettitt change-point tests on each of those series.

    Saves (with region-specific suffix):
    - trend_significance_sst_{region}.csv
    - trend_significance_mhw_metrics_{region}.csv
    - trend_sst_*_{region}.png
    - trend_mhw_*_{region}.png
    """
    # Region tag (if available)
    if "region" in summary_df.columns:
        safe_region = str(summary_df["region"].iloc[0]).replace(" ", "_")
    else:
        safe_region = "unknown_region"

    # ---- 1. Annual SST metrics ----
    sst_annual_mean = sst_ts.resample(time="YE").mean()
    sst_annual_max = sst_ts.resample(time="YE").max()
    sst_annual_min = sst_ts.resample(time="YE").min()

    years_sst = pd.to_datetime(sst_annual_mean.time.values).year
    idx_sst = pd.Index(years_sst, name="year")

    sst_mean_series = pd.Series(sst_annual_mean.values, index=idx_sst, name="annual_mean_sst")
    sst_max_series = pd.Series(sst_annual_max.values, index=idx_sst, name="annual_max_sst")
    sst_min_series = pd.Series(sst_annual_min.values, index=idx_sst, name="annual_min_sst")

    sst_metrics = {
        "annual_mean_sst": sst_mean_series,
        "annual_max_sst": sst_max_series,
        "annual_min_sst": sst_min_series,
    }

    results_sst = []
    for metric_name, series in sst_metrics.items():
        years = series.index.to_series()

        mk_res = mann_kendall_test(series)
        sen = sen_slope(series, years)
        lin_res = linear_trend(series, years)
        pt_res = pettitt_test(series)

        row = {
            "metric": metric_name,
            "n": mk_res["n"],
            "mk_tau": mk_res["mk_tau"],
            "mk_S": mk_res["mk_S"],
            "mk_varS": mk_res["mk_varS"],
            "mk_Z": mk_res["mk_Z"],
            "mk_p": mk_res["mk_p"],
            "sen_slope_per_year": sen,
        }
        row.update(lin_res)
        row["pettitt_index"] = pt_res["pettitt_index"]
        row["pettitt_x"] = pt_res["pettitt_x"]
        row["pettitt_p"] = pt_res["pettitt_p"]

        results_sst.append(row)

        out_name = os.path.join(plots_dir, f"trend_sst_{metric_name}_{safe_region}.png")
        _plot_trend_series(
            years=series.index.values.astype(int),
            values=series.values,
            metric_label=metric_name,
            out_path=out_name,
            lin_slope=lin_res["lin_slope"],
            lin_intercept=lin_res["lin_intercept"],
            lin_p=lin_res["lin_p"],
            mk_p=mk_res["mk_p"],
            pettitt_x=pt_res["pettitt_x"],
        )

    results_sst_df = pd.DataFrame(results_sst)
    sst_out_path = os.path.join(tables_dir, f"trend_significance_sst_{safe_region}.csv")
    results_sst_df.to_csv(sst_out_path, index=False)
    print(f"Saved SST trend significance table to {sst_out_path}")

    # ---- 2. MHW metrics per year ----
    if summary_df is None or summary_df.empty:
        print("No MHW summary_df provided or it is empty; skipping MHW trend tests.")
        return

    # Only numeric columns (exclude 'year', 'region', etc.)
    numeric_cols = summary_df.select_dtypes(include=["number"]).columns.tolist()
    mhw_metrics = [c for c in numeric_cols if c != "year"]

    results_mhw = []
    for metric_name in mhw_metrics:
        series = summary_df.set_index("year")[metric_name]
        years_idx = series.index.to_series()

        mk_res = mann_kendall_test(series)
        sen = sen_slope(series, years_idx)
        lin_res = linear_trend(series, years_idx)
        pt_res = pettitt_test(series)

        row = {
            "metric": metric_name,
            "n": mk_res["n"],
            "mk_tau": mk_res["mk_tau"],
            "mk_S": mk_res["mk_S"],
            "mk_varS": mk_res["mk_varS"],
            "mk_Z": mk_res["mk_Z"],
            "mk_p": mk_res["mk_p"],
            "sen_slope_per_year": sen,
        }
        row.update(lin_res)
        row["pettitt_index"] = pt_res["pettitt_index"]
        row["pettitt_x"] = pt_res["pettitt_x"]
        row["pettitt_p"] = pt_res["pettitt_p"]

        results_mhw.append(row)

        out_name = os.path.join(plots_dir, f"trend_mhw_{metric_name}_{safe_region}.png")
        _plot_trend_series(
            years=series.index.values.astype(int),
            values=series.values,
            metric_label=f"MHW {metric_name}",
            out_path=out_name,
            lin_slope=lin_res["lin_slope"],
            lin_intercept=lin_res["lin_intercept"],
            lin_p=lin_res["lin_p"],
            mk_p=mk_res["mk_p"],
            pettitt_x=pt_res["pettitt_x"],
        )

    results_mhw_df = pd.DataFrame(results_mhw)
    mhw_out_path = os.path.join(tables_dir, f"trend_significance_mhw_metrics_{safe_region}.csv")
    results_mhw_df.to_csv(mhw_out_path, index=False)
    print(f"Saved MHW trend significance table to {mhw_out_path}")
