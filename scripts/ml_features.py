# ml_features.py
#
# Feature engineering for marine heatwave prediction.
# Extracts temporal and statistical features from SST time series
# that can be used to predict upcoming MHW events.

import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Optional


def compute_rolling_statistics(
        sst_ts: xr.DataArray,
        window_days: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling statistics from SST time series.

    These features capture recent SST trends and variability that may
    precede marine heatwave events.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series (area-averaged).
    window_days : int
        Rolling window size in days (default: 30).

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with temporal features indexed by time.
    """
    # Convert to pandas Series for easier rolling operations
    sst_series = pd.Series(
        data=sst_ts.values,
        index=pd.to_datetime(sst_ts.time.values)
    )

    features = pd.DataFrame(index=sst_series.index)

    # Raw SST
    features['sst'] = sst_series.values

    # Rolling mean (smoothed SST trend)
    features[f'sst_rolling_mean_{window_days}d'] = (
        sst_series.rolling(window=window_days, center=False, min_periods=1).mean()
    )

    # Rolling standard deviation (recent variability)
    features[f'sst_rolling_std_{window_days}d'] = (
        sst_series.rolling(window=window_days, center=False, min_periods=1).std()
    )

    # Rolling min/max (recent extremes)
    features[f'sst_rolling_min_{window_days}d'] = (
        sst_series.rolling(window=window_days, center=False, min_periods=1).min()
    )
    features[f'sst_rolling_max_{window_days}d'] = (
        sst_series.rolling(window=window_days, center=False, min_periods=1).max()
    )

    # Short-term trend (7-day linear slope)
    features['sst_trend_7d'] = (
        sst_series.rolling(window=7, center=False, min_periods=2)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else np.nan, raw=True)
    )

    # Medium-term trend (30-day linear slope)
    features['sst_trend_30d'] = (
        sst_series.rolling(window=30, center=False, min_periods=2)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else np.nan, raw=True)
    )

    return features


def compute_anomaly_features(
        sst_ts: xr.DataArray,
        clim_ts: xr.DataArray,
        threshold_ts: xr.DataArray,
) -> pd.DataFrame:
    """
    Compute anomaly-based features relative to climatology and threshold.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series.
    clim_ts : xr.DataArray
        Daily climatological mean.
    threshold_ts : xr.DataArray
        Daily percentile threshold.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with anomaly features.
    """
    features = pd.DataFrame(index=pd.to_datetime(sst_ts.time.values))

    # SST anomaly (deviation from climatology)
    features['sst_anomaly'] = (sst_ts - clim_ts).values

    # Distance from threshold
    features['sst_above_threshold'] = (sst_ts - threshold_ts).values

    # Binary: is currently above threshold?
    features['is_above_threshold'] = (sst_ts.values > threshold_ts.values).astype(int)

    # Rolling fraction of days above threshold (past 30 days)
    above_thresh_series = pd.Series(
        data=features['is_above_threshold'].values,
        index=features.index
    )
    features['frac_above_threshold_30d'] = (
        above_thresh_series.rolling(window=30, center=False, min_periods=1).mean()
    )

    return features


def compute_seasonal_features(
        time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute seasonal/temporal features (cyclical encoding).

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Time index for the features.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with seasonal features.
    """
    features = pd.DataFrame(index=time_index)

    # Day of year (1-366)
    doy = time_index.dayofyear

    # Cyclical encoding: sin/cos of day-of-year
    # This captures seasonal patterns without introducing discontinuity at year boundaries
    features['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    features['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    # Month as cyclical feature
    month = time_index.month
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)

    return features


def compute_lagged_features(
        features_df: pd.DataFrame,
        lag_days: list = [1, 3, 7, 14],
        columns: list = ['sst', 'sst_anomaly'],
) -> pd.DataFrame:
    """
    Add lagged versions of key features.

    Lagged features allow the model to learn temporal patterns,
    e.g., "SST 7 days ago" as a predictor.

    Parameters
    ----------
    features_df : pd.DataFrame
        Base features.
    lag_days : list
        List of lag periods in days.
    columns : list
        Column names to create lags for.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with additional lagged columns.
    """
    df = features_df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        for lag in lag_days:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)

    return df


def create_ml_features(
        sst_ts: xr.DataArray,
        clim_ts: xr.DataArray,
        threshold_ts: xr.DataArray,
        include_lags: bool = True,
) -> pd.DataFrame:
    """
    Master function to create comprehensive feature set for ML prediction.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series.
    clim_ts : xr.DataArray
        Daily climatological mean.
    threshold_ts : xr.DataArray
        Daily percentile threshold.
    include_lags : bool
        Whether to include lagged features (default: True).

    Returns
    -------
    features_df : pd.DataFrame
        Complete feature matrix ready for ML modeling.
    """
    time_index = pd.to_datetime(sst_ts.time.values)

    # 1. Rolling statistics
    print("Computing rolling statistics...")
    rolling_feats = compute_rolling_statistics(sst_ts, window_days=30)

    # 2. Anomaly features
    print("Computing anomaly features...")
    anomaly_feats = compute_anomaly_features(sst_ts, clim_ts, threshold_ts)

    # 3. Seasonal features
    print("Computing seasonal features...")
    seasonal_feats = compute_seasonal_features(time_index)

    # Combine all features
    features_df = pd.concat([rolling_feats, anomaly_feats, seasonal_feats], axis=1)

    # 4. Lagged features (optional but recommended)
    if include_lags:
        print("Adding lagged features...")
        features_df = compute_lagged_features(
            features_df,
            lag_days=[1, 3, 7, 14],
            columns=['sst', 'sst_anomaly', 'sst_above_threshold']
        )

    return features_df


def create_prediction_targets(
        sst_ts: xr.DataArray,
        threshold_ts: xr.DataArray,
        forecast_window: int = 7,
        min_event_length: int = 5,
) -> pd.Series:
    """
    Create binary target variable: will an MHW occur within the next N days?

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series.
    threshold_ts : xr.DataArray
        Daily threshold time series.
    forecast_window : int
        Prediction window in days (e.g., 7 = predict MHW in next 7 days).
    min_event_length : int
        Minimum consecutive days above threshold to count as MHW.

    Returns
    -------
    targets : pd.Series
        Binary target (1 = MHW will occur, 0 = no MHW) for each day.
    """
    time_index = pd.to_datetime(sst_ts.time.values)
    sst_vals = sst_ts.values
    thresh_vals = threshold_ts.values

    # Identify all days that are part of an MHW event
    above = sst_vals > thresh_vals

    # Find contiguous sequences of min_event_length
    mhw_days = np.zeros(len(above), dtype=bool)

    in_event = False
    event_start = None

    for i in range(len(above)):
        if above[i] and not in_event:
            in_event = True
            event_start = i
        elif not above[i] and in_event:
            # Event ended
            event_length = i - event_start
            if event_length >= min_event_length:
                mhw_days[event_start:i] = True
            in_event = False
            event_start = None

    # Handle case where series ends during event
    if in_event:
        event_length = len(above) - event_start
        if event_length >= min_event_length:
            mhw_days[event_start:] = True

    # Create forward-looking target:
    # For each day t, target=1 if any MHW day occurs in [t+1, t+forecast_window]
    targets = np.zeros(len(time_index), dtype=int)

    for i in range(len(time_index)):
        # Look ahead up to forecast_window days
        end_idx = min(i + forecast_window + 1, len(mhw_days))
        if np.any(mhw_days[i + 1:end_idx]):
            targets[i] = 1

    return pd.Series(targets, index=time_index, name=f'mhw_in_{forecast_window}d')


def prepare_ml_dataset(
        sst_ts: xr.DataArray,
        clim_ts: xr.DataArray,
        threshold_ts: xr.DataArray,
        forecast_window: int = 7,
        min_event_length: int = 5,
        train_end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create complete ML dataset with train/test split.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series.
    clim_ts : xr.DataArray
        Daily climatological mean.
    threshold_ts : xr.DataArray
        Daily percentile threshold.
    forecast_window : int
        Days ahead to predict (default: 7).
    min_event_length : int
        Minimum MHW duration (default: 5 days).
    train_end_date : str, optional
        End date for training set (format: 'YYYY-MM-DD').
        If None, uses 80% of data for training.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training targets.
    y_test : pd.Series
        Test targets.
    """
    print("\n=== PREPARING ML DATASET ===")

    # Create features
    features = create_ml_features(sst_ts, clim_ts, threshold_ts, include_lags=True)

    # Create targets
    print(f"Creating prediction targets (forecast window: {forecast_window} days)...")
    targets = create_prediction_targets(
        sst_ts, threshold_ts, forecast_window, min_event_length
    )

    # Align features and targets (drop rows with NaN from lagged features)
    print("Aligning features and targets...")
    dataset = pd.concat([features, targets], axis=1).dropna()

    X = dataset.drop(columns=[targets.name])
    y = dataset[targets.name]

    print(f"Complete dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Target distribution: {y.sum()} MHW events ({100 * y.mean():.1f}% of days)")

    # Train/test split
    if train_end_date is not None:
        train_mask = X.index <= train_end_date
        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
        print(f"\nTrain/test split by date: {train_end_date}")
    else:
        # Default: 80/20 temporal split
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        split_date = X.index[split_idx]
        print(f"\nTrain/test split: 80/20 temporal (split at {split_date.date()})")

    print(f"Training set:   {len(X_train)} samples ({y_train.sum()} MHW events)")
    print(f"Test set:       {len(X_test)} samples ({y_test.sum()} MHW events)")
    print("=" * 50 + "\n")

    return X_train, X_test, y_train, y_test