# ml_workflow.py
#
# Standalone script for running ML-based marine heatwave prediction.
# Can be run independently or integrated into main.py pipeline.

import os
import sys
import argparse
import pandas as pd
import xarray as xr

from ml_features import prepare_ml_dataset
from ml_prediction import MHWPredictor, cross_validate_temporal
from ml_plotting import create_ml_evaluation_plots, plot_prediction_timeline, plot_cv_results


def run_ml_workflow(
        sst_ts: xr.DataArray,
        clim_ts: xr.DataArray,
        threshold_ts: xr.DataArray,
        plots_dir: str,
        tables_dir: str,
        region_name: str = "",
        forecast_window: int = 7,
        min_event_length: int = 5,
        train_end_date: str = None,
        run_cv: bool = True,
):
    """
    Complete ML workflow for marine heatwave prediction.

    Parameters
    ----------
    sst_ts : xr.DataArray
        Daily SST time series.
    clim_ts : xr.DataArray
        Daily climatological mean.
    threshold_ts : xr.DataArray
        Daily percentile threshold.
    plots_dir : str
        Directory to save plots.
    tables_dir : str
        Directory to save tables/results.
    region_name : str
        Name of the region for titles and filenames.
    forecast_window : int
        Days ahead to predict (default: 7).
    min_event_length : int
        Minimum MHW duration (default: 5 days).
    train_end_date : str, optional
        End date for training set (format: 'YYYY-MM-DD').
        If None, uses 80% of data for training.
    run_cv : bool
        Whether to run cross-validation (default: True).
    """
    safe_region = region_name.replace(" ", "_")

    print("\n" + "=" * 70)
    print("MACHINE LEARNING MARINE HEATWAVE PREDICTION")
    print("=" * 70)

    # 1. Prepare ML dataset
    print("\n[1/5] Preparing ML dataset...")
    X_train, X_test, y_train, y_test = prepare_ml_dataset(
        sst_ts, clim_ts, threshold_ts,
        forecast_window=forecast_window,
        min_event_length=min_event_length,
        train_end_date=train_end_date,
    )

    # 2. Train model
    print("\n[2/5] Training Random Forest model...")
    model = MHWPredictor(
        forecast_window=forecast_window,
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 3. Evaluate on test set
    print("\n[3/5] Evaluating model on test set...")
    metrics = model.evaluate(X_test, y_test, verbose=True)

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(tables_dir, f"ml_metrics_{safe_region}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics → {metrics_path}")

    # Save feature importance
    importance_path = os.path.join(tables_dir, f"ml_feature_importance_{safe_region}.csv")
    model.get_feature_importance(top_n=30).to_csv(importance_path)
    print(f"Saved feature importance → {importance_path}")

    # 4. Create evaluation plots
    print("\n[4/5] Creating evaluation plots...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    create_ml_evaluation_plots(
        y_test.values, y_pred, y_proba,
        model.get_feature_importance(top_n=20),
        plots_dir, region_name
    )

    # Prediction timeline plot
    # Get SST and threshold values for test period
    test_times = X_test.index
    sst_test = sst_ts.sel(time=test_times).values
    thresh_test = threshold_ts.sel(time=test_times).values

    timeline_path = os.path.join(plots_dir, f"ml_prediction_timeline_{safe_region}.png")
    plot_prediction_timeline(
        test_times, y_test.values, y_pred, y_proba,
        sst_values=sst_test,
        threshold_values=thresh_test,
        save_path=timeline_path,
        region_name=region_name,
        forecast_window=forecast_window,
    )

    # 5. Cross-validation (optional)
    if run_cv:
        print("\n[5/5] Running temporal cross-validation...")

        # Combine train and test for CV
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])

        cv_models, cv_metrics = cross_validate_temporal(
            X_full, y_full,
            n_splits=5,
            forecast_window=forecast_window,
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )

        # Save CV results
        cv_df = pd.DataFrame(cv_metrics)
        cv_path = os.path.join(tables_dir, f"ml_cv_results_{safe_region}.csv")
        cv_df.to_csv(cv_path, index=False)
        print(f"Saved CV results → {cv_path}")

        # Plot CV results
        cv_plot_path = os.path.join(plots_dir, f"ml_cv_results_{safe_region}.png")
        plot_cv_results(cv_metrics, save_path=cv_plot_path)
    else:
        print("\n[5/5] Skipping cross-validation (run_cv=False)")

    print("\n" + "=" * 70)
    print("ML WORKFLOW COMPLETE")
    print("=" * 70 + "\n")

    return model, metrics


def main():
    """
    Standalone CLI for ML workflow.
    Requires pre-computed SST, climatology, and threshold data.
    """
    parser = argparse.ArgumentParser(
        description="ML-based Marine Heatwave Prediction"
    )
    parser.add_argument(
        "--sst", required=True,
        help="Path to SST time series (NetCDF or pickle)"
    )
    parser.add_argument(
        "--clim", required=True,
        help="Path to climatology time series (NetCDF or pickle)"
    )
    parser.add_argument(
        "--threshold", required=True,
        help="Path to threshold time series (NetCDF or pickle)"
    )
    parser.add_argument(
        "--plots-dir", default="plots",
        help="Directory for output plots"
    )
    parser.add_argument(
        "--tables-dir", default="tables",
        help="Directory for output tables"
    )
    parser.add_argument(
        "--region-name", default="Region",
        help="Name of the region"
    )
    parser.add_argument(
        "--forecast-window", type=int, default=7,
        help="Forecast window in days (default: 7)"
    )
    parser.add_argument(
        "--min-event-length", type=int, default=5,
        help="Minimum MHW duration in days (default: 5)"
    )
    parser.add_argument(
        "--train-end-date", default=None,
        help="End date for training (YYYY-MM-DD). If not provided, uses 80/20 split"
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip cross-validation"
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.tables_dir, exist_ok=True)

    # Load data
    print(f"Loading SST from {args.sst}...")
    if args.sst.endswith('.nc'):
        sst_ts = xr.open_dataarray(args.sst)
    else:
        sst_ts = pd.read_pickle(args.sst)

    print(f"Loading climatology from {args.clim}...")
    if args.clim.endswith('.nc'):
        clim_ts = xr.open_dataarray(args.clim)
    else:
        clim_ts = pd.read_pickle(args.clim)

    print(f"Loading threshold from {args.threshold}...")
    if args.threshold.endswith('.nc'):
        threshold_ts = xr.open_dataarray(args.threshold)
    else:
        threshold_ts = pd.read_pickle(args.threshold)

    # Run workflow
    run_ml_workflow(
        sst_ts, clim_ts, threshold_ts,
        plots_dir=args.plots_dir,
        tables_dir=args.tables_dir,
        region_name=args.region_name,
        forecast_window=args.forecast_window,
        min_event_length=args.min_event_length,
        train_end_date=args.train_end_date,
        run_cv=not args.no_cv,
    )


if __name__ == "__main__":
    main()