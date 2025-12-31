# ml_plotting.py
#
# Visualization functions for ML model evaluation and predictions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Optional
import seaborn as sns


def plot_feature_importance(
        feature_importance: pd.Series,
        top_n: int = 20,
        save_path: Optional[str] = None,
):
    """
    Plot feature importance from Random Forest model.

    Parameters
    ----------
    feature_importance : pd.Series
        Feature importance scores (sorted descending).
    top_n : int
        Number of top features to display.
    save_path : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = feature_importance.head(top_n)

    ax.barh(range(len(top_features)), top_features.values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot → {save_path}")

    plt.close()


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
):
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    save_path : str, optional
        Path to save figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No MHW', 'MHW'],
        yticklabels=['No MHW', 'MHW'],
        ax=ax, cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix → {save_path}")

    plt.close()


def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for positive class.
    save_path : str, optional
        Path to save figure.
    """
    from sklearn.metrics import roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve → {save_path}")

    plt.close()


def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for positive class.
    save_path : str, optional
        Path to save figure.
    """
    from sklearn.metrics import average_precision_score

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(recall, precision, color='darkgreen', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')

    # Baseline: proportion of positive class
    baseline = y_true.sum() / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
            label=f'Random classifier (AP = {baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved precision-recall curve → {save_path}")

    plt.close()


def plot_prediction_timeline(
        time_index: pd.DatetimeIndex,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        sst_values: Optional[np.ndarray] = None,
        threshold_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        region_name: str = "",
        forecast_window: int = 7,
):
    """
    Plot prediction timeline showing actual vs predicted MHW events.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Time index for predictions.
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities.
    sst_values : np.ndarray, optional
        SST values to plot in background.
    threshold_values : np.ndarray, optional
        Threshold values to plot.
    save_path : str, optional
        Path to save figure.
    region_name : str
        Name of region for title.
    forecast_window : int
        Forecast window in days.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Panel 1: SST and threshold (if provided)
    if sst_values is not None:
        axes[0].plot(time_index, sst_values, 'b-', linewidth=0.8, label='SST', alpha=0.7)
        if threshold_values is not None:
            axes[0].plot(time_index, threshold_values, 'r--', linewidth=1.2,
                         label='MHW Threshold', alpha=0.8)
        axes[0].set_ylabel('SST (°C)', fontsize=11)
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f'Marine Heatwave Predictions - {region_name}',
                          fontsize=13, fontweight='bold')

    # Panel 2: Actual MHW events vs predictions
    actual_events = y_true.astype(bool)
    predicted_events = y_pred.astype(bool)

    # Create event periods for shading
    axes[1].fill_between(time_index, 0, 1, where=actual_events,
                         color='red', alpha=0.3, label='Actual MHW (will occur)')
    axes[1].fill_between(time_index, 0, 1, where=predicted_events,
                         color='blue', alpha=0.2, label='Predicted MHW')

    axes[1].set_ylabel('Event Status', fontsize=11)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['No MHW', 'MHW'])
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(alpha=0.3)

    # Panel 3: Prediction probability
    axes[2].plot(time_index, y_proba, 'purple', linewidth=1.2, alpha=0.7,
                 label=f'P(MHW in {forecast_window}d)')
    axes[2].axhline(0.5, color='k', linestyle='--', linewidth=1,
                    alpha=0.5, label='Decision threshold (0.5)')
    axes[2].fill_between(time_index, 0, 1, where=actual_events,
                         color='red', alpha=0.15)

    axes[2].set_xlabel('Date', fontsize=11)
    axes[2].set_ylabel('Probability', fontsize=11)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc='upper left', fontsize=10)
    axes[2].grid(alpha=0.3)

    # Format x-axis
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction timeline → {save_path}")

    plt.close()


def plot_cv_results(
        metrics_list: list,
        save_path: Optional[str] = None,
):
    """
    Plot cross-validation results showing metric distributions.

    Parameters
    ----------
    metrics_list : list
        List of metric dictionaries from cross-validation.
    save_path : str, optional
        Path to save figure.
    """
    # Extract metrics across folds
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_values = {name: [m[name] for m in metrics_list] for name in metric_names}

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = range(len(metric_names))
    bp = ax.boxplot(
        [metric_values[name] for name in metric_names],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
    )

    # Style boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names],
                       rotation=15, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Validation Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CV results plot → {save_path}")

    plt.close()


def create_ml_evaluation_plots(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        feature_importance: pd.Series,
        plots_dir: str,
        region_name: str = "",
):
    """
    Create comprehensive ML evaluation plots.

    Parameters
    ----------
    y_test : np.ndarray
        True test labels.
    y_pred : np.ndarray
        Predicted test labels.
    y_proba : np.ndarray
        Predicted probabilities.
    feature_importance : pd.Series
        Feature importance scores.
    plots_dir : str
        Directory to save plots.
    region_name : str
        Name of region for filenames.
    """
    import os
    safe_region = region_name.replace(" ", "_")

    print("\n=== CREATING ML EVALUATION PLOTS ===")

    # Feature importance
    plot_feature_importance(
        feature_importance,
        top_n=20,
        save_path=os.path.join(plots_dir, f"ml_feature_importance_{safe_region}.png")
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(plots_dir, f"ml_confusion_matrix_{safe_region}.png")
    )

    # ROC curve
    plot_roc_curve(
        y_test, y_proba,
        save_path=os.path.join(plots_dir, f"ml_roc_curve_{safe_region}.png")
    )

    # Precision-recall curve
    plot_precision_recall_curve(
        y_test, y_proba,
        save_path=os.path.join(plots_dir, f"ml_pr_curve_{safe_region}.png")
    )

    print("=" * 50 + "\n")