# ml_prediction.py
#
# Machine learning models for marine heatwave prediction.
# Uses Random Forest classifier to predict MHW occurrence within a forecast window.

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


class MHWPredictor:
    """
    Random Forest classifier for predicting marine heatwave events.

    Attributes
    ----------
    model : RandomForestClassifier
        The underlying Random Forest model.
    forecast_window : int
        Number of days ahead the model predicts.
    feature_names : list
        Names of features used for training.
    feature_importance : pd.Series
        Feature importance scores from the trained model.
    """

    def __init__(
            self,
            forecast_window: int = 7,
            n_estimators: int = 100,
            max_depth: int = 10,
            min_samples_split: int = 20,
            min_samples_leaf: int = 10,
            random_state: int = 42,
    ):
        """
        Initialize MHW predictor.

        Parameters
        ----------
        forecast_window : int
            Days ahead to predict (default: 7).
        n_estimators : int
            Number of trees in the forest (default: 100).
        max_depth : int
            Maximum tree depth (default: 10).
        min_samples_split : int
            Minimum samples to split a node (default: 20).
        min_samples_leaf : int
            Minimum samples in a leaf (default: 10).
        random_state : int
            Random seed for reproducibility.
        """
        self.forecast_window = forecast_window
        self.random_state = random_state

        # Initialize Random Forest with conservative parameters to avoid overfitting
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1,  # Use all CPU cores
        )

        self.feature_names = None
        self.feature_importance = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'MHWPredictor':
        """
        Train the Random Forest model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training targets (binary: 0 or 1).

        Returns
        -------
        self : MHWPredictor
            The fitted model.
        """
        print(f"\n=== TRAINING RANDOM FOREST ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {len(X_train.columns)}")
        print(f"Target distribution: {y_train.sum()} positive / {len(y_train)} total "
              f"({100 * y_train.mean():.1f}%)")

        self.feature_names = X_train.columns.tolist()

        # Fit model
        self.model.fit(X_train, y_train)

        # Extract feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        print(f"\nTop 10 most important features:")
        for i, (feat, imp) in enumerate(self.feature_importance.head(10).items(), 1):
            print(f"{i:2d}. {feat:40s} {imp:.4f}")

        print("=" * 50 + "\n")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary class (MHW / no MHW).

        Parameters
        ----------
        X : pd.DataFrame
            Features.

        Returns
        -------
        predictions : np.ndarray
            Binary predictions (0 or 1).
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of MHW occurrence.

        Parameters
        ----------
        X : pd.DataFrame
            Features.

        Returns
        -------
        probabilities : np.ndarray
            Array of shape (n_samples, 2) with probabilities for each class.
            [:, 1] gives P(MHW).
        """
        return self.model.predict_proba(X)

    def evaluate(
            self,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test targets.
        verbose : bool
            Whether to print detailed results.

        Returns
        -------
        metrics : dict
            Dictionary of performance metrics.
        """
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]  # Probability of MHW

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # ROC-AUC (if we have both classes)
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)
        except ValueError:
            roc_auc = np.nan
            avg_precision = np.nan

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
        }

        if verbose:
            print("\n=== MODEL EVALUATION ===")
            print(f"Test samples: {len(y_test)}")
            print(f"Actual MHW events: {y_test.sum()}")
            print(f"\nConfusion Matrix:")
            print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
            print(f"  FN: {fn:5d}  |  TP: {tp:5d}")
            print(f"\nMetrics:")
            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}  (of predicted MHWs, how many were correct)")
            print(f"  Recall:    {recall:.3f}  (of actual MHWs, how many were caught)")
            print(f"  F1 Score:  {f1:.3f}")
            print(f"  ROC-AUC:   {roc_auc:.3f}")
            print(f"  Avg Prec:  {avg_precision:.3f}")
            print("=" * 50 + "\n")

            # Classification report
            print("Detailed Classification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=['No MHW', 'MHW'],
                digits=3
            ))

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """
        Get feature importance scores.

        Parameters
        ----------
        top_n : int
            Number of top features to return.

        Returns
        -------
        importance : pd.Series
            Top N features and their importance scores.
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first.")

        return self.feature_importance.head(top_n)


def cross_validate_temporal(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        forecast_window: int = 7,
        **model_kwargs,
) -> Tuple[list, list]:
    """
    Perform temporal cross-validation.

    Uses expanding window approach: each fold uses all past data for training
    and the next time window for testing.

    Parameters
    ----------
    X : pd.DataFrame
        Complete feature matrix.
    y : pd.Series
        Complete target vector.
    n_splits : int
        Number of temporal folds.
    forecast_window : int
        Forecast window for the model.
    **model_kwargs
        Additional arguments for MHWPredictor.

    Returns
    -------
    models : list
        List of trained models (one per fold).
    metrics_list : list
        List of metrics dictionaries (one per fold).
    """
    print(f"\n=== TEMPORAL CROSS-VALIDATION ({n_splits} folds) ===\n")

    # Determine fold boundaries
    total_samples = len(X)
    fold_size = total_samples // (n_splits + 1)  # Leave room for initial training

    models = []
    metrics_list = []

    for fold in range(n_splits):
        # Expanding window: train on all data up to fold boundary
        train_end_idx = (fold + 2) * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, total_samples)

        X_train = X.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]

        print(f"Fold {fold + 1}/{n_splits}:")
        print(f"  Train: {X.index[0].date()} to {X.index[train_end_idx - 1].date()} "
              f"({len(X_train)} samples)")
        print(f"  Test:  {X.index[test_start_idx].date()} to {X.index[test_end_idx - 1].date()} "
              f"({len(X_test)} samples)")

        # Train model
        model = MHWPredictor(forecast_window=forecast_window, **model_kwargs)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = model.evaluate(X_test, y_test, verbose=False)
        metrics_list.append(metrics)
        models.append(model)

        print(f"  F1: {metrics['f1_score']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}\n")

    # Summary statistics
    print("\nCross-validation summary:")
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        values = [m[metric_name] for m in metrics_list]
        print(f"  {metric_name:15s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    print("=" * 50 + "\n")

    return models, metrics_list