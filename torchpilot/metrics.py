"""Metric functions for classification and regression evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


CLASSIFICATION_PRIMARY = "accuracy"
REGRESSION_PRIMARY = "rmse"


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> dict[str, float]:
    avg = "binary" if n_classes == 2 else "macro"
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    try:
        if n_classes == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            metrics["auroc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
    except ValueError:
        # Happens when val split is missing a class.
        metrics["auroc"] = float("nan")
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def primary_metric_name(task: str) -> str:
    return CLASSIFICATION_PRIMARY if task == "classification" else REGRESSION_PRIMARY


def primary_better(task: str, new: float, old: float) -> bool:
    """Higher accuracy is better; lower RMSE is better."""
    return new > old if task == "classification" else new < old
