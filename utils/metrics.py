"""Evaluation metrics for stroke outcome prediction."""
from typing import Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """Compute all metrics. y_pred: binary, y_prob: probability of class 1."""
    # Handle single-class edge case
    if len(np.unique(y_true)) < 2:
        return {
            "auc": 0.5, "accuracy": 0.0, "f1_macro": 0.0,
            "precision": 0.0, "recall": 0.0,
        }
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        start: float = 0.1, end: float = 0.9,
                        step: float = 0.01) -> Tuple[float, float]:
    """Search for threshold maximizing macro F1."""
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(start, end + step, step):
        pred = (y_prob >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        f1 = f1_score(y_true, pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh), float(best_f1)
