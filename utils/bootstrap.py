"""Bootstrap 95% confidence intervals for all metrics."""
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score
from .metrics import compute_metrics


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 n_iter: int = 1000, alpha: float = 0.05,
                 seed: int = 42,
                 fixed_threshold: Optional[float] = None) -> dict:
    """Compute 95% CI for all metrics via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metric_names = ["auc", "accuracy", "f1_macro", "precision", "recall"]
    all_metrics: dict = {k: [] for k in metric_names}

    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        threshold = fixed_threshold if fixed_threshold is not None else _find_thresh(yt, yp)[0]
        pred = (yp >= threshold).astype(int)
        m = compute_metrics(yt, pred, yp)
        for k in metric_names:
            all_metrics[k].append(m[k])

    results: dict = {}
    for k in metric_names:
        vals = np.array(all_metrics[k])
        if len(vals) == 0:
            results[k] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                          "ci_str": "N/A (no valid bootstrap samples)"}
            continue
        lower = np.percentile(vals, 100 * alpha / 2)
        upper = np.percentile(vals, 100 * (1 - alpha / 2))
        mean = np.mean(vals)
        results[k] = {
            "mean": round(float(mean), 4),
            "ci_lower": round(float(lower), 4),
            "ci_upper": round(float(upper), 4),
            "ci_str": f"{mean:.4f} (95% CI: {lower:.4f}-{upper:.4f})",
        }
    return results


def _find_thresh(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        pred = (y_prob >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        f1 = f1_score(y_true, pred, average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)
