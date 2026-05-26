"""Per-dataset evaluation with bootstrap 95% CI and publication-quality plots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.bootstrap import bootstrap_ci
from utils.metrics import compute_metrics, find_best_threshold
from utils.plotting import plot_roc_curve, plot_confusion_matrix, plot_ablation_bar


def evaluate_fold_results(all_y_true: dict, all_y_prob: dict,
                           results_dir: str, bootstrap_n: int = 1000) -> dict:
    """Compute metrics and bootstrap CIs for each dataset, save plots and CSVs."""
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    final_results: dict = {}

    for ds_name in all_y_true:
        y_true = np.array(all_y_true[ds_name])
        y_prob = np.array(all_y_prob[ds_name])

        best_t, _ = find_best_threshold(y_true, y_prob)
        y_pred = (y_prob >= best_t).astype(int)

        # Bootstrap CIs
        ci_results = bootstrap_ci(y_true, y_prob, n_iter=bootstrap_n)

        # Basic metrics
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["threshold"] = best_t
        final_results[ds_name] = {
            "point_estimates": {k: round(float(v), 4) for k, v in metrics.items()
                                if k != "threshold"},
            "threshold": round(float(best_t), 4),
            "bootstrap_95ci": ci_results,
        }

        print(f"\n{ds_name}:")
        for name, info in ci_results.items():
            print(f"  {name}: {info['ci_str']}")

        # ROC curve
        ds_results = results_root / "joint"
        plot_roc_curve(
            y_true, y_prob,
            title=f"ROC Curve — {ds_name}",
            save_path=str(ds_results / "roc_curves" / f"roc_{ds_name}.tif"),
        )

        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            title=f"Confusion Matrix — {ds_name}",
            save_path=str(ds_results / "confmat" / f"confmat_{ds_name}.tif"),
        )

        # Save metrics CSV
        df = pd.DataFrame([{
            "dataset": ds_name,
            **final_results[ds_name]["point_estimates"],
            "threshold": best_t,
        }])
        for name, info in ci_results.items():
            df[f"{name}_ci_lower"] = info["ci_lower"]
            df[f"{name}_ci_upper"] = info["ci_upper"]
        csv_path = ds_results / f"metrics_{ds_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Save full results JSON
    json_path = results_root / "joint" / "full_results.json"
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nFull results saved: {json_path}")

    return final_results


def evaluate_ablation(ablation_results: dict, results_dir: str) -> None:
    """Bar charts for ablation comparison."""
    results_root = Path(results_dir)

    for ds_name in ["77sets", "ISLE2024"]:
        if ds_name in ablation_results:
            plot_ablation_bar(
                ablation_results[ds_name],
                metric="auc",
                title=f"Ablation Study — {ds_name}",
                save_path=str(results_root / "ablation" / "bar_charts" / f"ablation_{ds_name}.tif"),
            )


if __name__ == "__main__":
    print("Evaluation module ready. Use with crossval_joint.py output.")
