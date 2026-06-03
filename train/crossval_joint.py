"""X-Former 5-fold cross-validation with joint training, DDP, TTA, and bootstrap CI."""
import sys
import os
import json
import warnings
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_provider.dataset_77sets import Dataset77sets
from data_provider.dataset_isle2024 import DatasetISLE2024
from data_provider.joint_sampler import (
    BalancedFullSampler,
    InterleavedBatchSampler,
    JointDataset,
    joint_collate_fn,
)
from model.x_former import XFormer
from train.trainer import Trainer
from utils.bootstrap import bootstrap_ci
from utils.metrics import compute_metrics
from utils.plotting import plot_roc_curve, plot_confusion_matrix

warnings.filterwarnings('ignore')


def setup_ddp() -> tuple[int, int]:
    """Initialize DDP from torchrun env vars. Returns (rank, world_size)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_isle_lesion_dir(config: dict, data_root: str) -> str | None:
    """Only expose lesion radiomics when the experiment explicitly enables it."""
    if not config.get("ablation", {}).get("use_lesion", False):
        return None
    lesion_dir = config["datasets"]["isle2024"].get("radiomics_lesion_dir")
    return os.path.join(data_root, lesion_dir) if lesion_dir else None


def prediction_rows(
    fold: int,
    dataset: str,
    subjects: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> list[dict]:
    y_pred = (y_prob >= threshold).astype(int)
    return [
        {
            "fold": fold + 1,
            "dataset": dataset,
            "subject": subject,
            "y_true": int(label),
            "y_prob": float(prob),
            "threshold": float(threshold),
            "y_pred": int(pred),
        }
        for subject, label, prob, pred in zip(subjects, y_true, y_prob, y_pred)
    ]


@torch.no_grad()
def evaluate_test_loader(model: nn.Module, loader: DataLoader, device: torch.device,
                         tta: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return y_true, y_prob, subjects for a test set, with optional TTA."""
    model.eval()
    probs_list, labels_list, subjects_list = [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model(batch, return_domain=False)["logits"]
        prob = torch.softmax(logits, dim=-1)[:, 1]

        if tta:
            batch_tta = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            batch_tta["image"] = torch.flip(batch_tta["image"], dims=[-1])
            logits_tta = model(batch_tta, return_domain=False)["logits"]
            prob_tta = torch.softmax(logits_tta, dim=-1)[:, 1]
            prob = (prob + prob_tta) / 2.0

        probs_list.append(prob.cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())
        subjects = batch.get("subject")
        if subjects is None:
            subjects_list.extend([""] * len(prob))
        elif isinstance(subjects, str):
            subjects_list.append(subjects)
        else:
            subjects_list.extend([str(s) for s in subjects])

    return np.concatenate(labels_list), np.concatenate(probs_list), subjects_list


def run_fold(config: dict, fold: int, device: torch.device,
             ds77_ref: Dataset77sets, ds_isle_ref: DatasetISLE2024,
             rank: int = 0, world_size: int = 1) -> dict:
    """Run a single fold: create splits, train, evaluate on both test sets."""
    cv_cfg = config["crossval"]
    train_cfg = config["training"]
    ds_cfg = config["datasets"]
    paths = config["paths"]
    data_root = paths["data_root"]
    batch_size = train_cfg["batch_size"]
    sampling_cfg = config.get("sampling", {})
    lesion_dir = resolve_isle_lesion_dir(config, data_root)

    labels77 = ds77_ref.labels
    labels_isle = ds_isle_ref.labels
    subjects77 = np.array(ds77_ref._subjects_order)
    subjects_isle = np.array(ds_isle_ref._subjects_order)

    # Create fold splits
    skf77 = StratifiedKFold(n_splits=cv_cfg["n_folds"], shuffle=True, random_state=cv_cfg["seed"])
    skf_isle = StratifiedKFold(n_splits=cv_cfg["n_folds"], shuffle=True, random_state=cv_cfg["seed"])

    fold_idx77 = list(skf77.split(np.zeros(len(ds77_ref)), labels77))[fold]
    fold_idx_isle = list(skf_isle.split(np.zeros(len(ds_isle_ref)), labels_isle))[fold]

    train_idx77, test_idx77 = fold_idx77
    sss77 = StratifiedShuffleSplit(n_splits=1, test_size=cv_cfg["test_size"],
                                    random_state=cv_cfg["seed"])
    train_sub77, val_sub77 = next(sss77.split(np.zeros(len(train_idx77)), labels77[train_idx77]))
    train_subjects77 = subjects77[train_idx77][train_sub77]
    val_subjects77 = subjects77[train_idx77][val_sub77]
    test_subjects77 = subjects77[test_idx77]

    train_idx_isle, test_idx_isle = fold_idx_isle
    sss_isle = StratifiedShuffleSplit(n_splits=1, test_size=cv_cfg["test_size"],
                                       random_state=cv_cfg["seed"])
    train_sub_isle, val_sub_isle = next(sss_isle.split(np.zeros(len(train_idx_isle)),
                                                        labels_isle[train_idx_isle]))
    train_subjects_isle = subjects_isle[train_idx_isle][train_sub_isle]
    val_subjects_isle = subjects_isle[train_idx_isle][val_sub_isle]
    test_subjects_isle = subjects_isle[test_idx_isle]

    # Per-fold train datasets (fit scalers on training data only - no leakage)
    ds77_train = Dataset77sets(
        clinical_path=os.path.join(data_root, ds_cfg["77sets"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["77sets"]["image_dir"]),
        radiomics_dir=os.path.join(data_root, ds_cfg["77sets"]["radiomics_dir"]),
        subjects=train_subjects77, normalize=True, is_train=True,
        target_shape=tuple(config["image"]["target_shape"]),
    )
    ds_isle_train = DatasetISLE2024(
        clinical_path=os.path.join(data_root, ds_cfg["isle2024"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["isle2024"]["image_dir"]),
        radiomics_brain_dir=os.path.join(data_root, ds_cfg["isle2024"]["radiomics_brain_dir"]),
        radiomics_lesion_dir=lesion_dir,
        subjects=train_subjects_isle, normalize=True, is_train=True,
        target_shape=tuple(config["image"]["target_shape"]),
    )

    # Per-fold val/test datasets (use train-fitted scalers)
    ds77_val = Dataset77sets(
        clinical_path=os.path.join(data_root, ds_cfg["77sets"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["77sets"]["image_dir"]),
        radiomics_dir=os.path.join(data_root, ds_cfg["77sets"]["radiomics_dir"]),
        subjects=val_subjects77, normalize=True, is_train=False,
        clinical_scaler=ds77_train.clinical_scaler,
        radiomics_scaler=ds77_train.radiomics_scaler,
        target_shape=tuple(config["image"]["target_shape"]),
    )
    ds_isle_val = DatasetISLE2024(
        clinical_path=os.path.join(data_root, ds_cfg["isle2024"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["isle2024"]["image_dir"]),
        radiomics_brain_dir=os.path.join(data_root, ds_cfg["isle2024"]["radiomics_brain_dir"]),
        radiomics_lesion_dir=lesion_dir,
        subjects=val_subjects_isle, normalize=True, is_train=False,
        clinical_scaler=ds_isle_train.clinical_scaler,
        radiomics_scaler=ds_isle_train.radiomics_scaler,
        target_shape=tuple(config["image"]["target_shape"]),
    )

    # Joint datasets for training
    joint_train = JointDataset(
        ds77_train,
        ds_isle_train,
        augment_77=sampling_cfg.get("augment_77", 1),
        augment_isle=sampling_cfg.get("augment_isle", 1),
    )
    joint_val = JointDataset(ds77_val, ds_isle_val)

    strategy = sampling_cfg.get("strategy", "auto")
    if strategy == "auto":
        strategy = "interleaved" if config["ablation"].get("use_domain_loss", True) else "random"
    if strategy == "balanced_full":
        train_sampler = BalancedFullSampler(
            joint_train,
            batch_size=batch_size,
            replacement_77=sampling_cfg.get("replacement_77", True),
            replacement_isle=sampling_cfg.get("replacement_isle", False),
            balance_77_classes=sampling_cfg.get("balance_77_classes", True),
            seed=sampling_cfg.get("seed", cv_cfg["seed"]),
            rank=rank,
            world_size=world_size,
        )
        train_loader = DataLoader(joint_train, batch_sampler=train_sampler,
                                  collate_fn=joint_collate_fn, num_workers=2, pin_memory=True)
    elif strategy == "interleaved":
        train_sampler = InterleavedBatchSampler(joint_train, batch_size_each=batch_size // 2,
                                                  rank=rank, world_size=world_size)
        train_loader = DataLoader(joint_train, batch_sampler=train_sampler,
                                  collate_fn=joint_collate_fn, num_workers=2, pin_memory=True)
    elif strategy == "random":
        if world_size > 1:
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(joint_train, shuffle=True,
                                          num_replicas=world_size, rank=rank)
        else:
            from torch.utils.data import RandomSampler
            sampler = RandomSampler(joint_train, replacement=False)
        train_loader = DataLoader(joint_train, batch_size=batch_size, sampler=sampler,
                                  collate_fn=joint_collate_fn, num_workers=2, pin_memory=True)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
    val_loader = DataLoader(joint_val, batch_size=batch_size, shuffle=False,
                            collate_fn=joint_collate_fn, num_workers=2, pin_memory=True)

    # Model — sync all GPUs before creating/switching models
    if world_size > 1:
        dist.barrier()
    model = XFormer(config).to(device)
    if world_size > 1:
        dist.barrier()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
                                                      find_unused_parameters=True)
    trainer = Trainer(model, config, device, rank=rank)
    result = trainer.fit(train_loader, val_loader, fold=fold)
    val_threshold = result.get("best_val_threshold", 0.5)
    best_epoch = result.get("best_epoch", result.get("epoch", 0))
    print(f"  Fold {fold + 1} best val score: {result.get('best_val_score', result['best_val_auc']):.4f} "
          f"(AUC: {result['best_val_auc']:.4f}, "
          f"77: {result.get('best_val_auc_77sets', 0.0):.4f}, "
          f"ISLE: {result.get('best_val_auc_isle2024', 0.0):.4f}, "
          f"gap: {result.get('best_val_auc_gap', 0.0):.4f}, "
          f"epoch: {best_epoch}, thr: {val_threshold:.4f})")

    # Test evaluation -- 77sets (use validation threshold, not test-optimized)
    ds77_test = Dataset77sets(
        clinical_path=os.path.join(data_root, ds_cfg["77sets"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["77sets"]["image_dir"]),
        radiomics_dir=os.path.join(data_root, ds_cfg["77sets"]["radiomics_dir"]),
        subjects=test_subjects77, normalize=True, is_train=False,
        clinical_scaler=ds77_train.clinical_scaler,
        radiomics_scaler=ds77_train.radiomics_scaler,
        target_shape=tuple(config["image"]["target_shape"]),
    )
    test_loader77 = DataLoader(ds77_test, batch_size=batch_size, shuffle=False, num_workers=2)
    use_tta = config["eval"]["tta"]
    y_true77, y_prob77, subjects77_eval = evaluate_test_loader(model, test_loader77, device, tta=use_tta)
    y_pred77 = (y_prob77 >= val_threshold).astype(int)
    metrics77 = compute_metrics(y_true77, y_pred77, y_prob77)
    metrics77["threshold"] = val_threshold

    # Test evaluation -- ISLE (use same validation threshold)
    ds_isle_test = DatasetISLE2024(
        clinical_path=os.path.join(data_root, ds_cfg["isle2024"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["isle2024"]["image_dir"]),
        radiomics_brain_dir=os.path.join(data_root, ds_cfg["isle2024"]["radiomics_brain_dir"]),
        radiomics_lesion_dir=lesion_dir,
        subjects=test_subjects_isle, normalize=True, is_train=False,
        clinical_scaler=ds_isle_train.clinical_scaler,
        radiomics_scaler=ds_isle_train.radiomics_scaler,
        target_shape=tuple(config["image"]["target_shape"]),
    )
    test_loader_isle = DataLoader(ds_isle_test, batch_size=batch_size, shuffle=False,
                                   num_workers=2, collate_fn=joint_collate_fn)
    y_true_isle, y_prob_isle, subjects_isle_eval = evaluate_test_loader(
        model, test_loader_isle, device, tta=use_tta
    )
    y_pred_isle = (y_prob_isle >= val_threshold).astype(int)
    metrics_isle = compute_metrics(y_true_isle, y_pred_isle, y_prob_isle)
    metrics_isle["threshold"] = val_threshold
    if rank == 0:
        print(f"  77sets test AUC: {metrics77['auc']:.4f}")
        print(f"  ISLE test AUC: {metrics_isle['auc']:.4f}")
    if world_size > 1:
        dist.barrier()

    return {
        "fold": fold,
        "77sets": {"metrics": metrics77, "y_true": y_true77, "y_prob": y_prob77},
        "ISLE2024": {"metrics": metrics_isle, "y_true": y_true_isle, "y_prob": y_prob_isle},
        "predictions": (
            prediction_rows(fold, "77sets", subjects77_eval, y_true77, y_prob77, val_threshold)
            + prediction_rows(fold, "ISLE2024", subjects_isle_eval, y_true_isle, y_prob_isle, val_threshold)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="X-Former 5-fold CV")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    config_path = Path(__file__).resolve().parent.parent / args.config
    config = load_config(str(config_path))
    project_root = config_path.parent.parent  # config/ -> project root

    if rank == 0:
        print(f"Device: {device}, World size: {world_size}")

    cv_cfg = config["crossval"]
    ds_cfg = config["datasets"]
    paths = config["paths"]
    # Resolve relative paths against project root
    data_root = str(project_root / paths["data_root"])
    results_root = str(project_root / paths["results_root"])
    config["paths"]["data_root"] = data_root
    config["paths"]["results_root"] = results_root
    lesion_dir = resolve_isle_lesion_dir(config, data_root)

    # Reference datasets (no normalization - used only for subject enumeration)
    ds77_ref = Dataset77sets(
        clinical_path=os.path.join(data_root, ds_cfg["77sets"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["77sets"]["image_dir"]),
        radiomics_dir=os.path.join(data_root, ds_cfg["77sets"]["radiomics_dir"]),
        normalize=False,
        target_shape=tuple(config["image"]["target_shape"]),
    )
    ds_isle_ref = DatasetISLE2024(
        clinical_path=os.path.join(data_root, ds_cfg["isle2024"]["clinical_file"]),
        image_dir=os.path.join(data_root, ds_cfg["isle2024"]["image_dir"]),
        radiomics_brain_dir=os.path.join(data_root, ds_cfg["isle2024"]["radiomics_brain_dir"]),
        radiomics_lesion_dir=lesion_dir,
        normalize=False,
        target_shape=tuple(config["image"]["target_shape"]),
    )

    all_y_true: dict = {"77sets": [], "ISLE2024": []}
    all_y_prob: dict = {"77sets": [], "ISLE2024": []}
    all_results_77: list = []
    all_results_isle: list = []
    all_predictions: list = []
    val_thresholds: list = []

    for fold in range(cv_cfg["n_folds"]):
        if rank == 0:
            print(f"\n{'=' * 60}\nFold {fold + 1}/{cv_cfg['n_folds']}\n{'=' * 60}")
        fold_result = run_fold(config, fold, device, ds77_ref, ds_isle_ref,
                               rank=rank, world_size=world_size)
        all_results_77.append(fold_result["77sets"]["metrics"])
        all_results_isle.append(fold_result["ISLE2024"]["metrics"])
        all_y_true["77sets"].append(fold_result["77sets"]["y_true"])
        all_y_prob["77sets"].append(fold_result["77sets"]["y_prob"])
        all_y_true["ISLE2024"].append(fold_result["ISLE2024"]["y_true"])
        all_y_prob["ISLE2024"].append(fold_result["ISLE2024"]["y_prob"])
        all_predictions.extend(fold_result["predictions"])
        val_thresholds.append(fold_result["77sets"]["metrics"].get("threshold", 0.5))

    if rank != 0:
        cleanup_ddp()
        return

    # Aggregate results with bootstrap CI (rank 0 only)
    print("\n" + "=" * 60)
    print("FINAL RESULTS (95% CI, 1000 Bootstrap)")
    print("=" * 60)

    for ds_name, results_list in [("77sets", all_results_77), ("ISLE2024", all_results_isle)]:
        print(f"\n{ds_name}:")
        for metric in ["auc", "accuracy", "f1_macro", "precision", "recall"]:
            vals = [r[metric] for r in results_list if metric in r]
            if vals:
                print(f"  {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Bootstrap CI on pooled predictions (use mean validation threshold)
    results_dir = Path(results_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = results_dir / "joint"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pooled_threshold = float(np.mean(val_thresholds)) if val_thresholds else 0.5

    for ds_name in all_y_true:
        y_true_pooled = np.concatenate(all_y_true[ds_name])
        y_prob_pooled = np.concatenate(all_y_prob[ds_name])

        y_pred_pooled = (y_prob_pooled >= pooled_threshold).astype(int)

        ci_results = bootstrap_ci(y_true_pooled, y_prob_pooled,
                                  n_iter=config["eval"]["bootstrap_n"],
                                  alpha=config["eval"].get("ci_alpha", 0.05),
                                  fixed_threshold=pooled_threshold)
        metrics = compute_metrics(y_true_pooled, y_pred_pooled, y_prob_pooled)

        print(f"\n{ds_name} (pooled):")
        for name, info in ci_results.items():
            print(f"  {name}: {info['ci_str']}")

        # Save metrics CSV
        df = pd.DataFrame([{"dataset": ds_name, **metrics, "threshold": pooled_threshold}])
        for name, info in ci_results.items():
            df[f"{name}_ci_lower"] = info["ci_lower"]
            df[f"{name}_ci_upper"] = info["ci_upper"]
        df.to_csv(csv_dir / f"metrics_{ds_name}.csv", index=False)

        # ROC curve
        plot_roc_curve(y_true_pooled, y_prob_pooled,
                       title=f"ROC Curve - {ds_name}",
                       save_path=str(csv_dir / "roc_curves" / f"roc_{ds_name}.tif"))

        # Confusion matrix
        plot_confusion_matrix(y_true_pooled, y_pred_pooled,
                              title=f"Confusion Matrix - {ds_name}",
                              save_path=str(csv_dir / "confmat" / f"confmat_{ds_name}.tif"))

    pd.DataFrame(all_predictions).to_csv(csv_dir / "predictions.csv", index=False)

    # Save fold results JSON
    fold_summary = {
        ds_name: [r for r in results_list]
        for ds_name, results_list in [("77sets", all_results_77),
                                       ("ISLE2024", all_results_isle)]
    }
    with open(csv_dir / "fold_results.json", 'w') as f:
        json.dump(fold_summary, f, indent=2, default=str)

    # Cross-modal alignment score (V9)
    # The modality gap = |77sets AUC - ISLE AUC| per fold
    # Small gap → bottleneck treats both modalities equally → cross-modal alignment
    print("\n" + "=" * 60)
    print("CROSS-MODAL ALIGNMENT")
    print("=" * 60)
    gaps = []
    for i in range(len(all_results_77)):
        gap = abs(all_results_77[i]["auc"] - all_results_isle[i]["auc"])
        gaps.append(gap)
        print(f"  Fold {i + 1}: |77sets AUC - ISLE AUC| = {gap:.4f}")
    print(f"  Mean modality gap: {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")
    print(f"  (small gap = bottleneck treats MRI/CT equally = cross-modal aligned)")

    print(f"\nResults saved to {csv_dir}/")


if __name__ == "__main__":
    main()
