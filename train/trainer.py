"""Single-fold training loop for X-Former joint training."""
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from losses.focal_loss import FocalLoss
from losses.contrastive_loss import SupervisedContrastiveLoss
from losses.consistency_loss import ConsistencyLoss
from utils.metrics import compute_metrics, find_best_threshold


class Trainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 device: torch.device, rank: int = 0) -> None:
        self.model = model
        self._raw_model = model.module if hasattr(model, 'module') else model
        self.rank = rank
        self.config = config
        self.device = device
        loss_cfg = config["losses"]
        train_cfg = config["training"]
        abl_cfg = config.get("ablation", {})

        self.focal_loss = FocalLoss(
            alpha=loss_cfg["focal"]["alpha"],
            gamma=loss_cfg["focal"]["gamma"],
        ).to(device)
        contrast_cfg = loss_cfg["contrastive"]
        self.contrastive_loss = SupervisedContrastiveLoss(
            temperature=contrast_cfg["temperature"],
            same_dataset_weight=contrast_cfg.get("same_dataset_positive_weight", 0.5),
            cross_dataset_weight=contrast_cfg.get("cross_dataset_positive_weight", 1.0),
        ).to(device)
        self.consistency_loss_fn = ConsistencyLoss().to(device)

        # Ablation flags
        self.use_domain_loss = abl_cfg.get("use_domain_loss", True)
        self.use_contrastive = abl_cfg.get("use_contrastive_loss", True)
        self.use_consistency = abl_cfg.get("use_consistency_loss", False)

        self.lambda_domain = loss_cfg["domain"]["weight"]
        self.lambda_contrast = loss_cfg["contrastive"]["weight"]
        self.lambda_consist = loss_cfg["consistency"]["weight"]
        self.loss_warmup_cfg = loss_cfg.get("warmup", {})

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["optimizer"]["lr"],
            weight_decay=train_cfg["optimizer"]["weight_decay"],
            betas=tuple(train_cfg["optimizer"]["betas"]),
        )

        self.scaler = GradScaler() if train_cfg["mixed_precision"] else None
        self.accumulation_steps = train_cfg["accumulation_steps"]
        self.max_epochs = train_cfg["max_epochs"]
        self.early_stop_patience = train_cfg["early_stop_patience"]
        self.min_checkpoint_epoch = train_cfg.get("min_checkpoint_epoch", 1)
        self.grad_clip_norm = train_cfg["grad_clip_norm"]
        checkpoint_cfg = train_cfg.get("checkpoint_selection", {})
        self.checkpoint_metric = checkpoint_cfg.get("metric", "balanced_auc")
        self.checkpoint_gap_penalty = checkpoint_cfg.get("gap_penalty", 0.25)

        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.warmup_epochs = train_cfg["scheduler"]["warmup_epochs"]

    def _build_scheduler(self) -> None:
        sc_cfg = self.config["training"]["scheduler"]
        name = sc_cfg.get("name", "cosine_warm_restart")
        if name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sc_cfg.get("T_max", self.max_epochs),
                eta_min=sc_cfg["eta_min"],
            )
        elif name in {"cosine_warm_restart", "cosine_warm_restarts"}:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=sc_cfg["T_0"], T_mult=sc_cfg["T_mult"],
                eta_min=sc_cfg["eta_min"],
            )
        else:
            self.scheduler = None

    def _scheduled_loss_weight(self, name: str, base_weight: float, epoch: int) -> float:
        warmup_cfg = self.loss_warmup_cfg
        if not warmup_cfg.get("enabled", False):
            return base_weight
        current_epoch = epoch + 1
        start_epoch = warmup_cfg.get(f"{name}_start_epoch", 0)
        ramp_epochs = max(1, warmup_cfg.get(f"{name}_ramp_epochs", 1))
        if current_epoch < start_epoch:
            return 0.0
        elapsed = current_epoch if start_epoch <= 0 else current_epoch - start_epoch + 1
        scale = min(1.0, elapsed / ramp_epochs)
        return base_weight * scale

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        if hasattr(loader, 'batch_sampler') and hasattr(loader.batch_sampler, 'set_epoch'):
            loader.batch_sampler.set_epoch(epoch)
        if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        domain_weight = self._scheduled_loss_weight("domain", self.lambda_domain, epoch)
        contrast_weight = self._scheduled_loss_weight("contrastive", self.lambda_contrast, epoch)
        consist_weight = self._scheduled_loss_weight("consistency", self.lambda_consist, epoch)
        total_loss = 0.0
        total_focal = 0.0
        total_domain = 0.0
        total_contrast = 0.0
        total_consist = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(enabled=self.scaler is not None):
                outputs = self.model(
                    batch,
                    return_domain=self.use_domain_loss and domain_weight > 0.0,
                )
                logits = outputs["logits"]
                labels = batch["label"]

                loss_focal = self.focal_loss(logits, labels)
                loss = loss_focal

                # Domain adversarial loss (GRL applied inside model)
                if self.use_domain_loss and domain_weight > 0.0:
                    loss_domain = domain_weight * F.cross_entropy(
                        outputs["domain_logits"], batch["dataset_id"])
                    loss = loss + loss_domain
                else:
                    loss_domain = torch.tensor(0.0, device=self.device)

                # Supervised contrastive loss
                if self.use_contrastive and contrast_weight > 0.0:
                    loss_contrast = contrast_weight * self.contrastive_loss(
                        outputs["bottleneck_features"], labels, batch.get("dataset_id"))
                    loss = loss + loss_contrast
                else:
                    loss_contrast = torch.tensor(0.0, device=self.device)

                # Consistency loss (with/without lesion for ISLE samples)
                if self.use_consistency and consist_weight > 0.0 and "lesion_radiomics" in batch:
                    has_l = batch.get("has_lesion", None)
                    if has_l is not None and has_l.any():
                        batch_no_l = {k: v for k, v in batch.items()
                                      if k != "lesion_radiomics"}
                        batch_no_l["has_lesion"] = torch.zeros_like(batch["has_lesion"])
                        self.model.eval()
                        with torch.no_grad():
                            out_no_l = self.model(batch_no_l, return_domain=False)
                        self.model.train()
                        loss_consist = consist_weight * self.consistency_loss_fn(
                            logits[has_l], out_no_l["logits"][has_l])
                        loss = loss + loss_consist
                    else:
                        loss_consist = torch.tensor(0.0, device=self.device)
                else:
                    loss_consist = torch.tensor(0.0, device=self.device)

            loss = loss / self.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            total_focal += loss_focal.item()
            total_domain += loss_domain.item()
            total_contrast += loss_contrast.item()
            total_consist += loss_consist.item()

        # Step optimizer for any remaining accumulated gradients at epoch end
        if len(loader) % self.accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()

        n = len(loader)
        return {
            "loss": total_loss / n,
            "focal_loss": total_focal / n,
            "domain_loss": total_domain / n,
            "contrast_loss": total_contrast / n,
            "consist_loss": total_consist / n,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        all_labels = []
        all_probs = []
        all_dataset_ids = []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = self.model(batch, return_domain=False)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            all_labels.append(batch["label"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            if "dataset_id" in batch:
                all_dataset_ids.append(batch["dataset_id"].cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        best_t, _ = find_best_threshold(y_true, y_prob)
        y_pred = (y_prob >= best_t).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["threshold"] = best_t
        if all_dataset_ids:
            dataset_ids = np.concatenate(all_dataset_ids)
            dataset_aucs = []
            for ds_id, ds_name in [(0, "77sets"), (1, "isle2024")]:
                ds_mask = dataset_ids == ds_id
                if not ds_mask.any():
                    continue
                ds_metrics = compute_metrics(
                    y_true[ds_mask],
                    y_pred[ds_mask],
                    y_prob[ds_mask],
                )
                for name, value in ds_metrics.items():
                    metrics[f"{name}_{ds_name}"] = value
                dataset_aucs.append(ds_metrics["auc"])
            if len(dataset_aucs) >= 2:
                auc_gap = abs(metrics["auc_77sets"] - metrics["auc_isle2024"])
                mean_auc = float(np.mean(dataset_aucs))
                metrics["auc_mean_by_dataset"] = mean_auc
                metrics["auc_gap_by_dataset"] = auc_gap
                metrics["balanced_score"] = mean_auc - self.checkpoint_gap_penalty * auc_gap
            elif dataset_aucs:
                metrics["auc_mean_by_dataset"] = float(dataset_aucs[0])
                metrics["auc_gap_by_dataset"] = 0.0
                metrics["balanced_score"] = float(dataset_aucs[0])
        metrics.setdefault("balanced_score", metrics["auc"])
        return metrics

    def _checkpoint_score(self, val_metrics: Dict[str, Any]) -> float:
        if self.checkpoint_metric == "auc":
            return float(val_metrics["auc"])
        if self.checkpoint_metric == "mean_auc":
            return float(val_metrics.get("auc_mean_by_dataset", val_metrics["auc"]))
        if self.checkpoint_metric == "min_auc":
            aucs = [
                val_metrics[key]
                for key in ("auc_77sets", "auc_isle2024")
                if key in val_metrics
            ]
            return float(min(aucs)) if aucs else float(val_metrics["auc"])
        return float(val_metrics.get("balanced_score", val_metrics["auc"]))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            fold: int = 0) -> Dict[str, Any]:
        self._build_scheduler()
        best_val_auc = float("-inf")
        best_checkpoint_score = float("-inf")
        best_observed_score = float("-inf")
        best_val_threshold = 0.5
        best_val_metrics: Dict[str, Any] = {}
        best_state: Optional[Dict] = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.max_epochs):
            current_epoch = epoch + 1
            if epoch < self.warmup_epochs:
                lr_scale = current_epoch / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config["training"]["optimizer"]["lr"] * lr_scale

            train_metrics = self.train_epoch(train_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = self.validate(val_loader)
            val_auc = val_metrics["auc"]
            checkpoint_score = self._checkpoint_score(val_metrics)
            checkpoint_eligible = current_epoch >= self.min_checkpoint_epoch
            improved_observed = checkpoint_score > best_observed_score
            if improved_observed:
                best_observed_score = checkpoint_score
            improved_checkpoint = checkpoint_eligible and checkpoint_score > best_checkpoint_score

            if improved_checkpoint:
                status_suffix = " *"
            elif not checkpoint_eligible:
                status_suffix = f"  (pre-min {current_epoch}/{self.min_checkpoint_epoch})"
            else:
                status_suffix = f"  ({patience_counter + 1}/{self.early_stop_patience})"

            # Print per-epoch summary (rank 0 only)
            if self.rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1:3d}/{self.max_epochs} | "
                      f"LR: {lr:.2e} | "
                      f"Loss: {train_metrics['loss']:.4f} "
                      f"(F:{train_metrics['focal_loss']:.4f} "
                      f"D:{train_metrics['domain_loss']:.4f} "
                      f"C:{train_metrics['contrast_loss']:.4f} "
                      f"K:{train_metrics['consist_loss']:.4f}) | "
                      f"Val AUC: {val_auc:.4f} "
                      f"77:{val_metrics.get('auc_77sets', 0.0):.4f} "
                      f"ISLE:{val_metrics.get('auc_isle2024', 0.0):.4f} "
                      f"Gap:{val_metrics.get('auc_gap_by_dataset', 0.0):.4f} "
                      f"Score:{checkpoint_score:.4f} "
                      f"Acc: {val_metrics['accuracy']:.4f} "
                      f"F1: {val_metrics['f1_macro']:.4f}"
                      + status_suffix)

            if improved_checkpoint:
                best_val_auc = val_auc
                best_checkpoint_score = checkpoint_score
                best_val_threshold = val_metrics.get("threshold", 0.5)
                best_val_metrics = dict(val_metrics)
                best_state = {k: v.clone() for k, v in self._raw_model.state_dict().items()}
                best_epoch = current_epoch
                patience_counter = 0
            elif checkpoint_eligible:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                if self.rank == 0:
                    print(f"  Early stop at epoch {current_epoch}, "
                          f"best score: {best_checkpoint_score:.4f}, "
                          f"best AUC: {best_val_auc:.4f}")
                break

        if best_state is not None:
            self._raw_model.load_state_dict(best_state)
        else:
            best_checkpoint_score = best_observed_score if np.isfinite(best_observed_score) else 0.0
            best_val_auc = 0.0
            if self.rank == 0:
                print("  Warning: no checkpoint met min_checkpoint_epoch; using final model state")
        return {
            "best_val_auc": best_val_auc,
            "best_val_score": best_checkpoint_score,
            "best_val_threshold": best_val_threshold,
            "best_val_auc_77sets": best_val_metrics.get("auc_77sets", 0.0),
            "best_val_auc_isle2024": best_val_metrics.get("auc_isle2024", 0.0),
            "best_val_auc_gap": best_val_metrics.get("auc_gap_by_dataset", 0.0),
            "best_epoch": best_epoch,
            "epoch": epoch + 1,
        }
