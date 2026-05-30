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
        self.contrastive_loss = SupervisedContrastiveLoss(
            temperature=loss_cfg["contrastive"]["temperature"],
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
        self.grad_clip_norm = train_cfg["grad_clip_norm"]

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
                        outputs["bottleneck_features"], labels)
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

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = self.model(batch, return_domain=False)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            all_labels.append(batch["label"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        best_t, _ = find_best_threshold(y_true, y_prob)
        y_pred = (y_prob >= best_t).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["threshold"] = best_t
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            fold: int = 0) -> Dict[str, Any]:
        self._build_scheduler()
        best_val_auc = 0.0
        best_val_threshold = 0.5
        best_state: Optional[Dict] = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            if epoch < self.warmup_epochs:
                lr_scale = (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config["training"]["optimizer"]["lr"] * lr_scale

            train_metrics = self.train_epoch(train_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = self.validate(val_loader)
            val_auc = val_metrics["auc"]

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
                      f"Acc: {val_metrics['accuracy']:.4f} "
                      f"F1: {val_metrics['f1_macro']:.4f}"
                      + (" *" if val_auc > best_val_auc else f"  ({patience_counter + 1}/{self.early_stop_patience})"))

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_threshold = val_metrics.get("threshold", 0.5)
                best_state = {k: v.clone() for k, v in self._raw_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                if self.rank == 0:
                    print(f"  Early stop at epoch {epoch + 1}, best AUC: {best_val_auc:.4f}")
                break

        if best_state is not None:
            self._raw_model.load_state_dict(best_state)
        return {"best_val_auc": best_val_auc, "best_val_threshold": best_val_threshold,
                "epoch": epoch + 1}
