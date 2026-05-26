"""X-Former: Cross-Modal Information Bottleneck Network for Stroke Outcome Prediction."""
from typing import Dict, Any
import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .clinical_encoder import PrototypeClinicalEncoder
from .radiomics_encoder import RadiomicsEncoder
from .bottleneck_fusion import BottleneckFusion
from losses.domain_loss import GradientReversal


class SimpleMLPClinicalEncoder(nn.Module):
    """Fallback: simple MLP clinical encoder (for ablation). Lazy-initialized on first forward."""

    def __init__(self, proto_dim: int = 128) -> None:
        super().__init__()
        self.proto_dim = proto_dim
        self.mlp: nn.Sequential | None = None

    def _init_mlp(self, n_features: int, device: torch.device) -> None:
        self.mlp = nn.Sequential(
            nn.Linear(n_features, self.proto_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.proto_dim * 2, self.proto_dim),
        ).to(device)

    def forward(self, x: torch.Tensor, dataset_id: torch.Tensor,
                feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mlp is None:
            self._init_mlp(x.shape[-1], x.device)
        return self.mlp(x).unsqueeze(1)  # [B, 1, proto_dim]


class XFormer(nn.Module):

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        img_cfg = model_cfg["image"]
        cli_cfg = model_cfg["clinical"]
        rad_cfg = model_cfg["radiomics"]
        bn_cfg = model_cfg["bottleneck"]
        cls_cfg = model_cfg["classifier"]
        loss_cfg = config["losses"]
        paths = config["paths"]
        abl_cfg = config.get("ablation", {})

        self.use_perceiver = abl_cfg.get("use_perceiver", True)
        self.use_bottleneck = abl_cfg.get("use_bottleneck", True)
        self.use_lesion = abl_cfg.get("use_lesion", True)
        self.use_proto_clinical = abl_cfg.get("use_proto_clinical", True)
        self.joint_training = abl_cfg.get("joint_training", True)
        self.grl_lambda = loss_cfg["domain"]["grl_lambda"]

        self.image_encoder = ImageEncoder(
            n_latents=img_cfg["perceiver_latents"],
            latent_dim=img_cfg["perceiver_dim"],
            weights_path=paths["pretrained_swin"] if img_cfg["swin_pretrained"] else None,
            freeze_layers=img_cfg["swin_freeze_layers"],
            use_perceiver=self.use_perceiver,
        )

        if self.use_proto_clinical:
            self.clinical_encoder = PrototypeClinicalEncoder(
                n_prototypes=cli_cfg["n_prototypes"],
                proto_dim=cli_cfg["proto_dim"],
                use_dataset_embedding=cli_cfg.get("use_dataset_embedding", True),
            )
        else:
            self.clinical_encoder = SimpleMLPClinicalEncoder(
                proto_dim=cli_cfg["proto_dim"],
            )

        self.radiomics_encoder = RadiomicsEncoder(
            proj_dim=rad_cfg["proj_dim"],
            nhead=rad_cfg["temporal_nhead"],
            nlayers=rad_cfg["temporal_nlayers"],
            dropout=rad_cfg["temporal_dropout"],
            use_lesion=self.use_lesion,
        )

        self.bottleneck = BottleneckFusion(
            dim=bn_cfg["dim"],
            n_slots=bn_cfg["n_slots"],
            n_layers=bn_cfg["n_layers"],
            nhead=bn_cfg["nhead"],
            ff_expansion=bn_cfg["ff_expansion"],
            dropout=bn_cfg["dropout"],
        )

        self.proj_img = nn.Linear(img_cfg["perceiver_dim"], bn_cfg["dim"])
        self.proj_cli = nn.Linear(cli_cfg["proto_dim"], bn_cfg["dim"])
        self.proj_rad = nn.Linear(rad_cfg["proj_dim"], bn_cfg["dim"])

        # Classifier head
        cls_layers = []
        prev_dim = bn_cfg["n_slots"] * bn_cfg["dim"]
        for hd in cls_cfg["hidden_dims"]:
            cls_layers.extend([
                nn.Linear(prev_dim, hd),
                nn.GELU(),
                nn.Dropout(cls_cfg["dropout"]),
            ])
            prev_dim = hd
        cls_layers.append(nn.Linear(prev_dim, 2))
        self.classifier = nn.Sequential(*cls_layers)

        self.domain_classifier = nn.Sequential(
            nn.Linear(bn_cfg["dim"] * bn_cfg["n_slots"], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, batch: Dict[str, torch.Tensor],
                return_domain: bool = True) -> Dict[str, torch.Tensor]:
        image = batch["image"]
        clinical = batch["clinical"]
        radiomics = batch["radiomics"]
        dataset_id = batch["dataset_id"]
        clinical_mask = batch.get("clinical_mask", None)
        radiomics_mask = batch.get("radiomics_mask", None)
        lesion_rad = batch.get("lesion_radiomics", None)
        has_lesion = batch.get("has_lesion", None)

        img_tokens = self.image_encoder(image)
        cli_tokens = self.clinical_encoder(clinical, dataset_id, clinical_mask)
        rad_token, lesion_token = self.radiomics_encoder(
            radiomics, dataset_id, radiomics_mask, lesion_rad, has_lesion
        )

        img_tokens = self.proj_img(img_tokens)
        cli_tokens = self.proj_cli(cli_tokens)
        rad_token = self.proj_rad(rad_token)

        all_tokens = [img_tokens, cli_tokens, rad_token]
        if lesion_token is not None:
            lesion_token = self.proj_rad(lesion_token)
            all_tokens.append(lesion_token)
        tokens = torch.cat(all_tokens, dim=1)

        if self.use_bottleneck:
            bottleneck_feat = self.bottleneck(tokens)
        else:
            bottleneck_feat = tokens.mean(dim=1, keepdim=True)
            if bottleneck_feat.size(1) < 8:
                pad = torch.zeros(bottleneck_feat.size(0), 8 - bottleneck_feat.size(1),
                                  bottleneck_feat.size(2), device=bottleneck_feat.device)
                bottleneck_feat = torch.cat([bottleneck_feat, pad], dim=1)

        flat = bottleneck_feat.flatten(1)
        logits = self.classifier(flat)

        if return_domain:
            domain_logits = self.domain_classifier(
                GradientReversal.apply(flat, self.grl_lambda)
            )
        else:
            domain_logits = None

        return {
            "logits": logits,
            "bottleneck_features": flat,
            "domain_logits": domain_logits,
            "img_tokens": img_tokens,
            "cli_tokens": cli_tokens,
        }
