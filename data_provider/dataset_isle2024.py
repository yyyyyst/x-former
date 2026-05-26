"""ISLE2024 CT Dataset: stroke outcome prediction with lesion data."""
import os
import warnings
from typing import Optional, Sequence, List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandAffine,
    RandGaussianNoise, RandAdjustContrast, RandCoarseDropout,
    RandShiftIntensity, CropForeground,
)

pd.set_option('future.no_silent_downcasting', True)


class DatasetISLE2024(Dataset):
    def __init__(
        self,
        clinical_path: str,
        image_dir: str,
        radiomics_brain_dir: str,
        radiomics_lesion_dir: Optional[str] = None,
        image_lesion_dir: Optional[str] = None,
        subjects: Optional[Sequence[str]] = None,
        normalize: bool = False,
        clinical_scaler: Optional[StandardScaler] = None,
        radiomics_scaler: Optional[StandardScaler] = None,
        target_shape: Sequence[int] = (64, 160, 160),
        is_train: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.radiomics_brain_dir = radiomics_brain_dir
        self.radiomics_lesion_dir = radiomics_lesion_dir
        self.image_lesion_dir = image_lesion_dir
        self.normalize = normalize
        self.clinical_scaler = clinical_scaler
        self.radiomics_scaler = radiomics_scaler
        self.target_shape = tuple(target_shape)
        self.is_train = is_train
        self.dataset_id = 1  # ISLE2024 = CT

        self.clinical_df = pd.read_csv(clinical_path)
        self.clinical_df.columns = [c.strip() for c in self.clinical_df.columns]
        if "unique_subject" not in self.clinical_df.columns:
            raise KeyError("缺少 'unique_subject' 列")
        if "newmRS" not in self.clinical_df.columns:
            raise KeyError("缺少 'newmRS' 标签列")

        gender_col = "Gender" if "Gender" in self.clinical_df.columns else (
            "Sex" if "Sex" in self.clinical_df.columns else None)
        if gender_col:
            self.clinical_df[gender_col] = (
                self.clinical_df[gender_col].astype(str).str.strip().str.lower()
                .replace({"male": 1, "m": 1, "female": 0, "f": 0})
            )
            self.clinical_df[gender_col] = pd.to_numeric(self.clinical_df[gender_col], errors="coerce")

        available_subjects = []
        for sid in self.clinical_df["unique_subject"].astype(str):
            img_path = os.path.join(self.image_dir, f"{sid}_ses-01_mip_max.nii.gz")
            rad_path = os.path.join(self.radiomics_brain_dir, f"{sid}_t0.csv")
            if os.path.exists(img_path) and os.path.exists(rad_path):
                available_subjects.append(sid)
        self.clinical_df = self.clinical_df[
            self.clinical_df["unique_subject"].astype(str).isin(available_subjects)
        ].reset_index(drop=True)

        if subjects is not None:
            sub_set = set(map(str, subjects))
            self.clinical_df = self.clinical_df[
                self.clinical_df["unique_subject"].astype(str).isin(sub_set)
            ].reset_index(drop=True)

        def _time_str_to_minutes(time_str: str) -> float:
            if pd.isna(time_str) or not isinstance(time_str, str):
                return 0.0
            parts = time_str.strip().split(':')
            if len(parts) != 3:
                return 0.0
            try:
                h, m, s = map(int, parts)
                return h * 60 + m + s / 60.0
            except ValueError:
                return 0.0

        time_cols = [
            c for c in self.clinical_df.columns
            if any(k in c.lower() for k in ["to", "time"])
            and self.clinical_df[c].astype(str).str.contains(':', regex=False).any()
        ]
        for col in time_cols:
            self.clinical_df[col] = self.clinical_df[col].apply(_time_str_to_minutes)

        self.labels = self.clinical_df["newmRS"].values.astype(int)
        drop_cols = [
            "Center", "mRS at admission", "mRS premorbid", "TICI postinterventional",
            "unique_subject", "NIHSS 24h", "mRS 24h", "NIHSS discharge", "mRS discharge",
            "mRS 3 months", "newmRS",
        ]
        self.feature_names = [c for c in self.clinical_df.columns if c not in drop_cols]
        self.clinical_df[self.feature_names] = self.clinical_df[self.feature_names].fillna(0)
        self.clinical_features = self.clinical_df[self.feature_names].astype(float).to_numpy()
        self.n_clinical = len(self.feature_names)

        if normalize:
            if self.is_train:
                if self.clinical_scaler is None:
                    self.clinical_scaler = StandardScaler().fit(self.clinical_features)
                self.clinical_features = self.clinical_scaler.transform(self.clinical_features)
            else:
                if self.clinical_scaler is None:
                    raise RuntimeError("验证/测试集缺失 clinical_scaler")
                self.clinical_features = self.clinical_scaler.transform(self.clinical_features)

        self._subjects_order = self.clinical_df["unique_subject"].astype(str).tolist()
        rad_list: List[np.ndarray] = []
        for sid in self._subjects_order:
            rad_path = os.path.join(self.radiomics_brain_dir, f"{sid}_t0.csv")
            rdf = pd.read_csv(rad_path)
            vals = rdf.drop(columns=rdf.columns[0], errors="ignore").to_numpy(dtype=float)
            vals = np.nan_to_num(vals, copy=False, posinf=0.0, neginf=0.0)
            rad_list.append(vals)
        self.radiomics_features = np.stack(rad_list, axis=0).astype(np.float32)
        self.n_radiomics_features = self.radiomics_features.shape[2]

        if normalize:
            NT = self.radiomics_features.shape[0] * self.radiomics_features.shape[1]
            F_dim = self.radiomics_features.shape[2]
            flat = self.radiomics_features.reshape(NT, F_dim)
            if self.is_train:
                if self.radiomics_scaler is None:
                    self.radiomics_scaler = StandardScaler().fit(flat)
                flat = self.radiomics_scaler.transform(flat)
            else:
                if not isinstance(self.radiomics_scaler, StandardScaler):
                    raise RuntimeError("验证集需要 radiomics_scaler")
                flat = self.radiomics_scaler.transform(flat)
            self.radiomics_features = flat.reshape(self.radiomics_features.shape)

        self._has_lesion = self.radiomics_lesion_dir is not None
        self._lesion_rad_list: Optional[List[np.ndarray]] = None
        self._has_lesion_per_sample: Optional[List[bool]] = None
        if self._has_lesion:
            # First pass: find max feature dimensionality
            max_lesion_dim = 1
            for sid in self._subjects_order:
                lr_path = os.path.join(self.radiomics_lesion_dir, f"{sid}.csv")
                if os.path.exists(lr_path):
                    lrdf = pd.read_csv(lr_path)
                    max_lesion_dim = max(max_lesion_dim, lrdf.shape[1])
            self.n_lesion_features = max_lesion_dim
            # Second pass: load with consistent dim, tracking per-subject presence
            lesion_rads = []
            has_l_per_sample = []
            for sid in self._subjects_order:
                lr_path = os.path.join(self.radiomics_lesion_dir, f"{sid}.csv")
                if os.path.exists(lr_path):
                    lrdf = pd.read_csv(lr_path)
                    lr_vals = lrdf.to_numpy(dtype=float)
                    lr_vals = np.nan_to_num(lr_vals, copy=False, posinf=0.0, neginf=0.0)
                    if lr_vals.shape[1] < max_lesion_dim:
                        padded = np.zeros((lr_vals.shape[0], max_lesion_dim), dtype=float)
                        padded[:, :lr_vals.shape[1]] = lr_vals
                        lr_vals = padded
                    lesion_rads.append(lr_vals)
                    has_l_per_sample.append(True)
                else:
                    lesion_rads.append(np.zeros((1, max_lesion_dim), dtype=float))
                    has_l_per_sample.append(False)
            self._lesion_rad_list = lesion_rads
            self._has_lesion_per_sample = has_l_per_sample

        self.clinical_features = self.clinical_features.astype(np.float32, copy=False)
        self.cropper = CropForeground(select_fn=lambda x: x > 0.05, margin=0)

        if self.is_train:
            self.aug_transform = Compose([
                RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
                RandAffine(prob=0.5, translate_range=(10, 10, 10),
                           scale_range=(0.15, 0.15, 0.15), padding_mode='zeros'),
                RandShiftIntensity(offsets=0.05, prob=0.5),
                RandGaussianNoise(prob=0.3, std=0.05),
                RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
                RandCoarseDropout(holes=2, spatial_size=(16, 16, 16),
                                  fill_value=0, prob=0.3),
            ])
        else:
            self.aug_transform = None

    def __len__(self) -> int:
        return len(self.clinical_df)

    def __getitem__(self, idx: int) -> dict:
        sid = str(self.clinical_df.iloc[idx]["unique_subject"])
        label = int(self.labels[idx])

        radiomics_mat = self.radiomics_features[idx]
        if radiomics_mat.ndim != 2:
            radiomics_mat = radiomics_mat.reshape(1, -1)

        img_path = os.path.join(self.image_dir, f"{sid}_ses-01_mip_max.nii.gz")
        nii_img = nib.load(img_path)
        nii_img = nib.as_closest_canonical(nii_img)
        image_np = np.nan_to_num(nii_img.get_fdata())
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)

        def get_window(img, center, width):
            lower, upper = center - width // 2, center + width // 2
            img_w = torch.clamp(img, min=lower, max=upper)
            return (img_w - lower) / (upper - lower)

        image = torch.stack([
            get_window(image_tensor, 40, 80),
            get_window(image_tensor, 80, 200),
            get_window(image_tensor, 40, 380),
        ], dim=0)
        image = self.cropper(image)
        image = F.interpolate(image.unsqueeze(0), size=self.target_shape,
                              mode='trilinear', align_corners=False).squeeze(0)
        if self.aug_transform is not None:
            image_aug = self.aug_transform(image)
            image = image_aug.float() if isinstance(image_aug, torch.Tensor) else torch.tensor(image_aug, dtype=torch.float32)

        result = {
            "subject": sid,
            "clinical": torch.tensor(self.clinical_features[idx], dtype=torch.float32),
            "radiomics": torch.tensor(radiomics_mat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "image": image,
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
            "has_lesion": torch.tensor(
                (self._has_lesion_per_sample[idx]
                 if self._has_lesion_per_sample is not None
                 else False), dtype=torch.bool),
        }

        if self._has_lesion and self._lesion_rad_list is not None:
            lesion_mat = self._lesion_rad_list[idx]
            result["lesion_radiomics"] = torch.tensor(lesion_mat, dtype=torch.float32)

        return result
