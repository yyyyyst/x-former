"""77sets MRI Dataset: stroke outcome prediction."""
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


class Dataset77sets(Dataset):
    def __init__(
        self,
        clinical_path: str,
        image_dir: str,
        radiomics_dir: str,
        subjects: Optional[Sequence[str]] = None,
        normalize: bool = False,
        clinical_scaler: Optional[StandardScaler] = None,
        radiomics_scaler: Optional[StandardScaler] = None,
        target_shape: Sequence[int] = (64, 160, 160),
        is_train: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.radiomics_dir = radiomics_dir
        self.normalize = normalize
        self.clinical_scaler = clinical_scaler
        self.radiomics_scaler = radiomics_scaler
        self.target_shape = tuple(target_shape)
        self.is_train = is_train
        self.dataset_id = 0  # 77sets = MRI

        self.clinical_df = pd.read_csv(clinical_path)
        self.clinical_df.columns = [c.strip() for c in self.clinical_df.columns]
        if "Gender" in self.clinical_df.columns:
            self.clinical_df["Gender"] = self.clinical_df["Gender"].replace({"Male": 1, "Female": 0})

        available_subjects = []
        for sid in self.clinical_df["unique_subject"].astype(str):
            img_path = os.path.join(image_dir, f"{sid}.nii.gz")
            rad_path = os.path.join(radiomics_dir, f"{sid}.csv")
            if os.path.exists(img_path) and os.path.exists(rad_path):
                available_subjects.append(sid)
        self.clinical_df = self.clinical_df[
            self.clinical_df["unique_subject"].astype(str).isin(available_subjects)
        ].reset_index(drop=True)

        if subjects is not None:
            subjects = set(map(str, subjects))
            self.clinical_df = self.clinical_df[
                self.clinical_df["unique_subject"].astype(str).isin(subjects)
            ].reset_index(drop=True)

        if "newmRS" not in self.clinical_df.columns:
            raise KeyError("缺少 'newmRS' 标签列")
        self.labels = self.clinical_df["newmRS"].values.astype(int)

        drop_cols = [
            "Serial", "Name", "Radiology ID", "subject",
            "new_subject", "90 day mRS", "newmRS", "unique_subject"
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
            rad_path = os.path.join(self.radiomics_dir, f"{sid}.csv")
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    flat = self.radiomics_scaler.transform(flat)
            else:
                if self.radiomics_scaler is None:
                    raise RuntimeError("验证/测试集缺失 radiomics_scaler")
                flat = self.radiomics_scaler.transform(flat)
            self.radiomics_features = flat.reshape(self.radiomics_features.shape)

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
                RandShiftIntensity(offsets=0.1, prob=0.5),
                RandGaussianNoise(prob=0.3, std=0.05),
                RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)),
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

        img_path = os.path.join(self.image_dir, f"{sid}.nii.gz")
        nii_img = nib.load(img_path)
        nii_img = nib.as_closest_canonical(nii_img)
        image_np = np.nan_to_num(nii_img.get_fdata())
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)

        # CT-style window: consistent with Paper 1
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

        return {
            "subject": sid,
            "clinical": torch.tensor(self.clinical_features[idx], dtype=torch.float32),
            "radiomics": torch.tensor(radiomics_mat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "image": image,
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
            "has_lesion": torch.tensor(0, dtype=torch.bool),
        }
