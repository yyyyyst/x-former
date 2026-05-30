"""Joint dataset and batch sampler for interleaved 77sets + ISLE2024 training."""
from typing import Iterator, List
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class JointDataset(Dataset):
    """Concatenated dataset with optional sample augmentation factor."""

    def __init__(self, dataset_77: Dataset, dataset_isle: Dataset,
                 augment_77: int = 1, augment_isle: int = 1) -> None:
        self.ds77 = dataset_77
        self.dsisle = dataset_isle
        self.n77 = len(dataset_77)
        self.nisle = len(dataset_isle)
        self.aug77 = augment_77
        self.aug_isle = augment_isle

    def __len__(self) -> int:
        return self.n77 * self.aug77 + self.nisle * self.aug_isle

    def __getitem__(self, idx: int) -> dict:
        n77_total = self.n77 * self.aug77
        if idx < n77_total:
            real_idx = idx % self.n77  # cycle through 77sets with different augmentation
            return self.ds77[real_idx]
        real_idx = (idx - n77_total) % self.nisle
        return self.dsisle[real_idx]


class InterleavedBatchSampler(Sampler):
    """Yields batches with equal numbers of 77sets and ISLE samples."""

    def __init__(
        self,
        joint_dataset: JointDataset,
        batch_size_each: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.n77 = joint_dataset.n77
        self.nisle = joint_dataset.nisle
        self.bs_each = batch_size_each
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.offset_isle = joint_dataset.n77 * joint_dataset.aug77
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.RandomState(self.seed + epoch)

    def __iter__(self) -> Iterator[List[int]]:
        idx77 = np.arange(self.n77)
        idxisle = np.arange(self.offset_isle, self.offset_isle + self.nisle)
        if self.shuffle:
            self.rng.shuffle(idx77)
            self.rng.shuffle(idxisle)

        n_batches = min(len(idx77) // self.bs_each, len(idxisle) // self.bs_each)
        batches: List[List[int]] = []
        for b in range(n_batches):
            batch: List[int] = []
            for i in range(self.bs_each):
                batch.append(int(idx77[b * self.bs_each + i]))
                batch.append(int(idxisle[b * self.bs_each + i]))
            batches.append(batch)

        while batches and len(batches) % self.world_size != 0:
            batches.append(list(batches[self.rng.randint(0, len(batches))]))

        for batch in batches[self.rank::self.world_size]:
            yield batch

    def __len__(self) -> int:
        total = min(self.n77 // self.bs_each, self.nisle // self.bs_each)
        if self.world_size <= 1:
            return total
        return int(np.ceil(total / self.world_size))


class BalancedFullSampler(Sampler):
    """Use all ISLE samples once per epoch and sample 77sets to match.

    This keeps ISLE data utilization high without duplicating the underlying
    JointDataset length. In DDP, the batch list is padded so every rank gets the
    same number of steps.
    """

    def __init__(
        self,
        joint_dataset: JointDataset,
        batch_size: int = 4,
        replacement_77: bool = True,
        replacement_isle: bool = False,
        balance_77_classes: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size < 2:
            raise ValueError("BalancedFullSampler requires batch_size >= 2")
        self.joint_dataset = joint_dataset
        self.n77 = joint_dataset.n77
        self.nisle = joint_dataset.nisle
        self.bs77 = max(1, batch_size // 2)
        self.bsisle = max(1, batch_size - self.bs77)
        self.replacement_77 = replacement_77
        self.replacement_isle = replacement_isle
        self.balance_77_classes = balance_77_classes
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.offset_isle = joint_dataset.n77 * joint_dataset.aug77
        self.rng = np.random.RandomState(seed)

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.RandomState(self.seed + epoch)

    def _sample_77(self, n: int) -> np.ndarray:
        idx77 = np.arange(self.n77)
        replace = self.replacement_77 or n > self.n77
        probs = None
        labels = getattr(self.joint_dataset.ds77, "labels", None)
        if self.balance_77_classes and labels is not None:
            labels_arr = np.asarray(labels)
            classes, counts = np.unique(labels_arr, return_counts=True)
            class_weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}
            weights = np.array([class_weights[label] for label in labels_arr], dtype=np.float64)
            probs = weights / weights.sum()
        if replace:
            return self.rng.choice(idx77, size=n, replace=True, p=probs)
        if self.shuffle:
            self.rng.shuffle(idx77)
        return idx77[:n]

    def _sample_isle(self, n_total: int) -> np.ndarray:
        idxisle = np.arange(self.offset_isle, self.offset_isle + self.nisle)
        if self.shuffle:
            self.rng.shuffle(idxisle)
        if n_total <= self.nisle:
            return idxisle[:n_total]
        n_extra = n_total - self.nisle
        extra = self.rng.choice(idxisle, size=n_extra, replace=True)
        return np.concatenate([idxisle, extra])

    def _num_batches(self) -> int:
        return int(np.ceil(self.nisle / self.bsisle))

    def _padded_num_batches(self) -> int:
        n_batches = self._num_batches()
        if self.world_size <= 1:
            return n_batches
        return int(np.ceil(n_batches / self.world_size) * self.world_size)

    def __iter__(self) -> Iterator[List[int]]:
        n_batches = self._num_batches()
        n_batches_padded = self._padded_num_batches()
        n_isle_needed = n_batches * self.bsisle
        n77_needed = n_batches * self.bs77

        idx77 = self._sample_77(n77_needed)
        idxisle = self._sample_isle(n_isle_needed)

        batches: List[List[int]] = []
        for b in range(n_batches):
            start77, end77 = b * self.bs77, (b + 1) * self.bs77
            start_isle, end_isle = b * self.bsisle, (b + 1) * self.bsisle
            batch = [int(i) for i in idx77[start77:end77]]
            batch.extend(int(i) for i in idxisle[start_isle:end_isle])
            if self.shuffle:
                self.rng.shuffle(batch)
            batches.append(batch)

        while len(batches) < n_batches_padded:
            batches.append(list(batches[self.rng.randint(0, len(batches))]))

        for batch in batches[self.rank::self.world_size]:
            yield batch

    def __len__(self) -> int:
        return self._padded_num_batches() // self.world_size


def joint_collate_fn(batch: List[dict]) -> dict:
    """Collate heterogeneous batch from interleaved 77sets + ISLE samples.

    Handles:
    - Variable clinical feature counts (pad to max F)
    - Variable radiomics feature counts (pad to max F_rad)
    - Optional lesion_radiomics (missing for 77sets -> zeros)
    - Variable-length lesion regions (pad_sequence)
    """
    # Collect union of all keys
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    collated: dict = {}

    # --- subject (strings, keep as list) ---
    if "subject" in all_keys:
        collated["subject"] = [item.get("subject", "") for item in batch]

    # --- clinical: pad to max feature count ---
    if "clinical" in all_keys:
        clinical_list = [item["clinical"] for item in batch]
        max_f = max(c.shape[-1] for c in clinical_list)
        clinical_padded = []
        for c in clinical_list:
            if c.shape[-1] < max_f:
                pad = torch.zeros(max_f - c.shape[-1], dtype=c.dtype)
                clinical_padded.append(torch.cat([c, pad]))
            else:
                clinical_padded.append(c)
        collated["clinical"] = torch.stack(clinical_padded)
        # Feature mask: 1 = real feature, 0 = padded
        feat_mask = torch.zeros(len(batch), max_f, dtype=torch.bool)
        for i, c in enumerate(clinical_list):
            feat_mask[i, :c.shape[-1]] = True
        collated["clinical_mask"] = feat_mask

    # --- radiomics: pad both time and feature dims to max ---
    if "radiomics" in all_keys:
        rad_list = [item["radiomics"] for item in batch]
        max_t_rad = max(r.shape[0] for r in rad_list)
        max_f_rad = max(r.shape[-1] for r in rad_list)
        rad_padded = []
        for r in rad_list:
            # Pad feature dim
            if r.shape[-1] < max_f_rad:
                r = torch.cat([r, torch.zeros(r.shape[0], max_f_rad - r.shape[-1],
                                             dtype=r.dtype)], dim=-1)
            # Pad time dim
            if r.shape[0] < max_t_rad:
                r = torch.cat([r, torch.zeros(max_t_rad - r.shape[0], r.shape[-1],
                                             dtype=r.dtype)], dim=0)
            rad_padded.append(r)
        collated["radiomics"] = torch.stack(rad_padded)
        # Feature mask for radiomics (time × feature)
        rad_mask = torch.zeros(len(batch), max_t_rad, max_f_rad, dtype=torch.bool)
        for i, r in enumerate(rad_list):
            rad_mask[i, :r.shape[0], :r.shape[-1]] = True
        collated["radiomics_mask"] = rad_mask

    # --- lesion_radiomics: pad_sequence, or zeros for 77sets ---
    if "lesion_radiomics" in all_keys:
        # Determine actual feature dimension from any ISLE sample with real data
        max_ldim = 1
        for item in batch:
            lr = item.get("lesion_radiomics", None)
            if lr is not None and lr.shape[-1] > 1:
                max_ldim = max(max_ldim, lr.shape[-1])
        lesion_list = []
        has_lesion = torch.zeros(len(batch), dtype=torch.bool)
        for i, item in enumerate(batch):
            lr = item.get("lesion_radiomics", None)
            if lr is not None and lr.shape[-1] == max_ldim:
                lesion_list.append(lr.float())
                has_lesion[i] = True
            else:
                h = item.get("has_lesion", torch.tensor(0, dtype=torch.bool))
                if h.item():
                    has_lesion[i] = True
                lesion_list.append(torch.zeros(1, max_ldim).float())
        collated["lesion_radiomics"] = torch.nn.utils.rnn.pad_sequence(
            lesion_list, batch_first=True
        )
        collated["has_lesion"] = has_lesion

    # --- remaining tensor keys: stack normally ---
    tensor_keys = ["label", "image", "dataset_id"]
    for key in tensor_keys:
        if key in all_keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values

    # --- has_lesion (if not already set from lesion_radiomics block) ---
    if "has_lesion" in all_keys and "has_lesion" not in collated:
        values = [item["has_lesion"] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated["has_lesion"] = torch.stack(values)
        else:
            collated["has_lesion"] = torch.tensor(values, dtype=torch.bool)

    return collated
