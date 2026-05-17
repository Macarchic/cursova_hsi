import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

import torch

from improved.dataset import HSIPatchDataset, create_split, \
    SpectralJitter, SpatialFlip, PatchRotation


class _SpectralShift:
    def __init__(self, max_shift: int = 5):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shift, dims=0)


class _RandomApply:
    """Apply each transform independently with probability p."""
    def __init__(self, transforms: list, p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            if torch.rand(1).item() < self.p:
                x = t(x)
        return x


def get_semi_dataloaders(
    hsi_pca:      np.ndarray,
    labels:       np.ndarray,
    train_idx:    np.ndarray,
    val_idx:      np.ndarray,
    test_idx:     np.ndarray,
    pseudo_idx:   np.ndarray,
    pseudo_cls:   np.ndarray,
    cfg,
):
    """Build dataloaders for one semi-supervised round.

    train_loader — GT pixels + accumulated pseudo-labeled, with augmentation
    val_loader   — GT val pixels, no augmentation, used for early stopping
    test_loader  — remaining GT pixels, no augmentation, evaluation only
    """
    aug = _RandomApply([
        _SpectralShift(max_shift=5),
        SpectralJitter(scale=0.1),
        SpatialFlip(),
        PatchRotation(),
    ], p=0.5) if getattr(cfg, 'use_augmentation', True) else None

    real_ds = HSIPatchDataset(hsi_pca, labels, train_idx, cfg.patch_size, augment=aug)

    if len(pseudo_idx) > 0:
        pseudo_labels_map = labels.copy()
        for (r, c), cls in zip(pseudo_idx, pseudo_cls):
            pseudo_labels_map[r, c] = int(cls)
        pseudo_ds = HSIPatchDataset(hsi_pca, pseudo_labels_map, pseudo_idx,
                                    cfg.patch_size, augment=aug)
        train_ds = ConcatDataset([real_ds, pseudo_ds])
    else:
        train_ds = real_ds

    val_ds  = HSIPatchDataset(hsi_pca, labels, val_idx,  cfg.patch_size)
    test_ds = HSIPatchDataset(hsi_pca, labels, test_idx, cfg.patch_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def split_for_semi(labels, hsi_pca, cfg):
    return create_split(
        labels,
        hsi_pca=hsi_pca,
        n_train_per_class=cfg.num_train_per_class,
        n_val_per_class=cfg.num_val_per_class,
        seed=cfg.seed,
        use_fps=getattr(cfg, 'use_fps', True),
    )
