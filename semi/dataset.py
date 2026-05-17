import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from improved.dataset import HSIPatchDataset, create_split, \
    SpectralJitter, SpatialFlip, PatchRotation


class _SpectralShift:
    def __init__(self, max_shift: int = 5):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shift, dims=0)


class _RandomApply:
    def __init__(self, transforms: list, p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            if torch.rand(1).item() < self.p:
                x = t(x)
        return x


def get_semi_dataloaders(
    hsi_pca:    np.ndarray,
    labels:     np.ndarray,
    train_idx:  np.ndarray,
    test_idx:   np.ndarray,
    pseudo_idx: np.ndarray,
    pseudo_cls: np.ndarray,
    cfg,
):
    """Build dataloaders for one semi-supervised round.

    train_loader — GT train + accumulated pseudo-labeled pixels, with augmentation
    test_loader  — full fixed test set, GT labels, no augmentation
                   used both as eval monitor during fine-tuning and final evaluation
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

    test_ds = HSIPatchDataset(hsi_pca, labels, test_idx, cfg.patch_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def split_for_semi(labels, hsi_pca, cfg):
    """Returns (train_idx, test_idx). Val is empty (num_val_per_class=0)."""
    train_idx, _, test_idx = create_split(
        labels,
        hsi_pca=hsi_pca,
        n_train_per_class=cfg.num_train_per_class,
        n_val_per_class=0,
        seed=cfg.seed,
        use_fps=getattr(cfg, 'use_fps', True),
    )
    return train_idx, test_idx
