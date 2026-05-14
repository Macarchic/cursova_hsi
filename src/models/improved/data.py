import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

from src.dataset import create_split, HSIPatchDataset


class SpectralAugment:
    def __call__(self, x):  # x: (C, H, W) float tensor
        x = x + torch.randn_like(x) * 0.01
        mask = torch.bernoulli(torch.ones(x.shape[0]) * 0.9)
        return x * mask.view(-1, 1, 1)


def get_improved_dataloaders(hsi_pca, labels, cfg):
    train_idx, val_idx, test_idx = create_split(
        labels, cfg.num_train_per_class, cfg.num_val_per_class, cfg.seed)
    print(f'Split — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}')

    train_ds = HSIPatchDataset(hsi_pca, labels, train_idx, cfg.patch_size, transform=SpectralAugment())
    val_ds   = HSIPatchDataset(hsi_pca, labels, val_idx,   cfg.patch_size)
    test_ds  = HSIPatchDataset(hsi_pca, labels, test_idx,  cfg.patch_size)

    train_labels = labels[train_idx[:, 0], train_idx[:, 1]] - 1
    counts  = np.bincount(train_labels, minlength=cfg.num_classes).astype(float)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)[train_labels]
    sampler = WeightedRandomSampler(torch.from_numpy(weights).float(), num_samples=len(weights))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,   num_workers=0)
    return train_loader, val_loader, test_loader
