import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def create_split(labels: np.ndarray, n_train_per_class: int, n_val_per_class: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in range(1, labels.max() + 1):
        pos = np.argwhere(labels == cls)
        if len(pos) == 0:
            continue
        pos = pos[rng.permutation(len(pos))]
        n_tr = min(n_train_per_class, len(pos))
        n_val = min(n_val_per_class, max(0, len(pos) - n_tr))
        train_idx.extend(pos[:n_tr].tolist())
        val_idx.extend(pos[n_tr:n_tr + n_val].tolist())
        test_idx.extend(pos[n_tr + n_val:].tolist())
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


class HSIPatchDataset(Dataset):
    def __init__(self, hsi_pca: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                 patch_size: int, transform=None):
        pad = patch_size // 2
        self.labels = labels
        self.indices = indices
        self.pad = pad
        self.transform = transform
        self.hsi = np.pad(hsi_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        r, c = self.indices[idx]
        label = int(self.labels[r, c]) - 1
        rp, cp = r + self.pad, c + self.pad
        patch = self.hsi[rp - self.pad:rp + self.pad + 1, cp - self.pad:cp + self.pad + 1, :]
        patch = torch.from_numpy(patch.copy()).permute(2, 0, 1)
        if self.transform is not None:
            patch = self.transform(patch)
        return patch, torch.tensor(label, dtype=torch.long)


def get_dataloaders(hsi_pca: np.ndarray, labels: np.ndarray, cfg):
    train_idx, val_idx, test_idx = create_split(
        labels,
        n_train_per_class=cfg.num_train_per_class,
        n_val_per_class=cfg.num_val_per_class,
        seed=cfg.seed,
    )
    print(f'Split — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}')

    train_ds = HSIPatchDataset(hsi_pca, labels, train_idx, cfg.patch_size)
    val_ds   = HSIPatchDataset(hsi_pca, labels, val_idx,   cfg.patch_size)
    test_ds  = HSIPatchDataset(hsi_pca, labels, test_idx,  cfg.patch_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
