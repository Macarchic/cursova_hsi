import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _fps_indices(vectors: np.ndarray, k: int, outlier_std: float = 2.5) -> np.ndarray:
    """Farthest Point Sampling with outlier pre-filtering.

    1. Remove outliers: discard samples whose L2 distance to the class centroid
       exceeds mean_dist + outlier_std * std_dist (keeps the bulk of the distribution).
    2. Start FPS from the inlier closest to the centroid (avoids anchoring to an extreme).
    3. Greedily pick the next sample farthest from the already-selected set.

    Returns indices into the *original* vectors array (before outlier removal).
    """
    n = len(vectors)
    if n <= k:
        return np.arange(n)

    # --- outlier filtering ---
    centroid = vectors.mean(axis=0)
    d = np.linalg.norm(vectors - centroid, axis=1)          # (n,) distances to centroid
    threshold = d.mean() + outlier_std * d.std()
    inlier_mask = d <= threshold
    inlier_idx = np.where(inlier_mask)[0]                   # original indices of inliers
    if len(inlier_idx) < k:                                  # safety: too aggressive → use all
        inlier_idx = np.arange(n)
    vecs = vectors[inlier_idx]                               # (m, C)

    # --- FPS on inliers, seeded from centroid-nearest sample ---
    d_to_centroid = np.linalg.norm(vecs - centroid, axis=1)
    selected = [int(np.argmin(d_to_centroid))]
    min_dists = np.full(len(vecs), np.inf)

    for _ in range(k - 1):
        last_vec = vecs[selected[-1]]
        d_new = np.linalg.norm(vecs - last_vec, axis=1)
        min_dists = np.minimum(min_dists, d_new)
        selected.append(int(np.argmax(min_dists)))

    return inlier_idx[selected]


def create_split(
    labels: np.ndarray,
    hsi_pca: np.ndarray,
    n_train_per_class: int,
    n_val_per_class: int = 5,
    seed: int = 42,
    use_fps: bool = True,
    outlier_std: float = 2.5,
):
    """Split labeled pixels into train / val / test sets.

    When use_fps=True, training samples are chosen via Farthest Point Sampling
    on PCA spectral vectors (with outlier pre-filtering), maximising in-class
    diversity while avoiding extreme outliers.
    Val and test are drawn randomly from the remaining pixels.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in range(1, labels.max() + 1):
        pos = np.argwhere(labels == cls)            # (N, 2) — (row, col) coordinates
        if len(pos) == 0:
            continue
        n_tr = min(n_train_per_class, len(pos))

        if use_fps and n_tr < len(pos):
            vectors = hsi_pca[pos[:, 0], pos[:, 1]]            # (N, C_pca)
            sel_local = _fps_indices(vectors, n_tr, outlier_std)
            train_pos = pos[sel_local]
            remaining_mask = np.ones(len(pos), dtype=bool)
            remaining_mask[sel_local] = False
            remaining_pos = pos[remaining_mask]
        else:
            perm = rng.permutation(len(pos))
            train_pos = pos[perm[:n_tr]]
            remaining_pos = pos[perm[n_tr:]]

        remaining_pos = remaining_pos[rng.permutation(len(remaining_pos))]
        n_val = min(n_val_per_class, len(remaining_pos))

        train_idx.extend(train_pos.tolist())
        val_idx.extend(remaining_pos[:n_val].tolist())
        test_idx.extend(remaining_pos[n_val:].tolist())

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


class HSIPatchDataset(Dataset):
    def __init__(self, hsi_pca: np.ndarray, labels: np.ndarray, indices: np.ndarray, patch_size: int):
        pad = patch_size // 2
        self.labels = labels
        self.indices = indices
        self.pad = pad
        self.hsi = np.pad(hsi_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        r, c = self.indices[idx]
        label = int(self.labels[r, c]) - 1
        rp, cp = r + self.pad, c + self.pad
        patch = self.hsi[rp - self.pad:rp + self.pad + 1, cp - self.pad:cp + self.pad + 1, :]
        patch = torch.from_numpy(patch.copy()).permute(2, 0, 1)
        return patch, torch.tensor(label, dtype=torch.long)


def get_dataloaders(hsi_pca: np.ndarray, labels: np.ndarray, cfg):
    train_idx, val_idx, test_idx = create_split(
        labels,
        hsi_pca=hsi_pca,
        n_train_per_class=cfg.num_train_per_class,
        n_val_per_class=cfg.num_val_per_class,
        seed=cfg.seed,
        use_fps=getattr(cfg, 'use_fps', True),
    )
    print(f'Split — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}')

    train_ds = HSIPatchDataset(hsi_pca, labels, train_idx, cfg.patch_size)
    val_ds   = HSIPatchDataset(hsi_pca, labels, val_idx,   cfg.patch_size)
    test_ds  = HSIPatchDataset(hsi_pca, labels, test_idx,  cfg.patch_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
