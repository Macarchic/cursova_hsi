import os
import numpy as np
import scipy.io
from sklearn.decomposition import PCA


def load_dataset(name: str, data_path: str = 'data'):
    if name == 'IP':
        d = scipy.io.loadmat(os.path.join(data_path, 'IP', 'Indian_pines_corrected.mat'))
        hsi = d['indian_pines_corrected']
        labels = np.load(os.path.join(data_path, 'IP', 'IPgt.npy'))
    elif name == 'PU':
        d = scipy.io.loadmat(os.path.join(data_path, 'Pavia', 'PaviaU.mat'))
        g = scipy.io.loadmat(os.path.join(data_path, 'Pavia', 'PaviaU_gt.mat'))
        hsi, labels = d['paviaU'], g['paviaU_gt']
    elif name == 'WHHH':
        d = scipy.io.loadmat(os.path.join(data_path, 'WHU-Hi-HongHu', 'WHU_Hi_HongHu.mat'))
        g = scipy.io.loadmat(os.path.join(data_path, 'WHU-Hi-HongHu', 'WHU_Hi_HongHu_gt.mat'))
        hsi, labels = d['WHU_Hi_HongHu'], g['WHU_Hi_HongHu_gt']
    else:
        raise ValueError(f'Unknown dataset: {name!r}. Available: IP, PU, WHHH')
    hsi = hsi.astype(np.float32)
    labels = labels.astype(np.int64)
    print(f'[{name}] HSI {hsi.shape}  Labels {labels.shape}  Labeled px: {(labels > 0).sum()}')
    return hsi, labels


def apply_pca(hsi: np.ndarray, n_components: int):
    H, W, C = hsi.shape
    flat = hsi.reshape(-1, C)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-8)
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    out = pca.fit_transform(flat).reshape(H, W, n_components).astype(np.float32)
    print(f'PCA {C} -> {n_components}  explained variance: {pca.explained_variance_ratio_.sum():.3f}')
    return out, pca


