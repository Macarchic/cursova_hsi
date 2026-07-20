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


def load_split_dataset(path: str):
    """Load the SCMT authors' fixed split file (e.g. Indian_10_1_split.mat).

    Returns hsi (H,W,C), labels (=TR+TE), TR, TE — the exact train/test masks
    used by the original SCMT paper pipeline.
    """
    d = scipy.io.loadmat(path)
    hsi = d['input'].astype(np.float32)
    TR = d['TR'].astype(np.int64)
    TE = d['TE'].astype(np.int64)
    labels = (TR + TE).astype(np.int64)
    print(f'[SCMT split] {path}')
    print(f'  HSI {hsi.shape}  train px (TR>0): {(TR > 0).sum()}  test px (TE>0): {(TE > 0).sum()}')
    return hsi, labels, TR, TE


def preprocess_scmt(hsi: np.ndarray, n_components: int):
    """SCMT-style preprocessing: per-band max-min normalization -> PCA(whiten=True).

    Mirrors SCMT/src/PCA.py::data_preprocessing (norm_type='max_min') + applyPCA."""
    H, W, C = hsi.shape
    norm = np.zeros_like(hsi, dtype=np.float32)
    for i in range(C):
        band = hsi[:, :, i]
        b_min, b_max = band.min(), band.max()
        denom = (b_max - b_min) if (b_max - b_min) != 0 else 1.0
        norm[:, :, i] = (band - b_min) / denom
    flat = norm.reshape(-1, C)
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    out = pca.fit_transform(flat).reshape(H, W, n_components).astype(np.float32)
    print(f'PCA(whiten) {C} -> {n_components}  explained variance: {pca.explained_variance_ratio_.sum():.3f}')
    return out, pca


def apply_pca(hsi: np.ndarray, n_components: int):
    H, W, C = hsi.shape
    flat = hsi.reshape(-1, C)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-8)
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    out = pca.fit_transform(flat).reshape(H, W, n_components).astype(np.float32)
    print(f'PCA {C} -> {n_components}  explained variance: {pca.explained_variance_ratio_.sum():.3f}')
    return out, pca


def pca_class_outlier_mask(vectors: np.ndarray, outlier_std: float = 2.5) -> np.ndarray:
    """Per-class PCA outlier mask: True where L2 to centroid > mean + outlier_std * std."""
    centroid = vectors.mean(axis=0)
    d = np.linalg.norm(vectors - centroid, axis=1)
    threshold = d.mean() + outlier_std * d.std()
    return d > threshold


def remove_pca_outliers(labels: np.ndarray, hsi_pca: np.ndarray, outlier_std: float = 2.5) -> np.ndarray:
    """Drop PCA outliers from the labeled set by setting their labels to 0 (background)."""
    labels_out = labels.copy()
    n_labeled_before = int((labels > 0).sum())
    n_removed = 0

    for cls in range(1, int(labels.max()) + 1):
        pos = np.argwhere(labels == cls)
        if len(pos) == 0:
            continue
        vectors = hsi_pca[pos[:, 0], pos[:, 1]]
        out_local = pca_class_outlier_mask(vectors, outlier_std)
        if out_local.any():
            out_pos = pos[out_local]
            labels_out[out_pos[:, 0], out_pos[:, 1]] = 0
            n_removed += int(out_local.sum())

    n_labeled_after = int((labels_out > 0).sum())
    pct = 100 * n_removed / n_labeled_before if n_labeled_before else 0.0
    print(
        f'PCA outlier removal (>{outlier_std}σ): {n_removed} px -> background '
        f'({pct:.1f}% of labeled)  |  labeled: {n_labeled_before} -> {n_labeled_after}'
    )
    return labels_out
