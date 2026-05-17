import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from improved.dataset import HSIPatchDataset


def pick_pseudo_labels(
    model: torch.nn.Module,
    hsi_pca: np.ndarray,
    pool_idx: np.ndarray,
    cfg,
    device: torch.device = torch.device('cpu'),
) -> tuple[np.ndarray, np.ndarray]:
    """Select top pseudo_per_class most-confident predictions per class from pool_idx.

    If a class has fewer than pseudo_per_class pixels predicted in it, all of them
    are taken with no compensation from other classes.

    Returns:
        selected_idx : (M, 2)  row/col coordinates in the original image
        pseudo_cls   : (M,)    predicted class labels (1-indexed)
    """
    if len(pool_idx) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    dummy_labels = np.zeros((hsi_pca.shape[0], hsi_pca.shape[1]), dtype=np.int64)
    for r, c in pool_idx:
        dummy_labels[r, c] = 1                    # any non-zero class to avoid dataset skip

    ds = HSIPatchDataset(hsi_pca, dummy_labels, pool_idx, cfg.patch_size, augment=None)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for patches, _ in loader:
            logits = model(patches.to(device))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)    # (N, num_classes)
    pred_cls  = all_probs.argmax(axis=1)              # 0-indexed predictions
    conf      = all_probs.max(axis=1)                 # confidence of top prediction

    selected_local = []
    for c in range(cfg.num_classes):
        mask = pred_cls == c
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        confs_c = conf[indices]
        top_k   = min(cfg.pseudo_per_class, len(indices))
        chosen  = indices[np.argsort(confs_c)[-top_k:]]
        selected_local.extend(chosen.tolist())

    selected_local = np.array(selected_local, dtype=int)
    if len(selected_local) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    return pool_idx[selected_local], (pred_cls[selected_local] + 1)  # back to 1-indexed
