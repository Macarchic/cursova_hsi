import numpy as np
import torch
from pathlib import Path

from src.models import get_model, get_config


def load_model(run_dir: str | Path, model_name: str, dataset: str):
    """Load best checkpoint from a results run directory."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    ckpts = sorted(ckpt_dir.glob('best-*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
    best_ckpt = ckpts[-1]

    cfg = get_config(model_name, dataset)
    ckpt_data = torch.load(best_ckpt, map_location='cpu', weights_only=False)

    lit = get_model(model_name, cfg)
    lit.load_state_dict(ckpt_data['state_dict'])
    return lit.model, cfg


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    hsi_pca: np.ndarray,
    labels: np.ndarray,
    patch_size: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Predict class for every labeled pixel and return a (H, W) prediction map."""
    H, W = labels.shape
    pad = patch_size // 2
    phsi = np.pad(hsi_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    coords = [(r, c) for r in range(H) for c in range(W) if labels[r, c] > 0]
    pred_map = np.zeros((H, W), dtype=np.int64)
    dev = next(model.parameters()).device
    model.eval()

    for s in range(0, len(coords), batch_size):
        batch = coords[s:s + batch_size]
        patches = []
        for r, c in batch:
            rp, cp = r + pad, c + pad
            p = phsi[rp - pad:rp + pad + 1, cp - pad:cp + pad + 1, :]
            patches.append(torch.from_numpy(p.copy()).permute(2, 0, 1))
        pr = model(torch.stack(patches).to(dev)).argmax(1).cpu().numpy()
        for (r, c), pred in zip(batch, pr):
            pred_map[r, c] = pred + 1

    return pred_map
