import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import cohen_kappa_score


def compute_metrics(preds: np.ndarray, trues: np.ndarray, num_classes: int) -> dict:
    oa = float((preds == trues).mean())
    try:
        kappa = float(cohen_kappa_score(trues, preds))
    except Exception:
        kappa = 0.0
    per_c = [
        float((preds[trues == c] == trues[trues == c]).mean())
        for c in range(num_classes) if (trues == c).sum() > 0
    ]
    return dict(OA=oa, AA=float(np.mean(per_c)) if per_c else 0.0, Kappa=kappa, per_class=per_c)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device
    preds, trues = [], []
    for x, y in loader:
        preds.extend(model(x.to(dev)).argmax(1).cpu().tolist())
        trues.extend(y.tolist())
    return np.array(preds), np.array(trues)


def get_seed_dir(run_dir: Path, seed: int) -> Path:
    d = run_dir / 'seeds' / f'seed_{seed}'
    d.mkdir(parents=True)
    return d


def get_run_dir(model_name: str, dataset: str, results_root: str = 'results') -> Path:
    base = Path(results_root)
    v = 1
    while (base / f'{model_name}_{dataset}_v{v}').exists():
        v += 1
    run_dir = base / f'{model_name}_{dataset}_v{v}'
    run_dir.mkdir(parents=True)
    return run_dir


def detect_accelerator() -> tuple[str, torch.device]:
    if torch.backends.mps.is_available():
        return 'mps', torch.device('mps')
    if torch.cuda.is_available():
        return 'gpu', torch.device('cuda')
    return 'cpu', torch.device('cpu')
