from src.models.baseline import config as _baseline_cfg
from src.models.baseline import model as _baseline_model

MODEL_REGISTRY = {
    'baseline': {
        'configs': _baseline_cfg.CONFIGS,
        'lit_cls': _baseline_model.TCFormerLit,
    },
}


def get_config(model_name: str, dataset: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {model_name!r}. Available: {list(MODEL_REGISTRY)}')
    configs = MODEL_REGISTRY[model_name]['configs']
    if dataset not in configs:
        raise ValueError(f'Unknown dataset: {dataset!r}. Available: {list(configs)}')
    return configs[dataset]


def get_model(model_name: str, cfg):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {model_name!r}. Available: {list(MODEL_REGISTRY)}')
    return MODEL_REGISTRY[model_name]['lit_cls'](cfg)
