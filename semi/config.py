import dataclasses
from dataclasses import dataclass
from improved.config import Config, CONFIGS


@dataclass
class SemiConfig(Config):
    teacher_ckpt:     str = ''
    pseudo_per_class: int = 100   # pseudo-labels to add per class per round
    num_val_per_class: int = 0    # no val split — test used as eval monitor
    max_rounds:       int = 10    # max pseudo-labeling rounds


def make_semi_config(dataset: str) -> SemiConfig:
    base = CONFIGS[dataset]
    d = dataclasses.asdict(base)
    d['num_val_per_class'] = 0
    return SemiConfig(**d)


SEMI_CONFIGS = {k: make_semi_config(k) for k in CONFIGS}
