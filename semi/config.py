import dataclasses
from dataclasses import dataclass
from improved.config import Config, CONFIGS


@dataclass
class SemiConfig(Config):
    teacher_ckpt:      str   = ''
    pseudo_per_class:  int   = 100   # pseudo-labels to add per class per round
    num_val_per_class: int   = 20    # val set size (overrides default=5)
    finetune_epochs:   int   = 30    # training epochs per round
    finetune_lr:       float = 5e-5  # learning rate for fine-tuning
    max_rounds:        int   = 10    # max pseudo-labeling rounds


def make_semi_config(dataset: str) -> SemiConfig:
    base = CONFIGS[dataset]
    d = dataclasses.asdict(base)
    d['num_val_per_class'] = 20
    return SemiConfig(**d)


SEMI_CONFIGS = {k: make_semi_config(k) for k in CONFIGS}
