from dataclasses import dataclass


@dataclass
class Config:
    # ── Required: dataset-specific (set via CONFIGS, no defaults) ─────────────
    dataset: str
    patch_size: int        # IP:15  PU:25  WHHH:31  (must be odd: dataset extracts 2*(p//2)+1 pixels)
    pca_components: int    # IP:150 PU:20  WHHH:135
    kernel_size: int       # IP:9   PU:17  WHHH:19
    num_classes: int       # IP:16  PU:9   WHHH:22
    batch_size: int        # IP:200 PU:100 WHHH:20

    # ── Optional: universal defaults (same across all datasets) ───────────────
    data_path: str = 'data'
    hidden_dim: int = 64
    num_heads: int = 1
    dropout: float = 0.1
    depth: int = 2
    lr: float = 5e-4          # Table 2
    epochs: int = 100
    weight_decay: float = 1e-4
    patience: int = 20
    num_train_per_class: int = 10
    num_val_per_class: int = 5
    seed: int = 42


CONFIGS = {
    'IP':   Config(dataset='IP',   patch_size=15, pca_components=150, kernel_size=9,  num_classes=16, batch_size=200),
    'PU':   Config(dataset='PU',   patch_size=25, pca_components=20,  kernel_size=17, num_classes=9,  batch_size=100),
    'WHHH': Config(dataset='WHHH', patch_size=31, pca_components=135, kernel_size=19, num_classes=22, batch_size=20),
    'HC':   Config(dataset='HC',   patch_size=31, pca_components=135, kernel_size=19, num_classes=16, batch_size=20),
    'LK':   Config(dataset='LK',   patch_size=31, pca_components=135, kernel_size=19, num_classes=9,  batch_size=20),
}
