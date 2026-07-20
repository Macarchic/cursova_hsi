from dataclasses import dataclass


@dataclass
class Config:
    """Unified config for improved_paper: base SCMT + improved layers, each toggleable.

    Defaults reproduce the proven *improved* model. Turning every toggle to its
    "base" value (mixing_impl='scmt', center_attn='plain', all use_* = False,
    label_smoothing=0, use_cosine_schedule=False) reduces the network to the
    original SCMT behaviour — enabling point-wise ablation of every addition.
    """

    # ── Required: dataset-specific ────────────────────────────────────────────
    dataset: str
    patch_size: int       # IP:15  PU:25  WHHH:31  (odd: needed for center pixel)
    pca_components: int   # IP:150 PU:20  WHHH:135
    kernel_size: int      # IP:9   PU:17  WHHH:19
    num_classes: int      # IP:16  PU:9   WHHH:22
    batch_size: int       # IP:200 PU:100 WHHH:20

    # ── Optional: universal defaults ──────────────────────────────────────────
    data_path: str = 'data'
    hidden_dim: int = 64
    num_heads: int = 1
    dropout: float = 0.1
    depth: int = 2
    lr: float = 5e-4
    epochs: int = 100
    weight_decay: float = 1e-4
    patience: int = 20
    num_train_per_class: int = 10
    num_val_per_class: int = 5
    seed: int = 42

    # ── Data toggles ──────────────────────────────────────────────────────────
    use_fps: bool = True
    remove_pca_outliers: bool = True
    outlier_std: float = 2.5
    use_augmentation: bool = True

    # ── Architecture toggles (per-layer on/off) ───────────────────────────────
    use_multiscale_stem: bool = False   # improved never wired it in → off by default
    use_se_block: bool = True
    use_pos_encoding: bool = True
    mixing_impl: str = 'improved'       # 'improved' (RWKV blocks) | 'scmt' (original former)
    use_bidirectional_wkv: bool = True  # only effective when mixing_impl == 'improved'
    center_attn: str = 'ring'           # 'ring' (MultiRingCenterAttention) | 'plain' (SCMT center token)

    # ── Training toggles ──────────────────────────────────────────────────────
    label_smoothing: float = 0.05       # 0.0 disables
    use_cosine_schedule: bool = True

    # ── Run-mode bookkeeping (recorded in config.json; no effect on the model) ──
    use_scmt_split: bool = False        # train on SCMT authors' fixed TR/TE split
    final_eval_only: bool = False       # SCMT-style: no per-epoch val, test once at end


CONFIGS = {
    'IP':   Config(dataset='IP',   patch_size=15, pca_components=150, kernel_size=9,  num_classes=16, batch_size=200),
    'PU':   Config(dataset='PU',   patch_size=25, pca_components=20,  kernel_size=17, num_classes=9,  batch_size=100),
    'WHHH': Config(dataset='WHHH', patch_size=31, pca_components=135, kernel_size=19, num_classes=22, batch_size=20),
}
