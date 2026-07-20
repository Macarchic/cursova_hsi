"""Flag-driven TC-Former: base SCMT stitched with improved layers.

Every addition over the base SCMT model is behind a config flag, so any single
new layer/trick can be switched off independently (full ablation). With the
default config the network is the proven *improved* model; with all toggles set
to their base value it reduces to the original SCMT behaviour:

    mixing_impl='scmt', center_attn='plain', use_multiscale_stem=False,
    use_se_block=False, use_pos_encoding=False, use_bidirectional_wkv=False,
    label_smoothing=0.0, use_cosine_schedule=False,
    use_fps=False, use_augmentation=False, remove_pca_outliers=False
"""

import numpy as np
import torch
import torch.nn as nn
import lightning as L

from improved_paper.config import Config
from improved_paper.utils import compute_metrics
from improved_paper.layers_improved import (
    SEBlock, SinCos2DPositionalEncoding, MultiScaleStem,
    TimeMixFormerBlock, HyperMixFormerBlock, MultiRingCenterAttention, MLPHead,
)
from improved_paper.layers_scmt import former, Init


# ── Lightning progress callback ────────────────────────────────────────────────

class EpochLogger(L.Callback):
    def __init__(self, log_every: int = 1):
        self.log_every = log_every

    def on_train_epoch_start(self, trainer, pl_module):
        ep = trainer.current_epoch + 1
        print(f'  Ep {ep:3d}/{trainer.max_epochs} ({ep / trainer.max_epochs:3.0%}) ...', end='', flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.log_every != 0 and epoch != 1:
            print()
            return
        print(
            f'\r  Ep {epoch:3d}/{trainer.max_epochs} ({epoch / trainer.max_epochs:3.0%})'
            f'  train: loss={pl_module._log_train_loss:.4f}  acc={pl_module._log_train_acc:.1%}'
            f'  │  val: loss={pl_module._log_val_loss:.4f}'
            f'  OA={pl_module._log_val_oa:.1%}  AA={pl_module._log_val_aa:.1%}'
            f'  κ={pl_module._log_val_kappa:.4f}'
        )


def _scmt_params(cfg: Config) -> dict:
    """Build the nested params dict expected by the original SCMT `former`,
    using the paper's default net hyper-params (scmtformer.json)."""
    net = {
        'n_attn': 64,
        'n_head': 64,
        'ctx_len': cfg.patch_size * cfg.patch_size,
        'n_embd': 128,
        'n_ffn': 64,
        'tiny_attn': 512,
        'tiny_head': 64,
    }
    data = {'num_classes': cfg.num_classes, 'spectral_size': cfg.pca_components}
    return {'net': net, 'data': data}


class TCFormer(nn.Module):
    """PCA → stem → [SE] → [PosEnc] → mixing (improved|scmt) → center (ring|plain) → MLPHead."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        # ── Stem ──────────────────────────────────────────────────────────────
        if getattr(cfg, 'use_multiscale_stem', False):
            self.stem = MultiScaleStem(cfg.pca_components, D, cfg.kernel_size)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(cfg.pca_components, D, kernel_size=cfg.kernel_size,
                          padding=cfg.kernel_size // 2, bias=False),
                nn.BatchNorm2d(D),
                nn.ReLU(inplace=True),
            )

        # ── Optional pre-sequence enhancements ───────────────────────────────
        self.se = SEBlock(D) if getattr(cfg, 'use_se_block', True) else None
        self.pos_enc = (
            SinCos2DPositionalEncoding(D, cfg.patch_size)
            if getattr(cfg, 'use_pos_encoding', True) else None
        )

        # ── Mixing stack ──────────────────────────────────────────────────────
        self.mixing_impl = getattr(cfg, 'mixing_impl', 'improved')
        if self.mixing_impl == 'improved':
            bi = getattr(cfg, 'use_bidirectional_wkv', False)
            self.time_blocks  = nn.ModuleList([TimeMixFormerBlock(D, cfg.num_heads, bidirectional=bi)  for _ in range(cfg.depth)])
            self.hyper_blocks = nn.ModuleList([HyperMixFormerBlock(D, cfg.num_heads, bidirectional=bi) for _ in range(cfg.depth)])
        elif self.mixing_impl == 'scmt':
            self.scmt_former = former(D, cfg.depth, _scmt_params(cfg),
                                      cfg.num_heads, D, D * 4, cfg.dropout)
            Init(self.scmt_former)
        else:
            raise ValueError(f"mixing_impl must be 'improved' or 'scmt', got {self.mixing_impl!r}")

        # ── Center aggregation ────────────────────────────────────────────────
        self.center_mode = getattr(cfg, 'center_attn', 'ring')
        if self.center_mode == 'ring':
            self.center_attn = MultiRingCenterAttention(D, cfg.patch_size, cfg.num_heads)
        elif self.center_mode != 'plain':
            raise ValueError(f"center_attn must be 'ring' or 'plain', got {self.center_mode!r}")

        self.head = MLPHead(D, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        x = self.stem(x)
        if self.se is not None:
            x = self.se(x)
        x = x.flatten(2).transpose(1, 2)          # (B, T, D)
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        if self.mixing_impl == 'improved':
            for block in self.time_blocks:
                x = block(x)
            for block in self.hyper_blocks:
                x = block(x)
        else:
            x, _ = self.scmt_former(x)

        if self.center_mode == 'ring':
            feat = self.center_attn(x)
        else:
            feat = x[:, x.shape[1] // 2, :]        # SCMT-style center token

        return self.head(feat)


# ── Lightning wrapper ─────────────────────────────────────────────────────────

class TCFormerLit(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg       = cfg
        self.model     = TCFormer(cfg)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=getattr(cfg, 'label_smoothing', 0.0))
        self._val_preds:    list = []
        self._val_trues:    list = []
        self._val_losses:   list = []
        self._train_losses: list = []
        self._train_accs:   list = []
        self._log_train_loss: float = float('nan')
        self._log_train_acc:  float = float('nan')
        self._log_val_loss:   float = float('nan')
        self._log_val_oa:     float = float('nan')
        self._log_val_aa:     float = float('nan')
        self._log_val_kappa:  float = float('nan')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc  = (logits.argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)
        self._train_losses.append(loss.item())
        self._train_accs.append(acc.item())
        return loss

    def on_train_epoch_end(self):
        if self._train_losses:
            self._log_train_loss = float(np.mean(self._train_losses))
            self._log_train_acc  = float(np.mean(self._train_accs))
            self._train_losses.clear()
            self._train_accs.clear()

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self._val_preds.append(logits.argmax(1).cpu())
        self._val_trues.append(y.cpu())
        self._val_losses.append(self.criterion(logits, y).item())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy()
        trues = torch.cat(self._val_trues).numpy()
        m = compute_metrics(preds, trues, self.cfg.num_classes)
        self.log('val_OA',    m['OA'],    prog_bar=True)
        self.log('val_AA',    m['AA'],    prog_bar=False)
        self.log('val_kappa', m['Kappa'], prog_bar=False)
        self.log('val_loss',  float(np.mean(self._val_losses)), prog_bar=False)
        self._log_val_oa    = m['OA']
        self._log_val_aa    = m['AA']
        self._log_val_kappa = m['Kappa']
        self._log_val_loss  = float(np.mean(self._val_losses))
        self._val_preds.clear()
        self._val_trues.clear()
        self._val_losses.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if getattr(self.cfg, 'use_cosine_schedule', True):
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cfg.epochs, eta_min=self.cfg.lr * 0.1
            )
            return [opt], [{'scheduler': sched, 'interval': 'epoch'}]
        return opt
