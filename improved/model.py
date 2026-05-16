import numpy as np
import torch
import torch.nn as nn
import lightning as L

from improved.config import Config
from improved.utils import compute_metrics


# ── Lightning callbacks ────────────────────────────────────────────────────────

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


# ── TC-Former building blocks ─────────────────────────────────────────────────

class WKVOperator(nn.Module):
    """RWKV-4 WKV (Eq. 5) — fully vectorized.
    Non-causal exponents masked to -inf before exp() to avoid NaN on MPS/CUDA."""

    def __init__(self, dim: int):
        super().__init__()
        self.w_log = nn.Parameter(torch.zeros(dim))
        self.u     = nn.Parameter(torch.zeros(dim))

    def forward(self, k, v):
        B, T, C = k.shape
        w = -torch.exp(self.w_log)

        idx = torch.arange(T, device=k.device, dtype=k.dtype)
        exp = (idx.unsqueeze(1) - idx.unsqueeze(0) - 1).unsqueeze(-1) * w

        causal   = idx.unsqueeze(1) > idx.unsqueeze(0)
        exp_safe = exp.masked_fill(~causal.unsqueeze(-1).expand_as(exp), float('-inf'))
        decay    = torch.exp(exp_safe)

        ek    = torch.exp(k)
        dp    = decay.permute(2, 0, 1)
        A     = torch.bmm(dp, (ek * v).permute(2, 1, 0)).permute(2, 1, 0)
        B_den = torch.bmm(dp, ek.permute(2, 1, 0)).permute(2, 1, 0)

        euk = torch.exp(self.u + k)
        return (A + euk * v) / (B_den + euk + 1e-8)


class SinCos2DPositionalEncoding(nn.Module):
    """Non-learned 2D sin-cos PE registered as a buffer (moves with model.to(device)).
    Zero trainable parameters. Requires dim % 4 == 0."""

    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        assert dim % 4 == 0, 'hidden_dim must be divisible by 4 for 2D sincos PE'
        P, d    = patch_size, dim // 4
        freq    = 1.0 / (10000 ** (torch.arange(d).float() / d))
        i_idx   = torch.arange(P).repeat_interleave(P)   # row index per token
        j_idx   = torch.arange(P).repeat(P)              # col index per token
        ri      = i_idx.unsqueeze(1) * freq.unsqueeze(0) # (P*P, d)
        cj      = j_idx.unsqueeze(1) * freq.unsqueeze(0)
        pe      = torch.stack([ri.sin(), ri.cos(), cj.sin(), cj.cos()], dim=-1)
        self.register_buffer('pe', pe.reshape(P * P, dim).unsqueeze(0))  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class BidirectionalWKVOperator(nn.Module):
    """Two independent WKV passes (forward L→R and backward R→L) fused via learned projection.
    Each direction has its own w_log and u parameters."""

    def __init__(self, dim: int):
        super().__init__()
        self.wkv_fwd = WKVOperator(dim)
        self.wkv_bwd = WKVOperator(dim)
        self.merge   = nn.Linear(2 * dim, dim, bias=False)
        # Init: treat both directions equally at the start
        with torch.no_grad():
            self.merge.weight.copy_(0.5 * torch.eye(dim).repeat(1, 2))

    def forward(self, k, v):
        fwd = self.wkv_fwd(k, v)
        bwd = self.wkv_bwd(k.flip(1), v.flip(1)).flip(1)
        return self.merge(torch.cat([fwd, bwd], dim=-1))


class TimeMixing(nn.Module):
    """Eq. 2-4, 6: time shift + element-wise WR/WK/WV + WKV + sigmoid gate."""

    def __init__(self, dim: int, wkv_cls=WKVOperator):
        super().__init__()
        self.mu_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_v = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.W_r  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_k  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_v  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_o  = nn.Linear(dim, dim, bias=False)
        self.wkv  = wkv_cls(dim)

    @staticmethod
    def _shift(x):
        return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

    def forward(self, x):
        p = self._shift(x)
        r = self.W_r * (self.mu_r * x + (1 - self.mu_r) * p)
        k = self.W_k * (self.mu_k * x + (1 - self.mu_k) * p)
        v = self.W_v * (self.mu_v * x + (1 - self.mu_v) * p)
        return self.W_o(torch.sigmoid(r) * self.wkv(k, v))


class HyperMixing(nn.Module):
    """Eq. 8-10: linear projections W'R/W'K + WKV + Mish + sigmoid gate."""

    def __init__(self, dim: int, wkv_cls=WKVOperator):
        super().__init__()
        self.mu_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.W_r  = nn.Linear(dim, dim, bias=False)
        self.W_k  = nn.Linear(dim, dim, bias=False)
        self.W_h  = nn.Linear(dim, dim, bias=False)
        self.mish = nn.Mish()
        self.wkv  = wkv_cls(dim)

    @staticmethod
    def _shift(x):
        return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

    def forward(self, x):
        p = self._shift(x)
        r = self.W_r(self.mu_r * x + (1 - self.mu_r) * p)
        k = self.W_k(self.mu_k * x + (1 - self.mu_k) * p)
        v_prime = self.wkv(k, x)
        return torch.sigmoid(r) * self.W_h(self.mish(k) * v_prime)


class TinyAttention(nn.Module):
    """Lightweight multi-head attention with shared QKV projection (single linear layer + chunk)."""

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        assert dim % num_heads == 0
        self.H     = num_heads
        self.D     = dim // num_heads
        self.scale = self.D ** -0.5
        self.qkv   = nn.Linear(dim, 3 * dim, bias=False)
        self.proj  = nn.Linear(dim, dim,     bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.H, self.D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, C))


class TimeMixFormerBlock(nn.Module):
    """Two TimeMixing layers (separate weights) + TinyAttention, each with residual + LayerNorm."""

    def __init__(self, dim: int, num_heads: int = 1, bidirectional: bool = False):
        super().__init__()
        WKV = BidirectionalWKVOperator if bidirectional else WKVOperator
        self.norm1 = nn.LayerNorm(dim)
        self.tm1   = TimeMixing(dim, WKV)
        self.norm2 = nn.LayerNorm(dim)
        self.tm2   = TimeMixing(dim, WKV)
        self.norm3 = nn.LayerNorm(dim)
        self.attn  = TinyAttention(dim, num_heads)

    def forward(self, x):
        x = x + self.tm1(self.norm1(x))
        x = x + self.tm2(self.norm2(x))
        x = x + self.attn(self.norm3(x))
        return x


class HyperMixFormerBlock(nn.Module):
    """HyperMixing called twice with shared weights + TinyAttention."""

    def __init__(self, dim: int, num_heads: int = 1, bidirectional: bool = False):
        super().__init__()
        WKV = BidirectionalWKVOperator if bidirectional else WKVOperator
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.hm    = HyperMixing(dim, WKV)
        self.attn  = TinyAttention(dim, num_heads)

    def forward(self, x):
        x = x + self.hm(self.norm1(x))
        x = x + self.hm(self.norm2(x))
        x = x + self.attn(self.norm3(x))
        return x


class CenterAttention(nn.Module):
    """Center pixel as Q, all patch tokens as K/V → single (B, D) vector."""

    def __init__(self, dim: int, patch_size: int, num_heads: int = 1):
        super().__init__()
        assert dim % num_heads == 0
        self.H          = num_heads
        self.D          = dim // num_heads
        self.scale      = self.D ** -0.5
        self.center_idx = (patch_size // 2) * patch_size + (patch_size // 2)
        self.norm    = nn.LayerNorm(dim)
        self.q_proj  = nn.Linear(dim, dim,     bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.proj    = nn.Linear(dim, dim,     bias=False)

    def forward(self, x):
        B, T, C = x.shape
        xn     = self.norm(x)
        center = xn[:, self.center_idx].unsqueeze(1)

        q    = self.q_proj(center).reshape(B, 1, self.H, self.D).transpose(1, 2)
        k, v = self.kv_proj(xn).chunk(2, dim=-1)
        k    = k.reshape(B, T, self.H, self.D).transpose(1, 2)
        v    = v.reshape(B, T, self.H, self.D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        out  = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, 1, C)
        return self.proj(out).squeeze(1) + x[:, self.center_idx]


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        h = in_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(h, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TCFormer(nn.Module):
    """PCA → Conv2D stem → TimeMixFormer×depth → HyperMixFormer×depth → CenterAttention → MLPHead."""

    def __init__(self, cfg: Config):
        super().__init__()
        D  = cfg.hidden_dim
        bi = getattr(cfg, 'use_bidirectional_wkv', False)
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.pca_components, D,
                      kernel_size=cfg.kernel_size,
                      padding=cfg.kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
        )
        self.pos_enc = (
            SinCos2DPositionalEncoding(D, cfg.patch_size)
            if getattr(cfg, 'use_pos_encoding', True) else None
        )
        self.time_blocks  = nn.ModuleList([TimeMixFormerBlock(D, cfg.num_heads, bidirectional=bi)  for _ in range(cfg.depth)])
        self.hyper_blocks = nn.ModuleList([HyperMixFormerBlock(D, cfg.num_heads, bidirectional=bi) for _ in range(cfg.depth)])
        self.center_attn  = CenterAttention(D, cfg.patch_size, cfg.num_heads)
        self.head         = MLPHead(D, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        for block in self.time_blocks:
            x = block(x)
        for block in self.hyper_blocks:
            x = block(x)
        return self.head(self.center_attn(x))


# ── Lightning wrapper ─────────────────────────────────────────────────────────

class TCFormerLit(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg       = cfg
        self.model     = TCFormer(cfg)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.epochs, eta_min=self.cfg.lr * 0.1
        )
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]
