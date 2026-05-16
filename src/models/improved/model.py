import math
import torch
import torch.nn as nn

from src.models.base_lit import BaseLitModule
from src.models.baseline.model import WKVOperator, TinyAttention, MLPHead
from src.models.improved.config import CONFIGS


class MultiScaleCenterAttention(nn.Module):
    """
    MS-CA: center pixel as Q; neighbors from 3 nested scales as K/V.

    For each scale a separate KV projection is learned, so the model can
    extract different semantics from local vs. global context.
    Scale outputs are concatenated and projected back to dim.

    Default scales: 3×3 (local texture), 7×7 (mid-range), full patch (global).
    """

    def __init__(self, dim: int, patch_size: int, num_heads: int = 1, scales: tuple = (3, 7)):
        super().__init__()
        P = patch_size
        assert dim % num_heads == 0
        self.H = num_heads
        self.D = dim // num_heads
        self.attn_scale = self.D ** -0.5
        self.center_idx = (P // 2) * P + (P // 2)

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)

        cr, cc = P // 2, P // 2
        all_scales = list(scales) + [None]  # None = full patch
        self.n_scales = len(all_scales)

        self.kv_projs = nn.ModuleList(
            [nn.Linear(dim, 2 * dim, bias=False) for _ in range(self.n_scales)]
        )
        self.out_proj = nn.Linear(dim * self.n_scales, dim, bias=False)

        for i, s in enumerate(all_scales):
            if s is None:
                idx = list(range(P * P))
            else:
                half = s // 2
                idx = [r * P + c
                       for r in range(cr - half, cr + half + 1)
                       for c in range(cc - half, cc + half + 1)
                       if 0 <= r < P and 0 <= c < P]
            self.register_buffer(f'_idx_{i}', torch.tensor(idx, dtype=torch.long))

    def _idx(self, i: int) -> torch.Tensor:
        return getattr(self, f'_idx_{i}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        B, _, C = x.shape
        xn = self.norm(x)
        center = xn[:, self.center_idx, :].unsqueeze(1)         # (B, 1, D)
        q = self.q_proj(center).reshape(B, 1, self.H, self.D).transpose(1, 2)  # (B, H, 1, Dh)

        scale_outs = []
        for i, kv_proj in enumerate(self.kv_projs):
            tokens = xn[:, self._idx(i), :]                     # (B, S, D)
            S = tokens.shape[1]
            k, v = kv_proj(tokens).chunk(2, dim=-1)
            k = k.reshape(B, S, self.H, self.D).transpose(1, 2)  # (B, H, S, Dh)
            v = v.reshape(B, S, self.H, self.D).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B, H, 1, S)
            out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, 1, C)
            scale_outs.append(out)

        fused = torch.cat(scale_outs, dim=-1)                   # (B, 1, D*n_scales)
        out = self.out_proj(fused).squeeze(1)                   # (B, D)
        return out + x[:, self.center_idx, :]                   # residual


class BidirectionalWKV(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.wkv_fwd = WKVOperator(dim)
        self.wkv_bwd = WKVOperator(dim)
        self.merge    = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, k, v):
        fwd = self.wkv_fwd(k, v)
        bwd = self.wkv_bwd(k.flip(1), v.flip(1)).flip(1)
        return self.merge(torch.cat([fwd, bwd], dim=-1))


class TimeMixingBi(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mu_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_v = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.W_r  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_k  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_v  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_o  = nn.Linear(dim, dim, bias=False)
        self.wkv  = BidirectionalWKV(dim)

    @staticmethod
    def _shift(x):
        return torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

    def forward(self, x):
        p = self._shift(x)
        r = self.W_r * (self.mu_r * x + (1 - self.mu_r) * p)
        k = self.W_k * (self.mu_k * x + (1 - self.mu_k) * p)
        v = self.W_v * (self.mu_v * x + (1 - self.mu_v) * p)
        return self.W_o(torch.sigmoid(r) * self.wkv(k, v))


class HyperMixingBi(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mu_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.W_r  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_k  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_h  = nn.Linear(dim, dim, bias=False)
        self.mish = nn.Mish()
        self.wkv  = BidirectionalWKV(dim)

    @staticmethod
    def _shift(x):
        return torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

    def forward(self, x):
        p = self._shift(x)
        r = self.W_r * (self.mu_r * x + (1 - self.mu_r) * p)
        k = self.W_k * (self.mu_k * x + (1 - self.mu_k) * p)
        v_prime = self.wkv(k, x)
        return torch.sigmoid(r) * self.W_h(self.mish(k) * v_prime)


class TimeMixFormerBlockBi(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.tm1   = TimeMixingBi(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.tm2   = TimeMixingBi(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.attn  = TinyAttention(dim, num_heads)

    def forward(self, x):
        x = x + self.tm1(self.norm1(x))
        x = x + self.tm2(self.norm2(x))
        x = x + self.attn(self.norm3(x))
        return x


class HyperMixFormerBlockBi(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.hm    = HyperMixingBi(dim)
        self.attn  = TinyAttention(dim, num_heads)

    def forward(self, x):
        x = x + self.hm(self.norm1(x))
        x = x + self.hm(self.norm2(x))
        x = x + self.attn(self.norm3(x))
        return x


class Pos2D(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        P = patch_size
        pe = torch.zeros(P * P, dim)
        pos_r = torch.arange(P).float().unsqueeze(1).repeat(1, P).flatten()
        pos_c = torch.arange(P).float().unsqueeze(0).repeat(P, 1).flatten()
        div = torch.exp(torch.arange(0, dim // 2, 2).float() * -(math.log(10000.0) / (dim // 2)))
        pe[:, 0::4] = torch.sin(pos_r.unsqueeze(1) * div)
        pe[:, 1::4] = torch.cos(pos_r.unsqueeze(1) * div)
        pe[:, 2::4] = torch.sin(pos_c.unsqueeze(1) * div)
        pe[:, 3::4] = torch.cos(pos_c.unsqueeze(1) * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe


class TCFormerImproved(nn.Module):
    """
    TC-Former + Bidirectional WKV + 2D Positional Encoding.
    """

    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.pca_components, D,
                      kernel_size=cfg.kernel_size,
                      padding=cfg.kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
        )
        self.pos = Pos2D(D, cfg.patch_size)
        self.time_mix_formers  = nn.ModuleList(
            [TimeMixFormerBlockBi(D, cfg.num_heads) for _ in range(cfg.depth)]
        )
        self.hyper_mix_formers = nn.ModuleList(
            [HyperMixFormerBlockBi(D, cfg.num_heads) for _ in range(cfg.depth)]
        )
        self.center_attn = MultiScaleCenterAttention(D, cfg.patch_size, cfg.num_heads)
        self.head        = MLPHead(D, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos(x)
        for block in self.time_mix_formers:
            x = block(x)
        for block in self.hyper_mix_formers:
            x = block(x)
        x = self.center_attn(x)
        return self.head(x)


class TCFormerImprovedLit(BaseLitModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = TCFormerImproved(cfg)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}
