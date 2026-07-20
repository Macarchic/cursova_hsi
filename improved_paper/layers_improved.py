"""Improved TC-Former layers, copied from improved/model.py.

Clean, self-contained RWKV-4 implementation plus the enhancement modules
(SE, 2D sin-cos PE, bidirectional WKV, multi-scale stem, multi-ring center
attention). Each is wired into the model behind a config flag (see model.py).
"""

import torch
import torch.nn as nn


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


class MultiScaleStem(nn.Module):
    """Three parallel Conv2D branches (kernels: 3, 5, kernel_size), outputs summed."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()

        def branch(k):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.small  = branch(3)
        self.medium = branch(5)
        self.large  = branch(kernel_size)

    def forward(self, x):
        return self.small(x) + self.medium(x) + self.large(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: reweight D feature channels based on global context."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):           # x: (B, D, H, W)
        s = x.mean(dim=[2, 3])
        s = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * s


class SinCos2DPositionalEncoding(nn.Module):
    """Non-learned 2D sin-cos PE registered as a buffer. Requires dim % 4 == 0."""

    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        assert dim % 4 == 0, 'hidden_dim must be divisible by 4 for 2D sincos PE'
        P, d    = patch_size, dim // 4
        freq    = 1.0 / (10000 ** (torch.arange(d).float() / d))
        i_idx   = torch.arange(P).repeat_interleave(P)
        j_idx   = torch.arange(P).repeat(P)
        ri      = i_idx.unsqueeze(1) * freq.unsqueeze(0)
        cj      = j_idx.unsqueeze(1) * freq.unsqueeze(0)
        pe      = torch.stack([ri.sin(), ri.cos(), cj.sin(), cj.cos()], dim=-1)
        self.register_buffer('pe', pe.reshape(P * P, dim).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class BidirectionalWKVOperator(nn.Module):
    """Two independent WKV passes (forward L→R and backward R→L) fused via learned projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.wkv_fwd = WKVOperator(dim)
        self.wkv_bwd = WKVOperator(dim)
        self.merge   = nn.Linear(2 * dim, dim, bias=False)
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
    """Lightweight multi-head attention with shared QKV projection."""

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


class MultiRingCenterAttention(nn.Module):
    """Center pixel as Q, attending separately to Ring1 (8 immediate neighbors)
    and Ring2 (all remaining tokens). Outputs fused via learned projection.
    Replaces the original center-token selection — same (B,T,D) -> (B,D) interface."""

    def __init__(self, dim: int, patch_size: int, num_heads: int = 1):
        super().__init__()
        assert dim % num_heads == 0
        self.H     = num_heads
        self.D     = dim // num_heads
        self.scale = self.D ** -0.5

        P = patch_size
        cy, cx = P // 2, P // 2
        self.center_idx = cy * P + cx

        ring1 = [
            (cy + di) * P + (cx + dj)
            for di in (-1, 0, 1) for dj in (-1, 0, 1)
            if not (di == 0 and dj == 0)
        ]
        ring2 = [i for i in range(P * P) if i != self.center_idx and i not in ring1]

        self.register_buffer('ring1_idx', torch.tensor(ring1))
        self.register_buffer('ring2_idx', torch.tensor(ring2))

        self.norm    = nn.LayerNorm(dim)
        self.q_proj  = nn.Linear(dim, dim,     bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.merge   = nn.Linear(2 * dim, dim, bias=False)
        self.proj    = nn.Linear(dim, dim,     bias=False)

        with torch.no_grad():
            self.merge.weight.copy_(0.5 * torch.eye(dim).repeat(1, 2))

    def _ring_attn(self, q, k, v, idx):
        B   = q.shape[0]
        n   = idx.shape[0]
        k_r = k[:, idx].reshape(B, n, self.H, self.D).transpose(1, 2)
        v_r = v[:, idx].reshape(B, n, self.H, self.D).transpose(1, 2)
        attn = (q @ k_r.transpose(-2, -1)) * self.scale
        return (attn.softmax(dim=-1) @ v_r).transpose(1, 2).reshape(B, 1, self.H * self.D)

    def forward(self, x):
        B, T, C = x.shape
        xn     = self.norm(x)
        center = xn[:, self.center_idx].unsqueeze(1)

        q    = self.q_proj(center).reshape(B, 1, self.H, self.D).transpose(1, 2)
        k, v = self.kv_proj(xn).chunk(2, dim=-1)

        out_r1 = self._ring_attn(q, k, v, self.ring1_idx)
        out_r2 = self._ring_attn(q, k, v, self.ring2_idx)

        fused = self.merge(torch.cat([out_r1, out_r2], dim=-1)).squeeze(1)
        return self.proj(fused) + x[:, self.center_idx]


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
