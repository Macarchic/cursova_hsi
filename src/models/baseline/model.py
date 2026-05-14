import torch
import torch.nn as nn

from src.models.base_lit import BaseLitModule
from src.models.baseline.config import Config


class WKVOperator(nn.Module):
    """
    RWKV-4 WKV (Eq. 5) — fully vectorized, no Python loop.

    For causal positions (t > s):  decay[t,s,c] = exp((t-s-1)*w[c])
    For non-causal (t <= s):       decay[t,s,c] = 0

    IMPORTANT: we mask non-causal exponents to -inf BEFORE exp()
    so that exp(-inf)=0. Without this, exp(+large) * False = Inf*0 = NaN
    on MPS/CUDA, which kills all gradients.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w_log = nn.Parameter(torch.zeros(dim))
        self.u     = nn.Parameter(torch.zeros(dim))

    def forward(self, k, v):
        B, T, C = k.shape
        w = -torch.exp(self.w_log)

        idx = torch.arange(T, device=k.device, dtype=k.dtype)
        exp = (idx.unsqueeze(1) - idx.unsqueeze(0) - 1).unsqueeze(-1) * w

        causal = (idx.unsqueeze(1) > idx.unsqueeze(0))
        exp_safe = exp.masked_fill(~causal.unsqueeze(-1).expand_as(exp), float('-inf'))
        decay = torch.exp(exp_safe)

        ek     = torch.exp(k)
        d_perm = decay.permute(2, 0, 1)
        A      = torch.bmm(d_perm, (ek * v).permute(2, 1, 0)).permute(2, 1, 0)
        B_den  = torch.bmm(d_perm, ek.permute(2, 1, 0)).permute(2, 1, 0)

        euk = torch.exp(self.u + k)
        return (A + euk * v) / (B_den + euk + 1e-8)


class TimeMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mu_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_v = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.W_r  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_k  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_v  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_o  = nn.Linear(dim, dim, bias=False)
        self.wkv  = WKVOperator(dim)

    @staticmethod
    def _shift(x):
        return torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

    def forward(self, x):
        p = self._shift(x)
        r = self.W_r * (self.mu_r * x + (1 - self.mu_r) * p)
        k = self.W_k * (self.mu_k * x + (1 - self.mu_k) * p)
        v = self.W_v * (self.mu_v * x + (1 - self.mu_v) * p)
        return self.W_o(torch.sigmoid(r) * self.wkv(k, v))


class HyperMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mu_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mu_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.W_r  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_k  = nn.Parameter(torch.ones(1, 1, dim))
        self.W_h  = nn.Linear(dim, dim, bias=False)
        self.mish = nn.Mish()
        self.wkv  = WKVOperator(dim)

    @staticmethod
    def _shift(x):
        return torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

    def forward(self, x):
        p = self._shift(x)
        r = self.W_r * (self.mu_r * x + (1 - self.mu_r) * p)
        k = self.W_k * (self.mu_k * x + (1 - self.mu_k) * p)
        v_prime = self.wkv(k, x)
        return torch.sigmoid(r) * self.W_h(self.mish(k) * v_prime)


class TinyAttention(nn.Module):
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
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TimeMixFormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.tm1   = TimeMixing(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.tm2   = TimeMixing(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.attn  = TinyAttention(dim, num_heads)

    def forward(self, x):
        x = x + self.tm1(self.norm1(x))
        x = x + self.tm2(self.norm2(x))
        x = x + self.attn(self.norm3(x))
        return x


class HyperMixFormerBlock(nn.Module):
    """HyperMixing called twice with shared weights + TinyAttention."""

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.hm    = HyperMixing(dim)
        self.attn  = TinyAttention(dim, num_heads)

    def forward(self, x):
        x = x + self.hm(self.norm1(x))
        x = x + self.hm(self.norm2(x))
        x = x + self.attn(self.norm3(x))
        return x


class CenterAttention(nn.Module):
    """Central token as Q; all tokens as K/V → single (B, D) vector."""

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
        xn = self.norm(x)
        center = xn[:, self.center_idx, :].unsqueeze(1)

        q  = self.q_proj(center).reshape(B, 1, self.H, self.D).transpose(1, 2)
        kv = self.kv_proj(xn)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, T, self.H, self.D).transpose(1, 2)
        v = v.reshape(B, T, self.H, self.D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        out  = self.proj(out).squeeze(1)
        return out + x[:, self.center_idx, :]


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
    """
    TC-Former: PCA → Conv2D Stem → TimeMixFormer×depth → HyperMixFormer×depth
               → CenterAttention → MLPHead
    """

    def __init__(self, cfg: Config):
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
        self.time_mix_formers  = nn.ModuleList(
            [TimeMixFormerBlock(D, cfg.num_heads) for _ in range(cfg.depth)]
        )
        self.hyper_mix_formers = nn.ModuleList(
            [HyperMixFormerBlock(D, cfg.num_heads) for _ in range(cfg.depth)]
        )
        self.center_attn = CenterAttention(D, cfg.patch_size, cfg.num_heads)
        self.head        = MLPHead(D, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        for block in self.time_mix_formers:
            x = block(x)
        for block in self.hyper_mix_formers:
            x = block(x)
        x = self.center_attn(x)
        return self.head(x)


class TCFormerLit(BaseLitModule):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.model = TCFormer(cfg)

    def forward(self, x):
        return self.model(x)
