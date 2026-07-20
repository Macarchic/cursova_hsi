import math
from utils import device
import torch
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
import torch.nn as nn

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

def Init(module):
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected nn.Module, but got {type(module)}")

    for m in module.modules():  # 遍历所有子模块
        if isinstance(m, (nn.Linear, nn.Embedding)):
            with torch.no_grad():
                shape = m.weight.data.shape
                gain = 1.0
                scale = getattr(m, 'scale_init', 1.0)

                if isinstance(m, nn.Linear):
                    if m.bias is not None:
                        m.bias.data.zero_()
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                if isinstance(m, nn.Embedding):
                    gain = math.sqrt(max(shape[0], shape[1]))

                gain *= scale
                if gain == 0:
                    nn.init.zeros_(m.weight)
                elif gain > 0:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.normal_(m.weight, mean=0, std=-gain)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class TimeMixFormer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_attn = params.get('n_attn')
        self.n_head = params.get('n_head')
        self.ctx_len = params.get('ctx_len')
        self.n_embd = params.get('n_embd')
        assert self.n_attn % self.n_head == 0, "n_attn must be divisible by n_head"
        self.head_size = self.n_attn // self.n_head

        # 初始化 time_w 曲线
        with torch.no_grad():
            ww = torch.ones(self.n_head, self.ctx_len)
            curve = torch.tensor([-(self.ctx_len - 1 - i) for i in range(self.ctx_len)])
            for h in range(self.n_head):
                if h < self.n_head - 1:
                    decay_speed = math.pow(self.ctx_len, -(h + 1) / (self.n_head - 1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
            self.time_w = nn.Parameter(ww)
        # 初始化
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, self.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, self.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(self.ctx_len, 1))
        # 时间位移操作
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # 定义线性层
        self.key = nn.Linear(128, 64)
        self.value = nn.Linear(128, 64)
        self.receptance = nn.Linear(128, 64)
        # 输出层
        self.output = nn.Linear(64, 64)
        # 初始化 scale_init
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        if x.shape[-1] != 128:
            x = nn.Linear(x.shape[-1], 128).to(device)(x)
        # 构造时间权重矩阵
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        # 时间位移操作
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        # 计算 key、value 和 receptance
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        # 动态计算 head_size
        assert self.n_head > 0, "n_head must be greater than 0"
        assert k.size(-1) % self.n_head == 0, "k.size(-1) must be divisible by n_head"
        head_size = k.size(-1) // self.n_head
        # 计算加权 value
        kv = (k * v).view(B, T, self.n_head, head_size)
        # 检查 w 和 kv 的值
        if torch.isnan(w).any() or torch.isinf(w).any():
            raise ValueError("w contains NaN or inf values")
        if torch.isnan(kv).any() or torch.isinf(kv).any():
            raise ValueError("kv contains NaN or inf values")

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)
        # 计算最终输出
        eps = 1e-8
        r = torch.clamp(r, min=-50, max=50)
        rwkv = torch.sigmoid(r) * wkv / (torch.cumsum(k, dim=1) + eps)
        rwkv = self.output(rwkv)

        return rwkv * self.time_gamma[:T, :]

class ChannelMixFormer(nn.Module):
    def __init__(self,  params):
        super().__init__()
        self.n_ffn = params.get('n_ffn')
        hidden_sz = 5 * params.get('n_ffn') // 2
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        input_dim = 256
        self.key = nn.Linear(input_dim, hidden_sz)
        self.value = nn.Linear(input_dim, hidden_sz)
        self.receptance = nn.Linear(input_dim, params.get('n_embd'))
        self.weight = nn.Linear(hidden_sz, params.get('n_embd'))
        self.output_adjust = nn.Linear(128, 64)
        self.receptance.scale_init = 0
        self.weight.scale_init = 0
    def forward(self, x):
        B, T, C = x.size()
        # 时间位移操作
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=2)
        if x.shape[-1] != self.key.in_features:
            x = nn.Linear(x.shape[-1], self.key.in_features).to(device)(x)
        # 计算 key、value 和 receptance
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        # 计算加权 value
        wkv = self.weight(F.mish(k) * v)
        rwkv = torch.sigmoid(r) * wkv
        rwkv = self.output_adjust(rwkv)

        return rwkv

class TinyAttn(nn.Module):
    def __init__(self,  params):
        super().__init__()
        self.params = params
        net_params = params['net']
        self.d_attn = net_params.get('tiny_attn')
        self.n_head = net_params.get('tiny_head')
        self.head_size = self.d_attn // self.n_head
        self.qkv = nn.Linear(net_params.get('n_embd') // 2, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, net_params.get('n_embd'))
        self.adjust_dim = nn.Linear(net_params.get('n_embd'), 64)
        self.out1 = nn.Linear(self.d_attn, 64)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        qk = self.dropout(qk)
        qk = qk.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v

        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)
            # qkv.transpose(1, 2)

        out = self.out(qkv)
        out = self.adjust_dim(out)

        return out

class former(nn.Module):
    def __init__(self, dim, depth,  params, heads, dim_heads, mlp_dim, dropout):
        super(former, self).__init__()
        self.layers = nn.ModuleList([])
        self.tiny_attn = TinyAttn(params)
        self.channel_mix = ChannelMixFormer(params['net'])
        self.time_mix = TimeMixFormer(params['net'])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNormalize(dim, Residual(self.time_mix)),
                LayerNormalize(dim,Residual(self.channel_mix)),
            ]))
    def forward(self, x):
        x_center_attention = []
        for time_mix,channel_mix in self.layers:
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.float32).to(device)
            x = time_mix(time_mix(x))
            x = self.tiny_attn(x, mask)
            x = channel_mix(channel_mix(x))
            x = self.tiny_attn(x, mask)
            index = int(x.shape[1] // 2)
            x_center_attention.append(x[:, index, :])
        return x,x_center_attention

class SCMT(nn.Module):
    def __init__(self, params):
        super(SCMT, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']
        self.model_type = net_params.get("model_type", 0)
        num_classes = data_params.get("num_classes", 16)
        self.spectral_size = data_params.get("spectral_size", 200)

        depth = net_params.get("depth", 2)
        heads = net_params.get("heads", 8)
        mlp_dim = net_params.get("mlp_dim", 8)
        kernal = net_params.get('kernal', 3)
        padding = net_params.get('padding', 1)
        dropout = net_params.get("dropout", 0)
        conv2d_out = 64
        dim = net_params.get("dim", 64)
        dim_heads = dim

        self.local_trans_pixel = former(dim, depth, params, heads, dim_heads, mlp_dim, dropout)
        self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.001)
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernal, kernal),
                      padding=(padding, padding)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
        )
        self.mlp_head = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)
        self.dropout = nn.Dropout(0.1)
        linear_dim = dim * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )

    def encoder_block(self, x):
        x_pixel = x
        x_pixel = self.conv2d_features(x_pixel)
        x_pixel = rearrange(x_pixel, 'b s w h -> b (w h) s')
        x_pixel = self.dropout(x_pixel)
        x_pixel, x_center_list = self.local_trans_pixel(x_pixel)
        logit_pixel = x_center_list[-1]
        logit_x = logit_pixel
        reduce_x = torch.mean(x_pixel, dim=1)
        return logit_x, reduce_x
    def forward(self, x, left=None, right=None):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height

        '''
        logit_x, _ = self.encoder_block(x)
        mean_left, mean_right = None, None
        if left is not None and right is not None:
            _, mean_left = self.encoder_block(left)
            _, mean_right = self.encoder_block(right)
        return self.classifier_mlp(logit_x)
