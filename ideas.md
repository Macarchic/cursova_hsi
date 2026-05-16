# BiTC-Former — Ideas for Improving TC-Former

## Problem Statement

Baseline OA: **29.62%** vs. paper target **90.01%** on Indian Pines (10 samples/class).
Root causes:
1. No augmentation — 10 samples/class with zero augmentation causes overfitting
2. Causal WKV — left-to-right causality is a wrong inductive bias for 2D spatial patches (HSI patches have no temporal ordering)
3. No positional signal — after flatten+transpose the model can't distinguish center vs. corner tokens

---

## Proposed "BiTC-Former" — 4 Improvements

### A. Bidirectional WKV (Core Architectural Change)

**Motivation:** The original WKV operator is causal (left→right), inherited from language model RWKV. But HSI patch tokens are rasterized 2D spatial data — there is no causal ordering. Adding a right→left backward pass captures symmetric spectral context.

**Implementation:**
```python
class BidirectionalWKVOperator(nn.Module):
    def __init__(self, dim):
        self.wkv_fwd = WKVOperator(dim)   # independent params per direction
        self.wkv_bwd = WKVOperator(dim)
        self.merge   = nn.Linear(2*dim, dim, bias=False)
        # Init: average both directions (stable start)
        self.merge.weight.data.copy_(0.5 * torch.eye(dim).repeat(1, 2))

    def forward(self, k, v):
        fwd = self.wkv_fwd(k, v)
        bwd = self.wkv_bwd(k.flip(1), v.flip(1)).flip(1)
        return self.merge(torch.cat([fwd, bwd], dim=-1))
```

**Affects:**
- `BidirectionalTimeMixing` — identical to `TimeMixing`, only `self.wkv = BidirectionalWKVOperator(dim)`
- `BidirectionalHyperMixing` — identical to `HyperMixing`, same swap
- `TimeMixFormerBlock` + `HyperMixFormerBlock` — accept `use_bidirectional: bool` flag

**Ablation value:** "remove BiWKV" → reveals how much symmetric spectral context helps

---

### B. 2D Sinusoidal Positional Encoding

**Motivation:** The stem flattens `(B, D, P, P)` to `(B, T, D)` — all spatial structure is lost. Adding sinusoidal 2D PE tells the model which token corresponds to which `(row, col)` in the patch, crucially distinguishing the center token used by CenterAttention.

**Implementation:** Non-learned buffer, zero extra trained parameters:
```python
class SinCos2DPositionalEncoding(nn.Module):
    def __init__(self, dim, patch_size):
        # requires dim % 4 == 0 (holds for hidden_dim=64)
        P, d = patch_size, dim // 4
        freq = 1.0 / (10000 ** (torch.arange(d).float() / d))
        i_idx = torch.arange(P).repeat_interleave(P)   # row per token
        j_idx = torch.arange(P).repeat(P)              # col per token
        ri = i_idx.unsqueeze(1) * freq.unsqueeze(0)    # (T, d)
        cj = j_idx.unsqueeze(1) * freq.unsqueeze(0)
        pe = torch.stack([ri.sin(), ri.cos(), cj.sin(), cj.cos()], dim=-1).reshape(P*P, dim)
        self.register_buffer('pe', pe.unsqueeze(0))    # (1, T, D)

    def forward(self, x):
        return x + self.pe
```

Injected in `TCFormer.forward` immediately after `stem(x).flatten(2).transpose(1, 2)`, before all transformer blocks.

---

### C. Spectral + Spatial Augmentation

**Motivation:** 10 samples/class is too few without augmentation. These transforms are all physically valid for HSI:

| Transform | What it does | Why valid |
|---|---|---|
| `SpectralJitter(scale=0.1)` | Multiply each PCA band by Uniform(0.9, 1.1) | Sensor noise, atmospheric variation |
| `BandDropout(p=0.15)` | Zero out 15% of PCA bands | Band masking / missing data robustness |
| `SpatialFlip()` | Random H/V flip, p=0.5 each | No preferred scan orientation |
| `PatchRotation()` | Random 0/90/180/270° rotation | Rotation-invariant land cover |

Applied only to training set via `augment=ComposeAugmentations([...])` parameter in `HSIPatchDataset`.
Val and test always get `augment=None`.

**Note on `PatchRotation` + `CenterAttention`:** The center pixel `(P//2, P//2)` is invariant under 90°/180°/270° rotation of a square patch — safe to use.

---

### D. Label Smoothing + Cosine LR Schedule

**Label smoothing** (one-liner):
```python
self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if cfg.use_label_smoothing else 0.0)
```
Prevents overconfidence on the tiny training set.

**Cosine annealing** in `configure_optimizers`:
```python
sched = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
return [opt], [{'scheduler': sched, 'interval': 'epoch'}]
```

---

## Ablation Study Design

Each improvement is controlled by a boolean flag in `Config`:
- `use_bidirectional_wkv: bool = True`
- `use_pos_encoding: bool = True`
- `use_augmentation: bool = True`
- `use_label_smoothing: bool = True`
- `lr_scheduler: str = 'cosine'`

**Ablation table** (run full model, then remove one improvement at a time):

| Run | BiWKV | PE | Aug | LS+LR | Expected IP OA |
|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | ✗ | 29.62% (measured) |
| Full improved | ✓ | ✓ | ✓ | ✓ | 40–55% target |
| Full − BiWKV | ✗ | ✓ | ✓ | ✓ | shows A's contribution |
| Full − PE | ✓ | ✗ | ✓ | ✓ | shows B's contribution |
| Full − Aug | ✓ | ✓ | ✗ | ✓ | shows C's contribution |
| Full − LS/LR | ✓ | ✓ | ✓ | ✗ | shows D's contribution |

**CLI:**
```bash
python -m improved.train --dataset IP --paper_mode --seeds 0 1 2 3 4
python -m improved.train --dataset IP --paper_mode --seeds 0 1 2 3 4 --no_bidirectional
python -m improved.train --dataset IP --paper_mode --seeds 0 1 2 3 4 --no_pos_encoding
python -m improved.train --dataset IP --paper_mode --seeds 0 1 2 3 4 --no_augmentation
python -m improved.train --dataset IP --paper_mode --seeds 0 1 2 3 4 --no_label_smoothing --no_lr_scheduler
```

---

## Files to Modify

| File | What changes |
|---|---|
| `improved/config.py` | Add 6 new fields to `Config`; define `Config` locally (not imported from baseline) |
| `improved/model.py` | Fix import; add `BidirectionalWKVOperator`, `BidirectionalTimeMixing`, `BidirectionalHyperMixing`, `SinCos2DPositionalEncoding`; update `TimeMixFormerBlock`, `HyperMixFormerBlock`, `TCFormer`, `TCFormerLit` |
| `improved/dataset.py` | Add 5 augmentation classes; add `augment` param to `HSIPatchDataset`; update `get_dataloaders` |
| `improved/train.py` | Fix imports; add 5 ablation CLI flags; apply flags to config; rename run dir to `improved_*` |

---

## Verification Steps

```bash
# Shape + NaN check
python -c "
import torch
from improved.config import CONFIGS
from improved.model import TCFormer
cfg = CONFIGS['IP']
out = TCFormer(cfg)(torch.randn(4, 150, 15, 15))
assert out.shape == (4, 16) and not out.isnan().any()
print('OK', out.shape)
"

# Augmentation sanity check
python -c "
from improved.dataset import SpectralJitter, BandDropout, SpatialFlip, PatchRotation, ComposeAugmentations
import torch
aug = ComposeAugmentations([SpectralJitter(), BandDropout(), SpatialFlip(), PatchRotation()])
x = torch.randn(150, 15, 15)
y = aug(x)
print('Before:', round(x.min().item(),3), round(x.max().item(),3))
print('After: ', round(y.min().item(),3), round(y.max().item(),3))
"

# Full run
python -m improved.train --dataset IP --paper_mode --seeds 42
```
