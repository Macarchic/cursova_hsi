"""
HSI patch augmentations — used only by the improved model.
All transforms operate on float tensors of shape (C, H, W).
"""
import random
import torch
import torch.nn.functional as F


class SpectralNoise:
    """Add Gaussian noise to all spectral channels."""
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class SpectralDropout:
    """Randomly zero out entire spectral bands."""
    def __init__(self, keep_prob: float = 0.9):
        self.keep_prob = keep_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(torch.full((x.shape[0],), self.keep_prob))
        return x * mask.view(-1, 1, 1)


class SpectralShift:
    """Shift all spectral values by a small random offset."""
    def __init__(self, max_shift: float = 0.02):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = (torch.rand(1).item() * 2 - 1) * self.max_shift
        return x + shift


class SpectralScale:
    """Scale all spectral values by a random factor close to 1."""
    def __init__(self, scale_range: tuple = (0.95, 1.05)):
        self.lo, self.hi = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.lo + torch.rand(1).item() * (self.hi - self.lo)
        return x * scale


class RandomFlip:
    """Random horizontal and/or vertical spatial flip."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = x.flip(dims=[2])  # horizontal
        if random.random() < 0.5:
            x = x.flip(dims=[1])  # vertical
        return x


class RandomRot90:
    """Random 90-degree rotation (0 / 90 / 180 / 270)."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = random.randint(0, 3)
        return torch.rot90(x, k, dims=[1, 2])


class CutoutPatch:
    """
    Zero out a random square region of the spatial patch.
    cut_ratio controls the fraction of the patch side that can be erased.
    """
    def __init__(self, cut_ratio: float = 0.25):
        self.cut_ratio = cut_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W = x.shape
        cut_h = max(1, int(H * self.cut_ratio))
        cut_w = max(1, int(W * self.cut_ratio))
        r = random.randint(0, H - cut_h)
        c = random.randint(0, W - cut_w)
        x = x.clone()
        x[:, r:r + cut_h, c:c + cut_w] = 0.0
        return x


class Compose:
    """Apply a list of transforms sequentially."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply:
    """Apply each transform independently with probability p."""
    def __init__(self, transforms: list, p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            if random.random() < self.p:
                x = t(x)
        return x


def build_train_transform() -> Compose:
    """Default augmentation pipeline for improved model training."""
    return Compose([
        RandomFlip(),
        RandomRot90(),
        RandomApply([
            SpectralNoise(std=0.01),
            SpectralDropout(keep_prob=0.9),
            SpectralShift(max_shift=0.02),
            SpectralScale(scale_range=(0.95, 1.05)),
            CutoutPatch(cut_ratio=0.25),
        ], p=0.5),
    ])
