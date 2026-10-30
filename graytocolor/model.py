"""Two differentiable U-Nets with quality-gated, multiscale SAR guidance."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .color import rgb_to_ycbcr, ycbcr_to_rgb


class ChannelNorm(nn.Module):
    """Normalize channels at each pixel; no running or spatial statistics."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).square().mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-6) * self.weight + self.bias


class GatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.spatial = nn.Conv2d(channels * 2, channels * 2, 3,
                                 padding=1, groups=channels * 2)
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        a, b = self.spatial(self.expand(self.norm(x))).chunk(2, dim=1)
        return x + self.scale * self.project(F.gelu(a) * b)


class ChannelAttention(nn.Module):
    """Global context through channel attention, without H*W squared memory."""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = math.gcd(channels, heads)
        self.norm = ChannelNorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.depthwise = nn.Conv2d(channels * 3, channels * 3, 3,
                                   padding=1, groups=channels * 3)
        self.out = nn.Conv2d(channels, channels, 1)
        self.temperature = nn.Parameter(torch.ones(self.heads, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.depthwise(self.qkv(self.norm(x))).chunk(3, dim=1)
        q, k, v = [t.reshape(b, self.heads, c // self.heads, h * w)
                   for t in (q, k, v)]
        # Keep correlation/softmax in fp32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            attention = ((q @ k.transpose(-2, -1)) * self.temperature.float()).softmax(-1)
            z = (attention @ v.float()).reshape(b, c, h, w).to(x.dtype)
        return x + self.out(z)


def blocks(channels, count):
    return nn.Sequential(*(GatedBlock(channels) for _ in range(count)))


class QualityFusion(nn.Module):
    """Learn discrepancy regions, then assess aligned SAR/optical reliability.

    Inspired by Zhao's decoupling and progressive fusion principles; this is
    a convolutional implementation, not the paper's Mamba HPDM/PFM.
    """
    def __init__(self, channels):
        super().__init__()
        self.align = nn.Conv2d(channels, channels, 1)
        self.region = nn.Sequential(nn.Conv2d(channels * 3, channels, 1),
                                    nn.GELU(), nn.Conv2d(channels, 1, 3, padding=1))
        self.coarse = nn.Sequential(nn.Conv2d(channels * 2, channels, 1),
                                    GatedBlock(channels))
        self.quality = nn.Conv2d(channels, channels, 1)
        self.refine = nn.Sequential(nn.Conv2d(channels * 3, channels, 1),
                                    GatedBlock(channels))

    def forward(self, optical, sar):
        sar = self.align(sar)
        haze = self.region(torch.cat((optical, sar, (optical - sar).abs()), 1)).sigmoid()
        coarse = self.coarse(torch.cat(((1 - haze) * optical, haze * sar), 1))
        quality = self.quality(coarse).sigmoid()
        refined = self.refine(torch.cat((quality * optical, (1 - quality) * sar,
                                        coarse), 1))
        return optical + haze * refined, haze


class StructureNet(nn.Module):
    def __init__(self, widths, depth, sar_channels, use_sar):
        super().__init__()
        self.use_sar = use_sar
        self.optical_stem = nn.Conv2d(3, widths[0], 3, padding=1)
        self.sar_stem = nn.Conv2d(sar_channels, widths[0], 3, padding=1)
        self.optical_blocks = nn.ModuleList([blocks(w, depth) for w in widths])
        self.sar_blocks = nn.ModuleList([blocks(w, depth) for w in widths])
        self.fusions = nn.ModuleList([QualityFusion(w) for w in widths])
        self.optical_down = nn.ModuleList([nn.Conv2d(a, b, 3, 2, 1)
                                           for a, b in zip(widths, widths[1:])])
        self.sar_down = nn.ModuleList([nn.Conv2d(a, b, 3, 2, 1)
                                       for a, b in zip(widths, widths[1:])])
        self.context = ChannelAttention(widths[-1])
        self.up = nn.ModuleList([nn.Conv2d(widths[i + 1] + widths[i], widths[i], 1)
                                for i in range(len(widths) - 1)])
        self.decoder = nn.ModuleList([blocks(w, depth) for w in widths[:-1]])
        self.out = nn.Conv2d(widths[0], 1, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, hazy, sar):
        x = self.optical_stem(hazy)
        s = self.sar_stem(sar) if self.use_sar else None
        features, masks = [], []
        for i, optical_block in enumerate(self.optical_blocks):
            x = optical_block(x)
            if self.use_sar:
                s = self.sar_blocks[i](s)
                x, mask = self.fusions[i](x, s)
            else:
                mask = x.new_zeros(x.shape[0], 1, *x.shape[-2:])
            features.append(x)
            masks.append(mask)
            if i < len(self.optical_down):
                x = self.optical_down[i](x)
                if self.use_sar:
                    s = self.sar_down[i](s)
        x = self.context(x)
        for i in reversed(range(len(self.up))):
            x = F.interpolate(x, size=features[i].shape[-2:], mode='bilinear',
                              align_corners=False)
            x = self.decoder[i](self.up[i](torch.cat((x, features[i]), 1)))
        base_y = rgb_to_ycbcr(hazy)[:, :1].clamp(1e-4, 1 - 1e-4)
        gray = torch.sigmoid(torch.logit(base_y) + self.out(x))
        return gray, features, masks


class ColorNet(nn.Module):
    def __init__(self, widths, depth, use_structure_features):
        super().__init__()
        self.use_structure_features = use_structure_features
        self.stem = nn.Conv2d(7, widths[0], 3, padding=1)
        self.encoder = nn.ModuleList([blocks(w, depth) for w in widths])
        self.guidance = nn.ModuleList([nn.Conv2d(w, w, 1) for w in widths])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, 2, 1)
                                  for a, b in zip(widths, widths[1:])])
        self.context = ChannelAttention(widths[-1])
        self.up = nn.ModuleList([nn.Conv2d(widths[i + 1] + widths[i], widths[i], 1)
                                for i in range(len(widths) - 1)])
        self.decoder = nn.ModuleList([blocks(w, depth) for w in widths[:-1]])
        self.out = nn.Conv2d(widths[0], 2, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, hazy, gray, features, haze):
        base_chroma = rgb_to_ycbcr(hazy)[:, 1:]
        x = self.stem(torch.cat((hazy, gray, base_chroma, haze), 1))
        skips = []
        for i, block in enumerate(self.encoder):
            if self.use_structure_features:
                x = x + self.guidance[i](features[i])
            x = block(x)
            skips.append(x)
            if i < len(self.down):
                x = self.down[i](x)
        x = self.context(x)
        for i in reversed(range(len(self.up))):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode='bilinear',
                              align_corners=False)
            x = self.decoder[i](self.up[i](torch.cat((x, skips[i]), 1)))
        chroma = base_chroma + 0.5 * torch.tanh(self.out(x))
        return ycbcr_to_rgb(gray, chroma)


class SARGrayColor(nn.Module):
    def __init__(self, width=32, depth=2, sar_channels=1, use_sar=True,
                 use_structure_features=True):
        super().__init__()
        if width < 4 or depth < 1 or sar_channels < 1:
            raise ValueError('width >= 4, depth >= 1, sar_channels >= 1 required')
        self.sar_channels = sar_channels
        widths = [width, width * 2, width * 4]
        self.structure = StructureNet(widths, depth, sar_channels, use_sar)
        self.colorizer = ColorNet(widths, depth, use_structure_features)

    def forward(self, hazy, sar, detach_structure=False, teacher_gray=None,
                teacher_ratio=0.0, structure_only=False):
        if hazy.ndim != 4 or hazy.shape[1] != 3:
            raise ValueError('hazy must be Bx3xHxW')
        if sar.ndim != 4 or sar.shape[1] != self.sar_channels:
            raise ValueError(f'SAR must be Bx{self.sar_channels}xHxW')
        if hazy.shape[0] != sar.shape[0] or hazy.shape[-2:] != sar.shape[-2:]:
            raise ValueError('SAR and optical images must be registered with equal shapes')
        gray, features, masks = self.structure(hazy, sar)
        if structure_only:
            return {'gray': gray, 'haze': masks[0], 'haze_pyramid': masks}
        color_gray = gray.detach() if detach_structure else gray
        color_features = [f.detach() for f in features] if detach_structure else features
        color_haze = masks[0].detach() if detach_structure else masks[0]
        if teacher_gray is not None:
            if not self.training or not 0 <= teacher_ratio <= 1:
                raise ValueError('Teacher guidance is training-only; ratio must be in [0,1]')
            color_gray = torch.lerp(color_gray, teacher_gray.detach(), teacher_ratio)
        rgb = self.colorizer(hazy, color_gray, color_features, color_haze)
        return {'rgb': rgb, 'gray': gray, 'haze': masks[0],
                'haze_pyramid': masks, 'chroma': rgb_to_ycbcr(rgb)[:, 1:]}
