"""Differentiable full-range BT.601 luma/chroma, in display RGB space.

Y is in [0, 1]; Cb and Cr are centered at zero. No sRGB linearization.
"""
import torch


def rgb_to_ycbcr(rgb):
    r, g, b = rgb.split(1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.cat((y, (b - y) / 1.772, (r - y) / 1.402), dim=1)


def ycbcr_to_rgb(y, chroma, gamut_map=True):
    cb, cr = chroma.split(1, dim=1)
    offsets = torch.cat((1.402 * cr,
                         -(0.114 * 1.772 * cb + 0.299 * 1.402 * cr) / 0.587,
                         1.772 * cb), dim=1)
    if gamut_map:
        # Contract chroma uniformly until RGB fits the gamut. Independent RGB
        # clipping would change Y and undermine the structure/color separation.
        bounds = torch.where(offsets >= 0, (1 - y) / offsets.abs().clamp_min(1e-6),
                             y / offsets.abs().clamp_min(1e-6))
        scale = bounds.amin(dim=1, keepdim=True).clamp(0, 1)
        offsets = offsets * scale
    return y + offsets
