"""Full-image, unshaved, float RGB metrics with data_range=1."""
import torch
import torch.nn.functional as F

from .color import rgb_to_ycbcr


def ssim(pred, target, window_size=11, sigma=1.5):
    # Gaussian local SSIM, valid windows; use an odd window for small fixtures.
    size = min(window_size, *pred.shape[-2:])
    size = size if size % 2 else size - 1
    if size < 1:
        raise ValueError('Empty image')
    coordinates = torch.arange(size, device=pred.device, dtype=torch.float32) - size // 2
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(pred.shape[1], 1, size, size)
    def average(x):
        return F.conv2d(x, window, groups=x.shape[1])
    # Explicitly disable autocast for stable variances in mixed precision.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        x, y = pred.float(), target.float()
        ux, uy = average(x), average(y)
        vx = (average(x*x) - ux*ux).clamp_min(0)
        vy = (average(y*y) - uy*uy).clamp_min(0)
        cov = average(x*y) - ux*uy
        score = ((2*ux*uy + 0.01**2) * (2*cov + 0.03**2)
                 / ((ux*ux + uy*uy + 0.01**2) * (vx + vy + 0.03**2)))
        return score.mean(dim=(1, 2, 3))


def psnr(pred, target):
    mse = (pred.float() - target.float()).square().mean(dim=(1, 2, 3))
    # A finite cap of 120 dB keeps identity/synthetic JSON metrics portable.
    return -10 * torch.log10(mse.clamp_min(1e-12))


def image_metrics(pred, target, gray=None):
    pred, target = pred.float().clamp(0, 1), target.float()
    delta = pred - target
    rmse_channel = delta.square().mean(dim=(2, 3)).sqrt()
    ergas = 100 * (rmse_channel / target.mean(dim=(2, 3)).clamp_min(1e-6)).square().mean(1).sqrt()
    # atan2 is stable for identical colors. Black/black -> 0 deg;
    # exactly one black vector -> 90 deg. RGB SAM is not multispectral SAM.
    a, b = pred.movedim(1, -1), target.movedim(1, -1)
    cross = torch.linalg.cross(a, b).norm(dim=-1)
    dot = (a * b).sum(-1)
    angle = torch.atan2(cross, dot)
    one_black = ((a.norm(dim=-1) < 1e-8) ^ (b.norm(dim=-1) < 1e-8))
    angle = torch.where(one_black, angle.new_tensor(torch.pi / 2), angle)
    result = {'psnr': psnr(pred, target), 'ssim': ssim(pred, target),
              'mae': delta.abs().mean(dim=(1, 2, 3)),
              'sam_deg': angle.mean(dim=(1, 2)) * (180 / torch.pi), 'ergas': ergas}
    if gray is not None:
        target_gray = rgb_to_ycbcr(target)[:, :1]
        result.update(gray_psnr=psnr(gray, target_gray), gray_ssim=ssim(gray, target_gray))
    return result
