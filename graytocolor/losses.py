import torch
from torch import nn
import torch.nn.functional as F

from .color import rgb_to_ycbcr
from .metrics import ssim


def edge_loss(pred, target):
    return (F.l1_loss(pred[..., 1:, :]-pred[..., :-1, :],
                      target[..., 1:, :]-target[..., :-1, :])
            + F.l1_loss(pred[..., :, 1:]-pred[..., :, :-1],
                        target[..., :, 1:]-target[..., :, :-1])) / 2


def frequency_loss(pred, target):
    # Compare complex spectra (including phase), not magnitude alone.
    difference = torch.fft.rfft2(pred.float(), norm='ortho') - torch.fft.rfft2(target.float(), norm='ortho')
    return difference.abs().mean()


class RestorationLoss(nn.Module):
    def __init__(self, weights, use_sar=True):
        super().__init__()
        self.weights = weights
        self.use_sar = use_sar

    def forward(self, output, hazy, target, phase='joint', sar_available=None):
        # All reconstruction terms are computed in fp32, even under AMP.
        with torch.autocast(device_type=target.device.type, enabled=False):
            target = target.float()
            target_ycc = rgb_to_ycbcr(target)
            gray = output['gray'].float()
            terms = {}
            if phase != 'color':
                terms['gray'] = F.l1_loss(gray, target_ycc[:, :1])
                terms['gray_ssim'] = (1-ssim(gray, target_ycc[:, :1])).mean()
                terms['edge'] = edge_loss(gray, target_ycc[:, :1])
                if self.use_sar and self.weights['gate']:
                    # A paired optical difference proxy, not physical transmission.
                    proxy = ((hazy.float()-target).abs().mean(1, keepdim=True)
                             / self.weights['gate_difference_scale']).clamp(0, 1).detach()
                    gate_losses = []
                    for mask in output['haze_pyramid']:
                        label = F.interpolate(proxy, size=mask.shape[-2:], mode='area')
                        loss = F.binary_cross_entropy(mask.float().clamp(1e-6, 1-1e-6),
                                                      label, reduction='none')
                        if sar_available is not None:
                            weight = sar_available.float()
                            loss = (loss * weight).sum() / (weight.sum() * loss.shape[-2] * loss.shape[-1]).clamp_min(1)
                        else:
                            loss = loss.mean()
                        gate_losses.append(loss)
                    terms['gate'] = torch.stack(gate_losses).mean()
            if phase != 'gray':
                rgb = output['rgb'].float()
                terms['rgb'] = F.l1_loss(rgb, target)
                terms['chroma'] = F.l1_loss(rgb_to_ycbcr(rgb)[:, 1:], target_ycc[:, 1:])
                terms['rgb_ssim'] = (1-ssim(rgb, target)).mean()
                if self.weights['frequency']:
                    terms['frequency'] = frequency_loss(rgb, target)
            total = sum(self.weights[name] * value for name, value in terms.items())
        return total, {name: value.detach().item() for name, value in terms.items()}
