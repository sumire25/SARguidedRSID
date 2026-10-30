"""Full-frame inference and overlap blending for memory-limited GPUs."""
import torch


@torch.inference_mode()
def predict(model, hazy, sar, tile_size=0, overlap=32):
    if tile_size < 0:
        raise ValueError('tile_size must be zero (full frame) or positive')
    h, w = hazy.shape[-2:]
    if not tile_size or (h <= tile_size and w <= tile_size):
        result = model(hazy, sar)
        return {k: result[k].float() for k in ('rgb', 'gray', 'haze')}
    if overlap < 0 or overlap >= tile_size:
        raise ValueError('Require 0 <= overlap < tile_size')
    def starts(length):
        if length <= tile_size:
            return [0]
        positions = list(range(0, length-tile_size+1, tile_size-overlap))
        if positions[-1] != length-tile_size:
            positions.append(length-tile_size)
        return positions
    result = {key: torch.zeros(hazy.shape[0], channels, h, w, device=hazy.device)
              for key, channels in (('rgb', 3), ('gray', 1), ('haze', 1))}
    weights = torch.zeros(1, 1, h, w, device=hazy.device)
    for top in starts(h):
        for left in starts(w):
            region = (..., slice(top, min(h, top+tile_size)), slice(left, min(w, left+tile_size)))
            output = model(hazy[region], sar[region])
            th, tw = output['gray'].shape[-2:]
            wy = torch.hann_window(th, periodic=False, device=hazy.device).clamp_min(0.1)
            wx = torch.hann_window(tw, periodic=False, device=hazy.device).clamp_min(0.1)
            window = (wy[:, None] * wx[None, :])[None, None]
            for key in result:
                result[key][region] += output[key].float() * window
            weights[region] += window
    return {key: value / weights for key, value in result.items()}
