"""Evaluate saved (8-bit quantized) RGB PNGs against the GT directory."""
import argparse
import csv
import json
from pathlib import Path

import torch

from graytocolor.data import index_pngs, load_rgb
from graytocolor.metrics import image_metrics
from graytocolor.runtime import select_device


def evaluate(predictions, target_dir, output_dir, device='auto'):
    pred_map, gt_map = index_pngs(predictions), index_pngs(target_dir)
    if set(pred_map) != set(gt_map):
        raise ValueError('Prediction IDs and GT IDs must match exactly; missing/extra PNGs found')
    device = select_device(device)
    rows = []
    with torch.inference_mode():
        for key in sorted(pred_map):
            pred = load_rgb(pred_map[key])[None].to(device)
            target = load_rgb(gt_map[key])[None].to(device)
            if pred.shape != target.shape:
                raise ValueError(f'Shape mismatch for {key}: {pred.shape} vs {target.shape}')
            rows.append({'id': key, **{k: v.item() for k, v in image_metrics(pred, target).items()}})
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with open(destination / 'per_image.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'images': len(rows), 'metric_protocol': '8-bit PNG RGB, full image, no shaving; '
              'Gaussian valid SSIM; mean per-image; RGB SAM degrees; ERGAS ratio=1',
              'mean': {k: sum(r[k] for r in rows)/len(rows) for k in rows[0] if k != 'id'}}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, help='Directory of dehazed RGB PNGs')
    parser.add_argument('--targets', required=True, help='Directory of haze-free RGB PNGs')
    parser.add_argument('--output-dir', default='/kaggle/working/results/png_metrics')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    evaluate(args.predictions, args.targets, args.output_dir, args.device)
