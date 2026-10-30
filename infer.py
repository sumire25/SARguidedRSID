"""Inference, optional intermediate images, and paired full-resolution metrics."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from graytocolor.data import MRSHazeDataset, dataset_options, paired_records
from graytocolor.inference import predict
from graytocolor.metrics import image_metrics
from graytocolor.model import SARGrayColor
from graytocolor.runtime import load_checkpoint, select_device


def save_image(tensor, path):
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    array = np.rint(array.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).save(path)


def run(args):
    checkpoint = load_checkpoint(args.weights)
    config = checkpoint['config']
    if checkpoint['phase'] == 'gray':
        raise ValueError('This checkpoint has only trained structure recovery. Finish color/joint '
                         'training before evaluating RGB dehazing.')
    device = select_device(args.device)
    model = SARGrayColor(**config['model']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    root = args.data_root or config['data']['root']
    records = paired_records(root, 'train' if args.split == 'val' else args.split,
                             require_gt=not args.no_gt)
    if args.split == 'val':
        val_ids = set(checkpoint['split']['val_ids'])
        records = [r for r in records if r['id'] in val_ids]
        if {r['id'] for r in records} != val_ids:
            raise ValueError('Dataset does not contain all saved validation IDs')
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        records = records[:args.limit]
    dataset = MRSHazeDataset(records, **dataset_options(config))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ['rgb'] + (['gray', 'haze'] if args.save_aux else []):
        (output_dir / name).mkdir(exist_ok=True)
    rows = []
    amp = args.amp and device.type == 'cuda'
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            hazy, sar = [batch[k].to(device) for k in ('hazy', 'sar')]
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = predict(model, hazy, sar, args.tile_size, args.overlap)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            row = {'id': batch['id'][0], 'inference_seconds': time.perf_counter()-started}
            if not args.no_gt:
                gt = batch['gt'].to(device)
                for key, value in image_metrics(prediction['rgb'], gt, prediction['gray']).items():
                    row[key] = value.item()
                for key, value in image_metrics(hazy, gt).items():
                    row[f'hazy_{key}'] = value.item()
            rows.append(row)
            for name in ['rgb'] + (['gray', 'haze'] if args.save_aux else []):
                save_image(prediction[name][0], output_dir / name / f'{row["id"]}.png')
            if i == 0 or (i+1) % 100 == 0 or i+1 == len(loader):
                print(f'{i+1}/{len(loader)}: {row}', flush=True)
    with open(output_dir / 'per_image.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'images': len(rows), 'split': args.split,
               'checkpoint': str(args.weights), 'checkpoint_epoch': checkpoint['epoch'],
               'mode': config['mode'], 'tile_size': args.tile_size, 'overlap': args.overlap,
               'amp': amp, 'limited_run': args.limit is not None,
               'metric_protocol': 'mean per-image; float RGB [0,1] before PNG quantization; '
                                  'no border shaving; Gaussian valid SSIM; RGB SAM degrees; ERGAS ratio=1',
               'mean': {key: sum(row[key] for row in rows)/len(rows)
                        for key in rows[0] if key != 'id'}}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--split', choices=['test', 'val', 'train'], default='test')
    parser.add_argument('--output-dir', default='/kaggle/working/results/sar_gray_color')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--tile-size', type=int, default=0, help='0: full-frame inference')
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--save-aux', action='store_true')
    parser.add_argument('--no-gt', action='store_true', help='Inference with hazy and SAR folders only')
    parser.add_argument('--amp', action='store_true', help='Optional fp16 inference on CUDA')
    parser.add_argument('--limit', type=int, help='Development subset, not a benchmark')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
