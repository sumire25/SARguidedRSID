"""Strictly paired MRSHaze PNG loading; validation is held out from train."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def resolve_root(root):
    root = Path(root)
    if (root / 'train').is_dir() or (root / 'test').is_dir():
        return root
    if (root / 'MRSHaze').is_dir():
        return root / 'MRSHaze'
    raise FileNotFoundError(f'MRSHaze root not found: {root}. Expected train/ or test/.')


def folder(split, name):
    matches = [p for p in Path(split).iterdir()
               if p.is_dir() and p.name.casefold() == name.casefold()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one {name}/ directory in {split}')
    return matches[0]


def index_pngs(directory):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() == '.png':
            if path.stem in result:
                raise ValueError(f'Duplicate image ID {path.stem} in {directory}')
            result[path.stem] = path
    if not result:
        raise ValueError(f'No PNG images found in {directory}')
    return result


def paired_records(root, split, require_gt=True):
    base = resolve_root(root) / split
    if not base.is_dir():
        raise FileNotFoundError(base)
    maps = {name: index_pngs(folder(base, name)) for name in ('hazy', 'SAR')}
    if require_gt:
        maps['GT'] = index_pngs(folder(base, 'GT'))
    expected = set(maps['hazy'])
    for name, mapping in maps.items():
        if set(mapping) != expected:
            missing = sorted(expected - set(mapping))[:5]
            extra = sorted(set(mapping) - expected)[:5]
            raise ValueError(f'Unpaired {split}/{name}: missing IDs {missing}; extra IDs {extra}')
    return [{'id': key, **{k: str(v[key]) for k, v in maps.items()}}
            for key in sorted(expected)]


def make_split(records, val_fraction, seed, manifest=None):
    ids = [r['id'] for r in records]
    if len(ids) < 2 or not 0 < val_fraction < 1:
        raise ValueError('At least two training images and 0 < val_fraction < 1 required')
    if manifest is None:
        shuffled = ids.copy()
        random.Random(seed).shuffle(shuffled)
        count = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
        val_ids = set(shuffled[:count])
        manifest = {'seed': seed, 'train_ids': sorted(set(ids) - val_ids),
                    'val_ids': sorted(val_ids), 'source_split': 'train'}
    train_ids, val_ids = manifest['train_ids'], manifest['val_ids']
    if (len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids)
            or set(train_ids) & set(val_ids)
            or set(train_ids) | set(val_ids) != set(ids)
            or not train_ids or not val_ids or manifest.get('source_split') != 'train'):
        raise ValueError('Split manifest is inconsistent with the training dataset')
    lookup = {r['id']: r for r in records}
    return [lookup[k] for k in train_ids], [lookup[k] for k in val_ids], manifest


def load_rgb(path):
    with Image.open(path) as image:
        if image.mode not in ('RGB', 'RGBA', 'L', 'P'):
            raise ValueError(f'Expected an 8-bit optical PNG, got {image.mode} at {path}')
        array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def load_sar(path, channels=1, scale=None, log1p=False):
    with Image.open(path) as image:
        array = np.asarray(image)
    # PNG may store a one-channel rendering in RGB. Preserve selected channels
    # for multichannel SAR instead of silently discarding a polarization.
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] not in (1, 2, 3, 4):
        raise ValueError(f'Unsupported SAR PNG shape {array.shape} at {path}')
    dtype = array.dtype
    if scale is None:
        if not np.issubdtype(dtype, np.unsignedinteger):
            raise ValueError(f'Specify sar_scale for non-unsigned SAR data: {path}, {dtype}')
        scale = float(np.iinfo(dtype).max)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('sar_scale must be finite and positive')
    array = array.astype(np.float32) / scale
    if channels == 1 and array.shape[-1] in (3, 4):
        array = array[..., :3].mean(axis=-1, keepdims=True)
    elif array.shape[-1] != channels:
        raise ValueError(f'Expected {channels} SAR channels, got {array.shape[-1]} at {path}')
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1 + 1e-6:
        raise ValueError(f'SAR outside [0,1] after scaling at {path}; check sar_scale')
    if log1p:
        array = np.log1p(array) / np.log(2.0)
    return torch.from_numpy(array.transpose(2, 0, 1).astype(np.float32).copy())


class MRSHazeDataset(Dataset):
    def __init__(self, records, crop_size=None, augment=False, sar_channels=1,
                 sar_scale=None, sar_log1p=False):
        self.records = records
        self.crop_size = crop_size
        self.augment = augment
        self.sar_options = (sar_channels, sar_scale, sar_log1p)
        if not records:
            raise ValueError('Empty dataset')
        if crop_size is not None and crop_size < 8:
            raise ValueError('crop_size must be >= 8 or null')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = {'hazy': load_rgb(record['hazy']),
                  'sar': load_sar(record['SAR'], *self.sar_options)}
        if 'GT' in record:
            images['gt'] = load_rgb(record['GT'])
        shapes = {tuple(x.shape[-2:]) for x in images.values()}
        if len(shapes) != 1:
            raise ValueError(f'Unregistered triplet {record["id"]}: spatial shapes {shapes}')
        h, w = images['hazy'].shape[-2:]
        if self.crop_size is not None:
            size = self.crop_size
            # Apply precisely the same padding/crop/dihedral transform to all modalities.
            if h < size or w < size:
                images = {k: F.pad(x, (0, max(0, size-w), 0, max(0, size-h)),
                                   mode='replicate') for k, x in images.items()}
                h, w = max(size, h), max(size, w)
            top = random.randint(0, h-size) if self.augment else (h-size)//2
            left = random.randint(0, w-size) if self.augment else (w-size)//2
            images = {k: x[:, top:top+size, left:left+size] for k, x in images.items()}
        if self.augment:
            turns, flip = random.randrange(4), random.randrange(2)
            images = {k: torch.rot90(x, turns, (-2, -1)) for k, x in images.items()}
            if flip:
                images = {k: x.flip(-1) for k, x in images.items()}
        return {**images, 'id': record['id']}


def dataset_options(config):
    return {k: config['data'][k] for k in ('sar_channels', 'sar_scale', 'sar_log1p')}


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, indent=2) + '\n')
