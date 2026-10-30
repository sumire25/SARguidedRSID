"""Configuration, reproducibility, and portable full-state checkpoints."""
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    return config


def validate_config(c):
    if c['mode'] not in ('hybrid', 'e2e', 'modular'):
        raise ValueError('mode must be hybrid, e2e or modular')
    t, d, m = c['train'], c['data'], c['model']
    if m['sar_channels'] != d['sar_channels']:
        raise ValueError('model.sar_channels and data.sar_channels must match')
    for key in ('gray_epochs', 'color_epochs', 'joint_epochs', 'num_workers'):
        if not isinstance(t[key], int) or t[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('batch_size', 'accumulation_steps', 'val_every'):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('max_train_batches', 'max_val_images'):
        if t[key] is not None and (not isinstance(t[key], int) or t[key] < 1):
            raise ValueError(f'{key} must be a positive integer or null')
    hours = t.get('max_hours')
    if hours is not None and (not isinstance(hours, (int, float)) or isinstance(hours, bool)
                              or not math.isfinite(hours) or hours <= 0):
        raise ValueError('max_hours must be finite and positive or null')
    if sum(t[k] for k in ('gray_epochs', 'color_epochs', 'joint_epochs')) < 1:
        raise ValueError('At least one training epoch is required')
    if not 0 <= t['sar_dropout'] < 1 or not 0 <= t['teacher_start'] <= 1:
        raise ValueError('Require sar_dropout in [0,1) and teacher_start in [0,1]')
    if not 0 < t['min_lr'] <= t['lr'] or t['gray_joint_lr_scale'] <= 0:
        raise ValueError('Require 0 < min_lr <= lr and gray_joint_lr_scale > 0')
    if t['warmup_epochs'] < 0 or t['grad_clip'] <= 0 or t['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    if c['loss']['gate_difference_scale'] <= 0:
        raise ValueError('gate_difference_scale must be > 0')
    if any(not math.isfinite(float(v)) or v < 0 for v in c['loss'].values()):
        raise ValueError('Loss weights must be finite and nonnegative')
    return c


def select_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N')
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CPU checks and reproducible sampling. Some CUDA kernels may
    # remain nondeterministic; full bitwise GPU reproducibility is not promised.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state(generator):
    np_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    generator.set_state(state['loader'].cpu())


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 1:
        raise ValueError('Expected a SARGrayColor full-state checkpoint (format_version=1)')
    return checkpoint


def assert_resume_compatible(current, saved):
    def comparable(config):
        result = copy.deepcopy(config)
        for key in ('run_dir', 'device'):
            result.pop(key, None)
        result['data'].pop('root', None)  # Kaggle mount locations can change.
        result['train'].pop('num_workers', None)
        result['train'].pop('max_hours', None)  # Per-session budget can change on resume.
        return result
    if comparable(current) != comparable(saved):
        raise ValueError('Resume settings differ from checkpoint. Use its config, or '
                         '--init-weights for a new experiment/schedule.')


def append_json(path, record):
    with open(path, 'a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
